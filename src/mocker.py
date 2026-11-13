import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil
from typing import Any, Literal, Self
from workload import Request, Trace, sessions_of

# Must be set before dynamo is first imported
os.environ.setdefault("DYN_LOG", "warn")

Objective = Literal["mean"] | float
# A trace row as fed to Mocker
Row = dict[str, Any]
# Mocker's record of a request
Rec = dict[str, Any]


@dataclass(frozen=True)
class MockerLatency:
    config: str # params_<config>.json
    decode_ms: int # one timestep
    a: float
    b: float
    c: float
    chunk: int
    kv_block: int
    kv_capacity: int
    max_context: int
    aic: dict[str, Any]

    @classmethod
    def load(cls, config: str) -> Self:
        with open(f"params_{config}.json") as f:
            return cls(**json.load(f))

    def prefill_ms(self, isl: int, hit: int = 0) -> float:
        hit = min(hit, isl - 1) // self.kv_block * self.kv_block
        return self.a * (isl - hit) + self.b * (isl ** 2 - hit ** 2) / 2 + self.c * ceil((isl - hit) / self.chunk)

    def __call__(self, isl: int, hit: int = 0) -> int:
        return max(1, round(self.prefill_ms(isl, hit) / self.decode_ms))


class Mocker:
    def __init__(self, latency: MockerLatency, blk_size: int) -> None:
        self.latency = latency
        self.blk_size = blk_size

    def engine_args(self) -> dict[str, Any]:
        p = self.latency
        return {
            "engine": {
                "backend": "vllm", "block_size": p.kv_block, "num_gpu_blocks": p.kv_capacity // p.kv_block,
                "max_num_seqs": 256, "max_num_batched_tokens": p.chunk, "enable_prefix_caching": True,
                "enable_chunked_prefill": True, "max_model_len": p.max_context,
                "timing_model": {"type": "external", "provider": "aic", "config": p.aic | {
                    "backend": "vllm",
                    "backend_version": "0.24.0", "worker_type": "aggregated", "gemm_quant_mode": "bfloat16",
                    "fmha_quant_mode": "bfloat16", "kvcache_quant_mode": "bfloat16", "kv_block_size": p.kv_block,
                    "estimation_mode": "op_level"}},
            },
            "num_gpu_blocks_is_explicit": True,
        }

    def replay(self, rows: list[Row], workers: int, **kwargs: Any) -> Any:
        from dynamo.replay import run_trace_replay

        # Rows with dependencies are sessions, replayed closed-loop in Mocker's agentic format
        agentic = "dependencies" in rows[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trace.jsonl")
            with open(path, "w") as f:
                if agentic:
                    f.write(json.dumps({"schema": "dynamo.agentic_mooncake", "version": 2, "block_size": self.blk_size,
                                        "hash_id_scope": "local", "source": {"format": "soldr", "digest": "-"}}) + "\n")
                f.writelines(json.dumps(r) + "\n" for r in rows)
            # No agentic_lanes: with lanes, Mocker measures not_before_ms from each session's start, which drops our
            # offsets
            extra = dict(trace_format="agentic_mooncake", execution_model=self.latency.aic["model"]) if agentic else {}
            return run_trace_replay(path, extra_engine_args=self.engine_args(), num_workers=workers,
                                    router_mode="kv_router", trace_block_size=self.blk_size, capture_per_request=True,
                                    **extra, **kwargs)

    # Mocker's record of each row
    def run(self, rows: list[Row], workers: int, **kwargs: Any) -> list[Rec]:
        agentic = "dependencies" in rows[0]
        report = self.replay(rows, workers, **kwargs)
        if agentic:
            by_id = {q["request_id"]: q for q in report.to_dict()["per_request"]}
            return [by_id[r["request_id"]] for r in rows]

        key = lambda arrival, isl, osl: (round(arrival), isl, osl)
        recs = defaultdict(list)
        for q in report.to_dict()["per_request"]:
            recs[key(q["arrival_time_ms"], q["input_length"], q["output_length"])].append(q)
        t0 = rows[0]["timestamp"]
        return [recs[key(r["timestamp"] - t0, r["input_length"], r["output_length"])].pop() for r in rows]

    def rows(self, reqs: list[Request], sessions: bool = True) -> list[Row]:
        dt = self.latency.decode_ms
        rows: list[Row] = [{"timestamp": (u.arrival or 0) * dt, "input_length": u.isl, "output_length": u.osl,
                            "hash_ids": u.hash_ids} for u in reqs]
        if not sessions or all(u.prev is None for u in reqs):
            return rows

        session = sessions_of(reqs)
        deps = lambda u: [] if u.prev is None else [dict(request_id=str(u.prev), trigger="completion",
                                                         delay_ms=u.think * dt, relation="sequence")]
        return [r | dict(request_id=str(j), play_id=str(session[j]), session_id=str(session[j]),
                         # First turns are released at their arrival, later turns by their dependency only
                         model=self.latency.aic["model"], not_before_ms=float(r["timestamp"]),
                         dependencies=deps(u))
                for j, (r, u) in enumerate(zip(rows, reqs))]


def summarize(ttfts: list[float], objective: Objective = "mean") -> float:
    if objective == "mean":
        return sum(ttfts) / len(ttfts)
    return sorted(ttfts)[ceil(objective * len(ttfts)) - 1]


# Mocker's decisions in Soldr's cost model
def soldr_ttft(rows: list[Row], recs: list[Rec], prefill_latency: MockerLatency,
               objective: Objective = "mean") -> float:
    free: dict[int, int] = {}
    ttfts: list[float] = []
    # In the order Mocker sent the requests (in a closed-loop replay, this depends on Mocker's own timing)
    for r, q in sorted(zip(rows, recs), key=lambda rq: rq[1]["arrival_time_ms"]):
        arrival = int(q["arrival_time_ms"]) // prefill_latency.decode_ms
        c = q["decode_worker_idx"]
        start = max(arrival, free.get(c, arrival))
        free[c] = start + prefill_latency(r["input_length"], q["reused_input_tokens"])
        ttfts.append(free[c] - arrival)
    return summarize(ttfts, objective)


# Memory and compute usage of the trace under Mocker's own kv_router decisions
class MockerUsage:
    def __init__(self, trace: Trace, num_caches: int, prefill_latency: MockerLatency) -> None:
        self.trace = trace
        self.num_caches = num_caches
        self.latency = prefill_latency
        mocker = Mocker(prefill_latency, trace.blk_size)
        self.recs = mocker.run(mocker.rows(trace.requests), num_caches)

    def peak_working_set(self) -> float:
        reqs = self.trace.requests
        last: dict[int, bool] = {}
        for j, u in enumerate(reqs):
            if u.prev is not None:
                last.pop(u.prev)
            last[j] = True

        events = []
        for j, (u, q) in enumerate(zip(reqs, self.recs)):
            held = 0 if u.prev is None else reqs[u.prev].isl + reqs[u.prev].osl
            events.append((q["arrival_time_ms"], u.isl + u.osl - held))
            if j in last:
                events.append((q["terminal_time_ms"], -(u.isl + u.osl)))

        live = peak = 0
        for _, delta in sorted(events):
            live += delta
            peak = max(peak, live)
        frac = peak / (self.num_caches * self.latency.kv_capacity)
        print(f"Peak working set: {frac:.1%} of mem")
        return frac

    @staticmethod
    def covered(intervals: Iterable[tuple[float, float]]) -> float:
        total, end = 0.0, float("-inf")
        for a, b in sorted(intervals):
            total += max(0, b - max(a, end))
            end = max(end, b)
        return total

    def compute_usage(self) -> tuple[float, float]:
        recs, C = self.recs, self.num_caches
        duration = max(q["terminal_time_ms"] for q in recs) - min(q["arrival_time_ms"] for q in recs)
        on = lambda c, end: self.covered((q["first_admit_ms"], q[end]) for q in recs if q["decode_worker_idx"] == c)
        busy = sum(on(c, "terminal_time_ms") for c in range(C)) / (C * duration)
        prefill = sum(on(c, "first_token_ms") for c in range(C)) / (C * duration)
        print(f"Avg compute usage: {busy:.1%} ({prefill:.1%} prefill, {busy - prefill:.1%} decode only)")
        return busy, prefill
