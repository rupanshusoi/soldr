import json
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from math import ceil
import numpy as np


def mocker_engine(p):
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
    aic: dict

    @classmethod
    def load(cls, config):
        with open(f"params_{config}.json") as f:
            return cls(**json.load(f))

    def prefill_ms(self, isl, hit=0):
        hit = min(hit, isl - 1) // self.kv_block * self.kv_block
        return self.a * (isl - hit) + self.b * (isl ** 2 - hit ** 2) / 2 + self.c * ceil((isl - hit) / self.chunk)

    def __call__(self, isl, hit=0):
        return max(1, round(self.prefill_ms(isl, hit) / self.decode_ms))


def run_mocker(rows, workers, block_size, p):
    os.environ.setdefault("DYN_LOG", "warn")
    from dynamo.llm import KvRouterConfig
    from dynamo.replay import run_trace_replay

    # Rows with dependencies are sessions, replayed closed-loop in Mocker's agentic format
    agentic = "dependencies" in rows[0]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "trace.jsonl")
        with open(path, "w") as f:
            if agentic:
                f.write(json.dumps({"schema": "dynamo.agentic_mooncake", "version": 2, "block_size": block_size,
                                    "hash_id_scope": "local", "source": {"format": "soldr", "digest": "-"}}) + "\n")
            f.writelines(json.dumps(r) + "\n" for r in rows)
        # No agentic_lanes: with lanes, Mocker measures not_before_ms from each session's start, which drops our offsets
        extra = dict(trace_format="agentic_mooncake", execution_model=p.aic["model"]) if agentic else {}
        report = run_trace_replay(path, extra_engine_args=mocker_engine(p), num_workers=workers, router_mode="kv_router",
                                  router_config=KvRouterConfig(router_track_active_blocks=False),
                                  trace_block_size=block_size, capture_per_request=True, **extra)
    if agentic:
        recs = {q["request_id"]: q for q in report.to_dict()["per_request"]}
        return [recs[r["request_id"]] for r in rows]

    key = lambda arrival, isl, osl: (round(arrival), isl, osl)
    recs = defaultdict(list)
    for q in report.to_dict()["per_request"]:
        recs[key(q["arrival_time_ms"], q["input_length"], q["output_length"])].append(q)
    return [recs[key(r["timestamp"] - rows[0]["timestamp"], r["input_length"], r["output_length"])].pop() for r in rows]


# objective is "mean" or a nearest-rank quantile in (0, 1), as in cache_packing.build_model
def summarize(ttfts, objective="mean"):
    if objective == "mean":
        return sum(ttfts) / len(ttfts)
    return sorted(ttfts)[ceil(objective * len(ttfts)) - 1]


# Mocker's decisions in Soldr's cost model
def soldr_ttft(rows, recs, prefill_latency, objective="mean"):
    free, ttfts = {}, []
    # In the order Mocker sent the requests (in a closed-loop replay, this depends on Mocker's own timing)
    for r, q in sorted(zip(rows, recs), key=lambda rq: rq[1]["arrival_time_ms"]):
        arrival = int(q["arrival_time_ms"]) // prefill_latency.decode_ms
        c = q["decode_worker_idx"]
        start = max(arrival, free.get(c, arrival))
        free[c] = start + prefill_latency(r["input_length"], q["reused_input_tokens"])
        ttfts.append(free[c] - arrival)
    return summarize(ttfts, objective)


def compare(trace, blk_size, solver, vars, prefill_latency, objective="mean", mocker_runs=3):
    dt = prefill_latency.decode_ms
    rows = [{"timestamp": (u.arrival or 0) * dt, "input_length": u.isl, "output_length": u.osl, "hash_ids": u.hash_ids}
            for u in trace]
    route = [next(c for c, r in enumerate(rs) if solver.Value(r)) for rs in vars["routes"]]
    submit = [solver.Value(s) * dt for s in vars["starts"]]
    release = [solver.Value(r) * dt for r in vars["releases"]]
    soldr_mocker = [0.0] * len(rows)
    for c in set(route):
        ix = sorted((j for j in range(len(rows)) if route[j] == c), key=lambda j: (submit[j], j))
        for j, q in zip(ix, run_mocker([rows[j] | {"timestamp": submit[j]} for j in ix], 1, blk_size, prefill_latency)):
            soldr_mocker[j] = submit[j] - release[j] + q["ttft_ms"]

    if any(u.prev is not None for u in trace):
        session = []
        for j, u in enumerate(trace):
            session.append(j if u.prev is None else session[u.prev])
        deps = lambda u: [] if u.prev is None else [dict(request_id=str(u.prev), trigger="completion",
                                                         delay_ms=u.think * dt, relation="sequence")]
        rows = [r | dict(request_id=str(j), play_id=str(session[j]), session_id=str(session[j]),
                         # First turns are released at their arrival, later turns by their dependency only
                         model=prefill_latency.aic["model"], not_before_ms=float(r["timestamp"]),
                         dependencies=deps(u))
                for j, (r, u) in enumerate(zip(rows, trace))]

    score = lambda recs: (soldr_ttft(rows, recs, prefill_latency, objective),
                          summarize([q["ttft_ms"] / dt for q in recs], objective))
    mocker = [score(run_mocker(rows, len(vars["routes"][0]), blk_size, prefill_latency)) for _ in range(mocker_runs)]
    soldr_soldr = summarize([solver.Value(t) for t in vars["ttfts"]], objective)
    table = [("Soldr decisions", soldr_soldr, summarize([t / dt for t in soldr_mocker], objective)),
             ("Mocker decisions", sum(s for s, _ in mocker) / mocker_runs, sum(m for _, m in mocker) / mocker_runs)]
    label = "Avg TTFT" if objective == "mean" else f"P{objective * 100:g} TTFT"
    print(f"{label:18s} {'Soldr':>7s} {'Mocker':>7s}")
    for name, soldr, mock in table:
        print(f"{name:18s} {soldr:7.1f} {mock:7.1f}")


def calibrate_ab(p):
    chunk = p.chunk
    rows, next_hash = [], 0
    for isl in [512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072, 196608, 245760]:
        for frac in [0, 0.25, 0.5, 0.75, 0.95]:
            hit, hashes = int(isl * frac) // 512 * 512, list(range(next_hash, next_hash + ceil(isl / 512)))
            next_hash += len(hashes)
            if hit:
                rows.append({"timestamp": 60_000 * len(rows), "input_length": hit, "output_length": 1,
                             "hash_ids": hashes[:hit // 512]})
            rows.append({"timestamp": 60_000 * len(rows), "input_length": isl, "output_length": 1, "hash_ids": hashes})
    x, y = [], []
    for r, q in zip(rows, run_mocker(rows, 1, 512, p)):
        new, hit = r["input_length"] - q["reused_input_tokens"], q["reused_input_tokens"]
        x.append([new, (r["input_length"] ** 2 - hit ** 2) / 2, ceil(new / chunk)])
        y.append(q["ttft_ms"])
    x, y = np.array(x), np.array(y)
    (a, b, _), *_ = np.linalg.lstsq(x / y[:, None], np.ones(len(y)), rcond=None)
    return float(a), float(b)


def calibrate(path, workers, config):
    p = MockerLatency.load(config)
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    recs = run_mocker(rows, workers, 512, p)
    target = sum(q["ttft_ms"] for q in recs) / len(recs)
    decode_ms = round(sum(q["last_token_ms"] - q["first_token_ms"] for q in recs)
                      / sum(q["output_length"] - 1 for q in recs))
    a, b = calibrate_ab(p)
    p = replace(p, decode_ms=decode_ms, a=a, b=b)
    # Bisect for c
    lo, hi = 0.0, 100.0
    for _ in range(20):
        p = replace(p, c=(lo + hi) / 2)
        if soldr_ttft(rows, recs, p) * decode_ms < target:
            lo = p.c
        else:
            hi = p.c
    p = replace(p, c=round((lo + hi) / 2, 1))
    with open(f"params_{config}.json", "w") as f:
        json.dump(asdict(p), f, indent=1)
    print(f"{p} (Mocker mean TTFT {target:.1f} ms)")
