import json
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from math import ceil
import numpy as np

# Must be set before dynamo is first imported
os.environ.setdefault("DYN_LOG", "warn")


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


def replay(rows, workers, block_size, p, **kwargs):
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
        return run_trace_replay(path, extra_engine_args=mocker_engine(p), num_workers=workers, router_mode="kv_router",
                                trace_block_size=block_size, capture_per_request=True, **extra, **kwargs)


def run_mocker(rows, workers, block_size, p, **kwargs):
    agentic = "dependencies" in rows[0]
    report = replay(rows, workers, block_size, p, **kwargs)
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


def mocker_rows(trace, prefill_latency, sessions=True):
    dt = prefill_latency.decode_ms
    rows = [{"timestamp": (u.arrival or 0) * dt, "input_length": u.isl, "output_length": u.osl, "hash_ids": u.hash_ids}
            for u in trace]
    if not sessions or all(u.prev is None for u in trace):
        return rows

    session = []
    for j, u in enumerate(trace):
        session.append(j if u.prev is None else session[u.prev])
    deps = lambda u: [] if u.prev is None else [dict(request_id=str(u.prev), trigger="completion",
                                                     delay_ms=u.think * dt, relation="sequence")]
    return [r | dict(request_id=str(j), play_id=str(session[j]), session_id=str(session[j]),
                     # First turns are released at their arrival, later turns by their dependency only
                     model=prefill_latency.aic["model"], not_before_ms=float(r["timestamp"]),
                     dependencies=deps(u))
            for j, (r, u) in enumerate(zip(rows, trace))]


def compare(trace, blk_size, solver, vars, prefill_latency, objective="mean", mocker_runs=3):
    dt = prefill_latency.decode_ms
    rows = mocker_rows(trace, prefill_latency, sessions=False)
    route = [next(c for c, r in enumerate(rs) if solver.Value(r)) for rs in vars["routes"]]
    submit = [solver.Value(s) * dt for s in vars["starts"]]
    release = [solver.Value(r) * dt for r in vars["releases"]]
    soldr_mocker = [0.0] * len(rows)
    # Soldr's schedule as Mocker replays it. Mocker's clock starts at each worker's first submission.
    schedules = {"Soldr": [None] * len(rows)}
    for c in set(route):
        ix = sorted((j for j in range(len(rows)) if route[j] == c), key=lambda j: (submit[j], j))
        for j, q in zip(ix, run_mocker([rows[j] | {"timestamp": submit[j]} for j in ix], 1, blk_size, prefill_latency)):
            soldr_mocker[j] = submit[j] - release[j] + q["ttft_ms"]
            t0 = submit[j] - q["arrival_time_ms"]
            schedules["Soldr"][j] = dict(worker=c, release=release[j], start=t0 + q["first_admit_ms"],
                                         prefill_end=t0 + q["first_token_ms"], end=t0 + q["terminal_time_ms"],
                                         reused=q["reused_input_tokens"])

    rows = mocker_rows(trace, prefill_latency)

    score = lambda recs: (soldr_ttft(rows, recs, prefill_latency, objective),
                          summarize([q["ttft_ms"] / dt for q in recs], objective))
    from dynamo._core import KvRouterConfig
    # Mean (Soldr model, Mocker) score over mocker_runs replays. The first replay is kept as the schedule to plot.
    def mocker(name, **kwargs):
        runs = [run_mocker(rows, len(vars["routes"][0]), blk_size, prefill_latency, **kwargs) for _ in range(mocker_runs)]
        schedules[name] = [dict(worker=q["decode_worker_idx"], release=q["arrival_time_ms"], start=q["first_admit_ms"],
                                prefill_end=q["first_token_ms"], end=q["terminal_time_ms"],
                                reused=q["reused_input_tokens"]) for q in runs[0]]
        runs = [score(recs) for recs in runs]
        return name, sum(s for s, _ in runs) / mocker_runs, sum(m for _, m in runs) / mocker_runs
    soldr_soldr = summarize([solver.Value(t) for t in vars["ttfts"]], objective)
    table = [("Soldr", soldr_soldr, summarize([t / dt for t in soldr_mocker], objective)),
             mocker("Mocker"),
             mocker("Mocker (w/o active tracking)", router_config=KvRouterConfig(router_track_active_blocks=False))]
    label = "Avg TTFT" if objective == "mean" else f"P{objective * 100:g} TTFT"
    print(f"{label:28s} {'Soldr':>7s} {'Mocker':>7s}")
    for name, soldr, mock in table:
        print(f"{name:28s} {soldr:7.1f} {mock:7.1f}")
    return schedules


def shared_tokens(u, v, blk_size):
    count = 0
    for b1, b2 in zip(u.hash_ids, v.hash_ids):
        if b1 != b2:
            break
        count += 1
    return min(count * blk_size, u.isl, v.isl)


def swimlane(trace, blk_size, schedules, path, prefill_latency, dpi=300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    session = []
    for j, u in enumerate(trace):
        session.append(j if u.prev is None else session[u.prev])
    # Lanes in order of first appearance in the trace, which is by estimated release
    lane = {s: k for k, s in enumerate(dict.fromkeys(session))}
    y = [lane[s] for s in session]
    earlier = [[i for i in range(j) if session[i] == session[j]] for j in range(len(trace))]
    shared = [{i: shared_tokens(trace[i], v, blk_size) for i in earlier[j]} for j, v in enumerate(trace)]
    achievable = [max(s.values(), default=0) for s in shared]

    workers = 1 + max(q["worker"] for recs in schedules.values() for q in recs)
    colors = plt.get_cmap("tab10" if workers <= 10 else "tab20").colors
    plt.style.use("dark_background")
    fig, axes = plt.subplots(len(schedules), 1, sharex=True, sharey=True, squeeze=False,
                             figsize=(16, len(schedules) * max(3, 0.6 + 0.14 * len(lane))))
    for ax, (name, recs) in zip(axes[:, 0], schedules.items()):
        sec = lambda key: [q[key] / 1000 for q in recs]
        release, start, prefill_end, end = sec("release"), sec("start"), sec("prefill_end"), sec("end")
        color = [colors[q["worker"] % len(colors)] for q in recs]
        width = lambda a, b: [t1 - t0 for t0, t1 in zip(a, b)]
        ax.barh(y, width(release, start), left=release, height=0.25, color="0.6", zorder=1)
        ax.barh(y, width(prefill_end, end), left=prefill_end, height=0.8, color=color, alpha=0.35, lw=0, zorder=2)
        ax.barh(y, width(start, prefill_end), left=start, height=0.8, color=color, lw=0, zorder=3)

        miss = {"routing": [], "eviction": []}
        switches = 0
        for j, q in enumerate(recs):
            switches += trace[j].prev is not None and recs[trace[j].prev]["worker"] != q["worker"]
            if achievable[j] - q["reused"] > blk_size:
                local = any(shared[j][i] == achievable[j] and recs[i]["worker"] == q["worker"] for i in earlier[j])
                miss["eviction" if local else "routing"].append(j)
        ax.scatter([start[j] for j in miss["routing"]], [y[j] for j in miss["routing"]], marker="x", s=30, lw=1.2,
                   color="#ff4d4d", zorder=4)
        ax.scatter([start[j] for j in miss["eviction"]], [y[j] for j in miss["eviction"]], marker="o", s=30, lw=1.2,
                   facecolors="none", edgecolors="white", zorder=4)
        reuse = sum(q["reused"] for q in recs) / max(1, sum(achievable))
        ttft = sum(width(release, prefill_end)) * 1000 / prefill_latency.decode_ms / len(recs)
        title = ax.set_title(name, fontsize=10, fontweight="bold", loc="left")
        ax.annotate(f"mean TTFT {ttft:.1f} steps, {reuse:.0%} of achievable reuse, {len(miss['routing'])} routing misses, "
                    f"{len(miss['eviction'])} eviction misses, {switches} worker switches", xy=(1, 0), xycoords=title,
                    xytext=(8, 0), textcoords="offset points", va="bottom", fontsize=10)
        ax.set_ylabel("session (by first request)")
        ax.grid(axis="x", color="0.25", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    ax.set_ylim(len(lane) - 0.4, -0.6)
    # Lanes are numbered from 0, so no ticks in the padding above the first lane
    step = max(1, ceil(len(lane) / 20))
    ax.set_yticks(range(0, len(lane), step))
    ax.set_xlabel("time (s)")
    handles = [Patch(color=colors[w % len(colors)], label=f"worker {w}") for w in range(workers)] + [
        Patch(color="0.8", label="prefill"), Patch(color="0.8", alpha=0.35, lw=0, label="decode"),
        Line2D([], [], color="0.6", lw=3, label="waiting to start"),
        Line2D([], [], color="#ff4d4d", marker="x", ls="", label="routing miss"),
        Line2D([], [], markerfacecolor="none", markeredgecolor="white", marker="o", ls="", label="eviction miss")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / fig.get_figheight()))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print("Wrote", path)


# Mocker's per-request records for the trace, under its own kv_router decisions
def mocker_recs(trace, blk_size, num_caches, prefill_latency):
    return run_mocker(mocker_rows(trace, prefill_latency), num_caches, blk_size, prefill_latency)


def peak_working_set(trace, recs, num_caches, prefill_latency):
    last = {}
    for j, u in enumerate(trace):
        if u.prev is not None:
            last.pop(u.prev)
        last[j] = True

    events = []
    for j, (u, q) in enumerate(zip(trace, recs)):
        held = 0 if u.prev is None else trace[u.prev].isl + trace[u.prev].osl
        events.append((q["arrival_time_ms"], u.isl + u.osl - held))
        if j in last:
            events.append((q["terminal_time_ms"], -(u.isl + u.osl)))

    live = peak = 0
    for _, delta in sorted(events):
        live += delta
        peak = max(peak, live)
    peak /= num_caches * prefill_latency.kv_capacity
    print(f"Peak working set: {peak:.1%} of mem")
    return peak


def covered(intervals):
    total, end = 0, float("-inf")
    for a, b in sorted(intervals):
        total += max(0, b - max(a, end))
        end = max(end, b)
    return total


def compute_usage(recs, num_caches):
    duration = max(q["terminal_time_ms"] for q in recs) - min(q["arrival_time_ms"] for q in recs)
    on = lambda c, end: covered((q["first_admit_ms"], q[end]) for q in recs if q["decode_worker_idx"] == c)
    busy = sum(on(c, "terminal_time_ms") for c in range(num_caches)) / (num_caches * duration)
    prefill = sum(on(c, "first_token_ms") for c in range(num_caches)) / (num_caches * duration)
    print(f"Avg compute usage: {busy:.1%} ({prefill:.1%} prefill, {busy - prefill:.1%} decode only)")
    return busy, prefill


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
