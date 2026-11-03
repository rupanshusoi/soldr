import argparse
from dataclasses import dataclass, replace
from heapq import heapify, heappop, heappush
import numpy as np
from ortools.sat.python import cp_model
from cache_packing import GapLogger, default_latency, from_weka, shared_tokens
from calibrate import run_mocker
from session_packing import plan, replay_kv_router, replay_routing, sessions_of, stats

# Like session_packing, whole sessions are routed to a cache, but each request gets a start time so it can queue when
# memory binds. A follow-up request either holds its session's prefix in cache through the think gap and gets the hit,
# or lets it be evicted when its source finishes and pays a full prefill. Unlike cache_packing, the hit source is fixed
# to the best earlier request in the session, so there are no reuse-pair variables.


@dataclass
class Req:
    session: int
    src: int | None  # earlier request in the session that provides the hit
    hit: int
    p_hit: int
    p_full: int


def requests_of(trace, groups, blk_size, prefill_latency):
    reqs = [None] * len(trace)
    for s, js in enumerate(groups):
        for k, j in enumerate(js):
            v = trace[j]
            hit, src = max(((shared_tokens(trace[i], v, blk_size), i) for i in js[:k]), default=(0, None))
            reqs[j] = Req(s, src if hit else None, hit, prefill_latency(v.isl, hit), prefill_latency(v.isl))
    return reqs


# Sessions in arrival order go to the cache where they raise the peak of the uncontended KV profiles the least
def greedy_route(groups, plans, num_caches, bucket=1000):
    horizon = max(b for p in plans for _, b, _ in p.profile) // bucket + 1
    load, route = np.zeros((num_caches, horizon)), []
    for p in plans:
        prof = np.zeros(horizon)
        for a, b, tokens in p.profile:
            prof[a // bucket:b // bucket + 1] = np.maximum(prof[a // bucket:b // bucket + 1], tokens)
        lo, hi = p.profile[0][0] // bucket, p.profile[-1][1] // bucket + 1
        c = min(range(num_caches), key=lambda c: ((load[c, lo:hi] + prof[lo:hi]).max(), route.count(c)))
        load[c] += prof
        route.append(c)
    return route


# FIFO prefill per cache. A request that doesn't fit evicts other sessions' held prefixes, latest next use first, and
# otherwise waits for a running request to finish. The result satisfies build_model's constraints.
def greedy_schedule(trace, groups, reqs, route, mem):
    N = len(trace)
    start, hold, end = [0] * N, [False] * N, [None] * N
    consumers = [[] for _ in range(N)]
    for j, r in enumerate(reqs):
        if r.src is not None:
            consumers[r.src].append(j)
    pos = {j: (s, k) for s, js in enumerate(groups) for k, j in enumerate(js)}

    def next_use(k):
        prev = trace[k].prev
        return float("inf") if end[prev] is None else end[prev] + trace[k].think

    for c in set(route):
        ready = [(trace[js[0]].arrival, js[0]) for s, js in enumerate(groups) if route[s] == c]
        heapify(ready)
        running, holds = [], {}
        used = free = now = 0

        def evict(target, keep=None):
            nonlocal used
            for k in sorted(holds, key=next_use, reverse=True):
                if used <= target:
                    break
                if k != keep:
                    used -= holds.pop(k)

        def finish():
            nonlocal used
            e, i = heappop(running)
            used -= trace[i].isl + trace[i].osl
            for k in consumers[i]:
                holds[k] = reqs[k].hit
                used += reqs[k].hit
            evict(mem)
            s, k = pos[i]
            if k + 1 < len(groups[s]):
                heappush(ready, (e + trace[groups[s][k + 1]].think, groups[s][k + 1]))
            return e

        while ready or running:
            if not ready:
                finish()
                continue
            release, j = ready[0]
            t = max(release, free, now)
            if running and running[0][0] <= t:
                finish()
                continue
            v = trace[j]
            need = v.isl + v.osl - holds.get(j, 0)
            evict(mem - need, keep=j)
            if used + need > mem:
                now = max(now, finish())
                continue
            heappop(ready)
            hold[j] = j in holds
            used += v.isl + v.osl - holds.pop(j, 0)
            p = reqs[j].p_hit if hold[j] else reqs[j].p_full
            start[j], free, end[j] = t, t + p, t + p + v.osl
            heappush(running, (end[j], j))
    return start, hold


def build_model(trace, groups, reqs, num_caches, mem, horizon, hint=None):
    N, S, C = len(trace), len(groups), num_caches
    model = cp_model.CpModel()

    routes = [[model.NewBoolVar(f"route[{s},{c}]") for c in range(C)] for s in range(S)]
    for s in range(S):
        model.AddExactlyOne(routes[s])
    # Symmetry breaking: session s (in order of arrival) only uses the first s + 1 caches
    for s in range(min(S, C)):
        for c in range(s + 1, C):
            model.Add(routes[s][c] == 0)

    starts = [model.NewIntVar(0, horizon, f"start[{j}]") for j in range(N)]
    prefill_ends = [model.NewIntVar(0, horizon, f"prefill_end[{j}]") for j in range(N)]
    holds = [model.NewBoolVar(f"hold[{j}]") if r.src is not None else 0 for j, r in enumerate(reqs)]
    prefills = [r.p_full - (r.p_full - r.p_hit) * holds[j] for j, r in enumerate(reqs)]
    ends = [prefill_ends[j] + v.osl for j, v in enumerate(trace)]
    releases = [v.arrival if v.prev is None else ends[v.prev] + v.think for v in trace]
    for j in range(N):
        model.Add(prefill_ends[j] == starts[j] + prefills[j])
        model.Add(starts[j] >= releases[j])

    prefill_ivs, kv_ivs, kv_demands = [[] for _ in range(C)], [[] for _ in range(C)], [[] for _ in range(C)]
    hold_lens, held = [None] * N, [[None] * C for _ in range(N)]
    for j, (v, r) in enumerate(zip(trace, reqs)):
        for c in range(C):
            lit = routes[r.session][c]
            prefill_ivs[c].append(model.NewOptionalIntervalVar(starts[j], prefills[j], prefill_ends[j], lit,
                                                               f"prefill[{j},{c}]"))
            kv_ivs[c].append(model.NewOptionalIntervalVar(starts[j], prefills[j] + v.osl, ends[j], lit, f"run[{j},{c}]"))
            kv_demands[c].append(v.isl + v.osl)
        if r.src is None:
            continue
        # The source's prefix stays resident from when the source finishes until this request starts
        hold_lens[j] = model.NewIntVar(0, horizon, f"hold_len[{j}]")
        model.Add(starts[j] == ends[r.src] + hold_lens[j])
        for c in range(C):
            held[j][c] = model.NewBoolVar(f"held[{j},{c}]")
            model.AddBoolOr([holds[j].Not(), routes[r.session][c].Not(), held[j][c]])
            model.AddImplication(held[j][c], holds[j])
            model.AddImplication(held[j][c], routes[r.session][c])
            kv_ivs[c].append(model.NewOptionalIntervalVar(ends[r.src], hold_lens[j], starts[j], held[j][c],
                                                          f"hold[{j},{c}]"))
            kv_demands[c].append(r.hit)

    for c in range(C):
        model.AddNoOverlap(prefill_ivs[c])
        model.AddCumulative(kv_ivs[c], kv_demands[c], mem)

    ttfts = [prefill_ends[j] - releases[j] for j in range(N)]
    model.minimize(sum(ttfts))

    if hint is not None:
        route, start, hold = hint
        for s in range(S):
            for c in range(C):
                model.AddHint(routes[s][c], route[s] == c)
        for j, r in enumerate(reqs):
            p = r.p_hit if hold[j] else r.p_full
            model.AddHint(starts[j], start[j])
            model.AddHint(prefill_ends[j], start[j] + p)
            if r.src is not None:
                rs = reqs[r.src]
                src_end = start[r.src] + (rs.p_hit if hold[r.src] else rs.p_full) + trace[r.src].osl
                model.AddHint(holds[j], hold[j])
                model.AddHint(hold_lens[j], start[j] - src_end)
                for c in range(C):
                    model.AddHint(held[j][c], hold[j] and route[r.session] == c)

    return model, dict(routes=routes, starts=starts, holds=holds, prefill_ends=prefill_ends, releases=releases,
                       ttfts=ttfts)


# Releases and TTFTs of a schedule in Soldr's cost model
def timeline(trace, reqs, start, hold):
    release, ttft, end = [], [], []
    for j, (v, r) in enumerate(zip(trace, reqs)):
        prefill_end = start[j] + (r.p_hit if hold[j] else r.p_full)
        release.append(v.arrival if v.prev is None else end[v.prev] + v.think)
        ttft.append(prefill_end - release[j])
        end.append(prefill_end + v.osl)
    return release, ttft


# Mocker serves each cache's requests open loop at the solver's start times, so queueing is Soldr's decision
def replay_schedule(trace, reqs, route, start, hold, blk_size, prefill_latency):
    dt, ttfts = prefill_latency.decode_ms, [0.0] * len(trace)
    release, _ = timeline(trace, reqs, start, hold)
    for c in set(route):
        ix = sorted((j for j, r in enumerate(reqs) if route[r.session] == c), key=lambda j: (start[j], j))
        rows = [dict(timestamp=start[j] * dt, input_length=trace[j].isl, output_length=trace[j].osl,
                     hash_ids=trace[j].hash_ids) for j in ix]
        for j, q in zip(ix, run_mocker(rows, 1, blk_size, prefill_latency)):
            ttfts[j] = start[j] - release[j] + q["ttft_ms"] / dt
    return ttfts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", type=int, default=4)
    ap.add_argument("--sessions", type=int, default=128)
    ap.add_argument("--requests", type=int, default=4096, help="0 for whole sessions")
    ap.add_argument("--stagger", type=float, default=None, help="seconds between session starts, default 900 / sessions")
    ap.add_argument("--mem-div", type=float, default=1, help="divide each cache's KV capacity by this")
    ap.add_argument("--time-limit", type=float, default=120)
    ap.add_argument("--kv-router-runs", type=int, default=1)
    args = ap.parse_args()

    C = args.caches
    stagger_s = 900 // args.sessions if args.stagger is None else args.stagger
    mem = int(default_latency.kv_capacity / args.mem_div)
    latency = replace(default_latency, kv_capacity=mem)

    trace, blk_size = from_weka("traces/weka_256k.jsonl", sessions=args.sessions, n=args.requests or None,
                                stagger_s=stagger_s)
    groups = sessions_of(trace)
    reqs = requests_of(trace, groups, blk_size, latency)
    plans = [plan(trace, js, blk_size, latency) for js in groups]
    print(f"Total requests: {len(trace)}, sessions: {len(groups)}, mem: {mem}")

    route0 = greedy_route(groups, plans, C)
    start0, hold0 = greedy_schedule(trace, groups, reqs, route0, mem)
    _, greedy = timeline(trace, reqs, start0, hold0)
    print(f"Greedy: avg TTFT {sum(greedy) / len(greedy):.1f}, evicted {sum(r.src is not None and not h for r, h in zip(reqs, hold0))}"
          f" of {sum(r.src is not None for r in reqs)} hits")

    horizon = 2 * max(s + r.p_full + v.osl for s, r, v in zip(start0, reqs, trace))
    model, vars = build_model(trace, groups, reqs, C, mem, horizon, hint=(route0, start0, hold0))
    print("Built model, beginning solve...")
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.time_limit
    status = solver.Solve(model, GapLogger())
    print(solver.StatusName(status))
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    print(f"Soldr avg TTFT {solver.ObjectiveValue() / len(trace):.1f}, bound {solver.BestObjectiveBound() / len(trace):.1f}")

    route = [next(c for c in range(C) if solver.Value(xs[c])) for xs in vars["routes"]]
    start = [solver.Value(x) for x in vars["starts"]]
    hold = [bool(solver.Value(x)) for x in vars["holds"]]
    print("Sessions per cache:", [route.count(c) for c in range(C)],
          f"evicted {sum(r.src is not None and not h for r, h in zip(reqs, hold))} hits,"
          f" queued {sum(s > r for s, r in zip(start, (solver.Value(x) for x in vars['releases'])))} requests")

    rr = [s % C for s in range(len(groups))]
    print(f"\nTTFT in timesteps ({latency.decode_ms} ms)")
    print(f"{'uncontended lower bound':34s}", stats([t for p in plans for t in p.ttfts]))
    print(f"{'session_queueing (Soldr)':34s}", stats(timeline(trace, reqs, start, hold)[1]))
    print(f"{'session_queueing (Mocker, starts)':34s}", stats(replay_schedule(trace, reqs, route, start, hold, blk_size, latency)))
    print(f"{'session_queueing routing (Mocker)':34s}", stats(replay_routing(trace, groups, route, blk_size, latency)))
    print(f"{'greedy hint (Soldr)':34s}", stats(greedy))
    print(f"{'greedy routing (Mocker)':34s}", stats(replay_routing(trace, groups, route0, blk_size, latency)))
    print(f"{'sticky round robin (Mocker)':34s}", stats(replay_routing(trace, groups, rr, blk_size, latency)))
    for run in range(args.kv_router_runs):
        ttfts, workers = replay_kv_router(trace, C, blk_size, latency)
        moved = sum(trace[j].prev is not None and workers[j] != workers[trace[j].prev] for j in range(len(trace)))
        print(f"{f'Mocker kv_router run {run}':34s}", stats(ttfts), f"(session switched worker {moved} times)")
