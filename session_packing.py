import json
import sys
from dataclasses import dataclass
from heapq import heappop, heappush
from ortools.sat.python import cp_model
from cache_packing import GapLogger, default_latency, from_weka, shared_tokens
from calibrate import run_mocker, summarize

# Cache packing at the level of sessions instead of requests, for Weka-style traces where requests form think-time
# chains and only share prefixes within their session. The only decision is which cache serves each session (and
# whether to evict it once), so the model has no per-request variables. Each session's timeline is fixed to the one
# it would have without contention, which gives every session a fixed KV usage profile per cache.
#
# Ignored relative to cache_packing: queueing delays don't shift a session's later requests, and contention between
# sessions is a pairwise estimate rather than a NoOverlap schedule.


def sessions_of(trace):
    head, groups = [], {}
    for j, u in enumerate(trace):
        head.append(j if u.prev is None else head[u.prev])
        groups.setdefault(head[j], []).append(j)
    return list(groups.values())


@dataclass
class Plan:
    evict: int | None              # the session's KV is evicted after its request k, before request k + 1
    prefills: list[tuple[int, int]]  # (start, end) of each request's prefill
    ttfts: list[int]
    profile: list[tuple[int, int, int]]  # (start, end, tokens) the session keeps resident


# Uncontended timeline of one session. Each request gets its largest hit from an earlier request still resident,
# and a block stays resident until the last request that hits on it starts.
def plan(trace, js, blk_size, prefill_latency, evict=None):
    starts, ends, prefills, ttfts, live, events = [], [], [], [], {}, {}
    for k, j in enumerate(js):
        v = trace[j]
        t = v.arrival if k == 0 else ends[-1] + v.think
        lo = 0 if evict is None or k <= evict else evict + 1
        hit, src = max(((shared_tokens(trace[js[i]], v, blk_size), i) for i in range(lo, k)), default=(0, None))
        p = prefill_latency(v.isl, hit)
        starts.append(t)
        ends.append(t + p + v.osl)
        prefills.append((t, t + p))
        ttfts.append(p)
        for h in v.hash_ids:
            live.setdefault(h, []).append((t, ends[-1]))
        if hit:
            for h in trace[js[src]].hash_ids[:hit // blk_size]:
                live[h].append((ends[src], t))
        events[t] = events.get(t, 0) + v.osl
        events[ends[-1]] = events.get(ends[-1], 0) - v.osl

    for spans in live.values():
        spans.sort()
        lo, hi = spans[0]
        for a, b in spans[1:] + [(float("inf"), None)]:
            if a > hi:
                events[lo] = events.get(lo, 0) + blk_size
                events[hi] = events.get(hi, 0) - blk_size
                lo, hi = a, b
            else:
                hi = max(hi, b)

    profile, level, times = [], 0, sorted(events)
    for a, b in zip(times, times[1:]):
        level += events[a]
        if level > 0:
            profile.append((a, b, level))
    return Plan(evict, prefills, ttfts, profile)


# FIFO prefill delay one session's requests impose on another's if they share a cache
def contention(pa, pb):
    cost = 0
    for sa, ea in pa.prefills:
        for sb, eb in pb.prefills:
            if sa <= sb < ea:
                cost += ea - sb
            elif sb < sa < eb:
                cost += eb - sa
    return cost


def build_model(trace, blk_size, num_caches, mem, prefill_latency, max_evictions=0, evict_candidates=3):
    groups, C = sessions_of(trace), num_caches

    # Eviction candidates are each session's longest think gaps, where holding its KV costs the most
    plans = []
    for js in groups:
        gaps = sorted(range(len(js) - 1), key=lambda k: -trace[js[k + 1]].think)[:evict_candidates if max_evictions else 0]
        plans.append([plan(trace, js, blk_size, prefill_latency, g) for g in [None] + sorted(gaps)])
    S = len(groups)

    model = cp_model.CpModel()
    # assign[s][c][o]: session s runs on cache c with eviction option o (o = 0 is no eviction)
    assign = [[[model.NewBoolVar(f"assign[{s},{c},{o}]") for o in range(len(plans[s]))] for c in range(C)]
              for s in range(S)]
    routes = [[model.NewBoolVar(f"route[{s},{c}]") for c in range(C)] for s in range(S)]
    for s in range(S):
        model.AddExactlyOne(x for xs in assign[s] for x in xs)
        for c in range(C):
            model.Add(routes[s][c] == sum(assign[s][c]))
    # Symmetry breaking: session s (in order of arrival) only uses the first s + 1 caches
    for s in range(min(S, C)):
        for c in range(s + 1, C):
            for x in assign[s][c]:
                model.Add(x == 0)

    model.Add(sum(x for s in range(S) for c in range(C) for x in assign[s][c][1:]) <= max_evictions)

    for c in range(C):
        intervals, demands = [], []
        for s in range(S):
            for o, p in enumerate(plans[s]):
                for a, b, tokens in p.profile:
                    intervals.append(model.NewOptionalFixedSizeIntervalVar(a, b - a, assign[s][c][o], f"kv[{s},{c},{o},{a}]"))
                    demands.append(tokens)
        model.AddCumulative(intervals, demands, mem)

    # same[a][b]: sessions a and b share a cache, which costs the pairwise contention estimate
    costs = {(a, b): contention(plans[a][0], plans[b][0]) for a in range(S) for b in range(a + 1, S)}
    same = {}
    for (a, b), cost in costs.items():
        if cost > 0:
            same[a, b] = model.NewBoolVar(f"same[{a},{b}]")
            for c in range(C):
                model.AddBoolOr([routes[a][c].Not(), routes[b][c].Not(), same[a, b]])

    # Sum of estimated TTFTs
    model.minimize(sum(sum(p.ttfts) * assign[s][c][o] for s in range(S) for c in range(C) for o, p in enumerate(plans[s]))
                   + sum(costs[k] * x for k, x in same.items()))
    return model, dict(groups=groups, plans=plans, assign=assign, same=same)


def solve(model, num_workers=0, time_limit=None):
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = num_workers
    if time_limit:
        solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model, GapLogger())
    print(solver.StatusName(status))
    return solver, status


def extract(solver, vars):
    route, evicts = [], []
    for s, xs in enumerate(vars["assign"]):
        c, o = next((c, o) for c in range(len(xs)) for o in range(len(xs[c])) if solver.Value(xs[c][o]))
        route.append(c)
        evicts.append(vars["plans"][s][o].evict)
    return route, evicts


# Soldr's cost model under a session routing, with FIFO prefill per cache and contention shifting later requests
def simulate(trace, groups, route, blk_size, prefill_latency):
    ttfts = [0] * len(trace)
    for c in set(route):
        heap = [(trace[js[0]].arrival, s, 0) for s, js in enumerate(groups) if route[s] == c]
        free = 0
        while heap:
            release, s, k = heappop(heap)
            js = groups[s]
            v = trace[js[k]]
            hit = max((shared_tokens(trace[i], v, blk_size) for i in js[:k]), default=0)
            start = max(release, free)
            free = start + prefill_latency(v.isl, hit)
            ttfts[js[k]] = free - release
            if k + 1 < len(js):
                heappush(heap, (free + v.osl + trace[js[k + 1]].think, s, k + 1))
    return ttfts


def agentic_rows(trace, js, prefill_latency):
    dt, head = prefill_latency.decode_ms, {}
    rows = []
    for j in js:
        u = trace[j]
        head[j] = j if u.prev is None else head[u.prev]
        deps = [] if u.prev is None else [dict(request_id=str(u.prev), trigger="completion", delay_ms=u.think * dt,
                                               relation="sequence")]
        rows.append(dict(timestamp=(u.arrival or 0) * dt, input_length=u.isl, output_length=u.osl, hash_ids=u.hash_ids,
                         request_id=str(j), play_id=str(head[j]), session_id=str(head[j]),
                         model=prefill_latency.aic["model"], not_before_ms=float((u.arrival or 0) * dt),
                         dependencies=deps))
    return rows


# Closed-loop Mocker replay. Sessions never share KV, so pinning a session routing is exact with one Mocker per cache.
def replay_routing(trace, groups, route, blk_size, prefill_latency):
    ttfts = [0.0] * len(trace)
    for c in set(route):
        js = sorted(j for s in range(len(groups)) if route[s] == c for j in groups[s])
        for j, q in zip(js, run_mocker(agentic_rows(trace, js, prefill_latency), 1, blk_size, prefill_latency)):
            ttfts[j] = q["ttft_ms"] / prefill_latency.decode_ms
    return ttfts


def replay_kv_router(trace, num_caches, blk_size, prefill_latency):
    recs = run_mocker(agentic_rows(trace, range(len(trace)), prefill_latency), num_caches, blk_size, prefill_latency)
    return [q["ttft_ms"] / prefill_latency.decode_ms for q in recs], [q["decode_worker_idx"] for q in recs]


def stats(ttfts):
    return " ".join(f"{name}={summarize(ttfts, q):7.1f}" for name, q in
                    [("mean", "mean"), ("p50", 0.5), ("p90", 0.9), ("p99", 0.99)])


if __name__ == "__main__":
    # Same instance as cache_packing.py
    num_requests = 512
    num_caches = 4
    sessions = 16 * num_caches
    stagger_s = 900 // sessions
    mem = default_latency.kv_capacity // int(sys.argv[1]) if len(sys.argv) > 1 else default_latency.kv_capacity

    trace, blk_size = from_weka("traces/weka_256k.jsonl", sessions=sessions, n=num_requests, stagger_s=stagger_s)
    print("Total requests: ", len(trace), "mem: ", mem)

    # Outer loop over the eviction budget: double it until feasible, then one more to see if it helps
    best, first = None, None
    for max_evictions in [0, 1, 2, 4, 8, 16, 32, 64]:
        model, vars = build_model(trace, blk_size, num_caches, mem, default_latency, max_evictions=max_evictions)
        print(f"\nmax_evictions={max_evictions}: {len(vars['groups'])} sessions, {len(vars['same'])} contention pairs")
        solver, status = solve(model, time_limit=60)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print(f"Est. avg TTFT {solver.ObjectiveValue() / len(trace):.1f}, bound {solver.BestObjectiveBound() / len(trace):.1f}")
            first = max_evictions if first is None else first
            if best is None or solver.ObjectiveValue() < best[0]:
                best = (solver.ObjectiveValue(), *extract(solver, vars))
        if first is not None and max_evictions >= 2 * first + 1:
            break

    groups = vars["groups"]
    _, route, evicts = best
    print("\nSessions per cache:", [route.count(c) for c in range(num_caches)])
    print("Evictions:", [(s, k) for s, k in enumerate(evicts) if k is not None])

    dt = default_latency.decode_ms
    rr = [s % num_caches for s in range(len(groups))]
    print(f"\nTTFT in timesteps ({dt} ms)")
    print(f"{'session_packing (Soldr sim)':32s}", stats(simulate(trace, groups, route, blk_size, default_latency)))
    print(f"{'session_packing (Mocker)':32s}", stats(replay_routing(trace, groups, route, blk_size, default_latency)))
    print(f"{'sticky round robin (Soldr sim)':32s}", stats(simulate(trace, groups, rr, blk_size, default_latency)))
    print(f"{'sticky round robin (Mocker)':32s}", stats(replay_routing(trace, groups, rr, blk_size, default_latency)))
    for run in range(3):
        ttfts, workers = replay_kv_router(trace, num_caches, blk_size, default_latency)
        moved = sum(trace[j].prev is not None and workers[j] != workers[trace[j].prev] for j in range(len(trace)))
        print(f"{f'Mocker kv_router run {run}':32s}", stats(ttfts), f"(session switched worker {moved} times)")
