import json
from math import ceil
from ortools.sat.python import cp_model
from cache_packing import default_latency, from_mooncake, from_weka, saving, solve
from calibrate import cache_lanes, compare, compute_usage, mocker_recs, peak_working_set, shared_tokens, swimlane

# Tokens the session cache holds once u has finished
def sc_size(u):
    return u.isl + u.osl

def build_model(trace, blk_size, num_caches, mem, horizon, prefill_latency, objective="mean"):
    N, C = len(trace), num_caches
    assert mem >= max([u.isl + u.osl for u in trace])
    for u in trace:
        assert ceil(u.isl / blk_size) == len(u.hash_ids)

    model = cp_model.CpModel()

    routes = [[model.NewBoolVar(f"route[{i},{c}]") for c in range(C)] for i in range(N)]

    # For symmetry breaking
    for j in range(min(N, C)):
        for c in range(j + 1, C):
            model.Add(routes[j][c] == False)

    for i, u in enumerate(trace):
        model.AddExactlyOne(routes[i][c] for c in range(C))

    starts = [model.NewIntVar(0, horizon, f"start[{i}]") for i in range(N)]
    ends = [model.NewIntVar(0, horizon, f"end[{i}]") for i in range(N)]
    prefill_ends = [model.NewIntVar(1, horizon, f"prefill_end[{i}]") for i in range(N)]

    # For Mooncake a request is released at its arrival timestamp as given in
    # the trace. For Weka, it's released think_time after the end of its
    # predecessor.
    releases = [u.arrival if u.prev is None else ends[u.prev] + u.think for u in trace]
    for i in range(N):
        model.Add(starts[i] >= releases[i])

    # The session cache (SC) of a session is what its latest request left behind. While a request runs, the SC is part
    # of the request's own footprint. Between the end of a request and the start of its successor, the SC either stays
    # on the worker, occupying memory, or is dropped.
    #
    # hits[j] means the SC is on j's worker when j starts, so j gets a full cache hit. Otherwise j pays full prefill.
    hits = [model.NewConstant(0)] * N
    intervals = [[] for c in range(C)]
    sizes = [[] for c in range(C)]
    for j, v in enumerate(trace):
        if v.prev is None or saving(trace[v.prev], v, blk_size, prefill_latency) == 0:
            continue
        hits[j] = model.NewBoolVar(f"hit[{j}]")
        # The successor starts after its release, so the gap is never negative
        gap = model.NewIntVar(v.think, horizon, f"gap[{j}]")
        model.Add(gap == starts[j] - ends[v.prev])

        # keeps[c] means the SC waits for j on c, which needs both j and its predecessor on c
        keeps = [model.NewBoolVar(f"keep[{j},{c}]") for c in range(C)]
        model.Add(sum(keeps) == hits[j])
        for c in range(C):
            model.AddImplication(keeps[c], routes[v.prev][c])
            model.AddImplication(keeps[c], routes[j][c])
            intervals[c].append(model.NewOptionalIntervalVar(ends[v.prev], gap, starts[j], keeps[c], f"sc[{j},{c}]"))
            sizes[c].append(sc_size(trace[v.prev]))

    prefills = [[] for c in range(C)]
    for j, v in enumerate(trace):
        full = prefill_latency(v.isl)
        if v.prev is None:
            prefill = full
        else:
            prefill = model.NewIntVar(1, full, f"prefill[{j}]")
            model.Add(prefill == full - hits[j] * saving(trace[v.prev], v, blk_size, prefill_latency))
        duration = prefill + v.osl
        model.Add(prefill_ends[j] == starts[j] + prefill)
        for c in range(C):
            intervals[c].append(model.NewOptionalIntervalVar(starts[j], duration, ends[j], routes[j][c], f"lifetime[{j},{c}]"))
            sizes[c].append(v.isl + v.osl)
            prefills[c].append(model.NewOptionalIntervalVar(starts[j], prefill, prefill_ends[j], routes[j][c], f"prefill[{j},{c}]"))

    for c in range(C):
        model.AddNoOverlap(prefills[c])
        # Running requests plus waiting SCs must fit in memory
        model.AddCumulative(intervals[c], sizes[c], mem)

    ttfts = []
    for i, u in enumerate(trace):
        ttft = model.NewIntVar(0, horizon, f"ttft[{i}]")
        # Weird, we can equivalently write ttft = prefill_ends[i] - releases[i] but that confuses the solver...
        model.Add(ttft == ends[i] - u.osl - releases[i])
        ttfts.append(ttft)

    quantile = None
    if objective == "mean":
        model.minimize(sum(ttfts))
    else:
        k = ceil(objective * N)
        quantile = model.NewIntVar(0, horizon, "quantile")
        lates = [model.NewBoolVar(f"late[{i}]") for i in range(N)]
        for ttft, late in zip(ttfts, lates):
            model.Add(ttft <= quantile).OnlyEnforceIf(late.Not())
        model.Add(sum(lates) <= N - k)
        model.minimize(quantile)

    return model, dict(routes=routes, starts=starts, ends=ends, prefill_ends=prefill_ends, hits=hits, releases=releases,
                       ttfts=ttfts, quantile=quantile)

def dump_schedule(config, trace, blk_size, solver, status, vars, prefill_latency):
    value = solver.Value
    succ = {v.prev: j for j, v in enumerate(trace) if v.prev is not None}
    requests = []
    for j, v in enumerate(trace):
        hit = bool(value(vars["hits"][j]))
        # The SC that j leaves is dropped when j ends, unless its successor uses it
        kept = j in succ and value(vars["hits"][succ[j]])
        requests.append(dict(
            id=j, cache=next(c for c, r in enumerate(vars["routes"][j]) if value(r)), prev=v.prev, think=v.think,
            isl=v.isl, osl=v.osl, release=value(vars["releases"][j]), start=value(vars["starts"][j]),
            prefill_end=value(vars["prefill_ends"][j]), end=value(vars["ends"][j]), src=v.prev if hit else None,
            hit=shared_tokens(trace[v.prev], v, blk_size) if hit else 0, prefill_full=prefill_latency(v.isl),
            ttft=value(vars["ttfts"][j]), evict=value(vars["starts"][succ[j]]) if kept else None))
    path = f"schedule_{config}.json"
    with open(path, "w") as f:
        json.dump(dict(params=prefill_latency.config, decode_ms=prefill_latency.decode_ms, blk_size=blk_size,
                       status=solver.StatusName(status), obj=solver.ObjectiveValue(), bound=solver.BestObjectiveBound(),
                       requests=requests), f, indent=1)
    print("\nWrote", path)

if __name__ == "__main__":
    # num_requests = 128
    # trace, blk_size = from_mooncake("traces/mooncake_conversation.jsonl", n=num_requests)

    num_requests = None
    num_caches = 4
    sessions = 4 * num_caches
    # 900 s is the average session length for Weka
    stagger_s = 5
    objective = "mean"

    trace, blk_size = from_weka("traces/weka_256k.jsonl", sessions=sessions, n=num_requests, stagger_s=stagger_s)
    print("Total requests: ", len(trace))
    # Sessions that survived the truncation due to num_requests
    total_sessions = sum(u.prev is None for u in trace)
    print("Total sessions: ", total_sessions)
    print(f"Avg session length: {len(trace) / total_sessions:.1f} requests")
    print(f"Session stagger: {stagger_s} s")
    recs = mocker_recs(trace, blk_size, num_caches, default_latency)
    peak_working_set(trace, recs, num_caches, default_latency)
    compute_usage(recs, num_caches)
    model, vars = build_model(trace, blk_size, num_caches=num_caches, mem=default_latency.kv_capacity, horizon=10_000_000,
                              prefill_latency=default_latency, objective=objective)
    print("Built model, beginning solve...")
    solver, status = solve(model, vars, objective)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        config = f"session_{num_caches}c_{sessions}s_{len(trace)}r_{stagger_s}st_{default_latency.config.split('-')[0]}"
        schedules = compare(trace, blk_size, solver, vars, default_latency, objective)
        swimlane(trace, blk_size, schedules, f"swimlane_{config}.png", default_latency)
        cache_lanes(trace, blk_size, schedules, f"caches_{config}.png", default_latency)
        dump_schedule(config, trace, blk_size, solver, status, vars, default_latency)
