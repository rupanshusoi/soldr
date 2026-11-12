import json
from dataclasses import dataclass, replace
from itertools import islice
from math import ceil
from ortools.sat.python import cp_model
from calibrate import MockerLatency, compare, compute_usage, mocker_recs, peak_working_set, shared_tokens, swimlane

@dataclass
class Request:
    isl: int
    osl: int
    arrival: int | None
    hash_ids: list[int]
    prev: int | None = None
    think: int = 0

default_latency = MockerLatency.load("llama3.1-405b_h200x8")

def from_mooncake(path, n=None, ms_per_step=default_latency.decode_ms):
    with open(path) as f:
        rows = [json.loads(line) for line in islice(f, n)]
    t0 = rows[0]["timestamp"]
    trace = [Request(r["input_length"], r["output_length"], (r["timestamp"] - t0) // ms_per_step, r["hash_ids"]) for r in rows]
    return trace, 512

# Arrival timestamps for Weka are ignored, we use only think_time. Session starts are staggered.
def from_weka(path, sessions, n=None, stagger_s=0, ms_per_step=default_latency.decode_ms, idle_cap_s=300):
    chosen = []
    with open(path) as f:
        for line in f:
            session = json.loads(line)
            # Ignore sessions with subagents
            if all(r["type"] != "subagent" for r in session["requests"]):
                chosen.append(session)
            if len(chosen) == sessions:
                break
    assert len(chosen) == sessions, f"only {len(chosen)} sessions without subagents"
    blk_sizes = {p["block_size"] for p in chosen}
    assert len(blk_sizes) == 1

    # est is a release time estimate using the recorded API times, used to pick the first n requests
    trace, est, offset = [], [], 0
    for s, session in enumerate(chosen):
        t = s * stagger_s
        for k, r in enumerate(session["requests"]):
            # Cap think time. This appears to be a standard transformation for
            # Weka and helps decrease the horizon for us.
            think = 0 if k == 0 else min(r["think_time"], idle_cap_s)
            t += think
            est.append(t)
            arrival = round(s * stagger_s * 1000) // ms_per_step if k == 0 else None
            prev = None if k == 0 else len(trace) - 1
            trace.append(Request(r["in"], max(1, r["out"]), arrival, [h + offset for h in r["hash_ids"]], prev,
                                 round(think * 1000) // ms_per_step))
            t += r["api_time"]
        offset += 1 + max(h for r in session["requests"] for h in r["hash_ids"])

    assert n is None or len(trace) >= n
    order = sorted(range(len(trace)), key=lambda i: est[i])[:n]
    pos = {i: k for k, i in enumerate(order)}
    trace = [replace(trace[i], prev=None if trace[i].prev is None else pos[trace[i].prev]) for i in order]
    return trace, blk_sizes.pop()

# Reduction in prefill time of v if it gets a cache hit from u
def saving(u, v, blk_size, prefill_latency=default_latency):
    return prefill_latency(v.isl) - prefill_latency(v.isl, shared_tokens(u, v, blk_size))

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

    # reuses[i][j] means j gets the largest possible cache hit from i.
    # TODO: Allow fine-grained eviction
    shared = [[shared_tokens(u, v, blk_size) if i != j else 0 for j, v in enumerate(trace)] for i, u in enumerate(trace)]
    pairs = [(i, j) for i in range(N) for j in range(N) if shared[i][j] > 0]
    reuses = [[model.NewConstant(0)] * N for i in range(N)]
    for i, j in pairs:
        reuses[i][j] = model.NewBoolVar(f"reuse[{i},{j}]")

    # To get a cache hit, you must start after your provider's prefill is done
    for i, j in pairs:
        model.Add(prefill_ends[i] <= starts[j]).OnlyEnforceIf(reuses[i][j])

    # You may only get a cache hit from one provider
    for j in range(N):
        model.AddAtMostOne(reuses[i][j] for i in range(N))

    # If i provides j a cache hit and j is routed to c, then i must also be routed to c
    for i, j in pairs:
        for c in range(C):
            model.AddBoolOr([reuses[i][j].Not(), routes[j][c].Not(), routes[i][c]])

    intervals = [[] for c in range(C)]
    sizes = [[] for c in range(C)]
    prefills = [[] for c in range(C)]
    for j, v in enumerate(trace):
        # Full cache hit costs 1 timestep
        prefill = model.NewIntVar(1, prefill_latency(v.isl), f"prefill[{j}]")
        model.Add(prefill == prefill_latency(v.isl) - sum(reuses[i][j] * saving(u, v, blk_size, prefill_latency)
                                                          for i, u in enumerate(trace) if shared[i][j] > 0))
        duration = prefill + v.osl
        model.Add(prefill_ends[j] == starts[j] + prefill)
        for c in range(C):
            intervals[c].append(model.NewOptionalIntervalVar(starts[j], duration, ends[j], routes[j][c], f"lifetime[{j},{c}]"))
            sizes[c].append(v.isl + v.osl)
            prefills[c].append(model.NewOptionalIntervalVar(starts[j], prefill, prefill_ends[j], routes[j][c], f"prefill[{j},{c}]"))

    for c in range(C):
        model.AddNoOverlap(prefills[c])

    evicts, providers = [], []
    for i, u in enumerate(trace):
        # Time at which we evict u's prompt
        evict = model.NewIntVar(0, horizon, f"evict[{i}]")
        model.add(evict >= ends[i])

        # Is u providing a cache hit to anybody?
        is_provider = model.NewBoolVar(f"provider[{i}]")
        evicts.append(evict)
        providers.append(is_provider)

        for j in range(N):
            if shared[i][j] > 0:
                # If v is getting a cache hit from u, u's prompt must not be evicted until v starts
                model.Add(evict >= starts[j]).OnlyEnforceIf(reuses[i][j])
                # ... and u is a provider
                model.AddImplication(reuses[i][j], is_provider)

        _evict = model.NewIntVar(0, horizon, f"_evict[{i}]")
        for c in range(C):
            # cond == is_provider AND routes[i][c]
            # It's annoying that CP-SAT doesn't let us write SAT formulas naturally...
            cond = model.NewBoolVar(f"cond[{i},{c}]")
            model.AddBoolOr([is_provider.Not(), routes[i][c].Not(), cond])
            model.AddImplication(cond, is_provider)
            model.AddImplication(cond, routes[i][c])

            intervals[c].append(model.NewOptionalIntervalVar(ends[i], _evict, evict, cond, f"eviction[{i},{c}]"))
            # TODO: This is an overestimate I think
            sizes[c].append(max(shared[i]))

    for c in range(C):
        model.AddCumulative(intervals[c], sizes[c], mem)

    # waits = []
    # for i, u in enumerate(trace):
    #     wait = model.NewIntVar(0, horizon, f"waits[{i}]")
    #     model.Add(wait == starts[i] - u.arrival)
    #     waits.append(wait)
    # model.minimize(sum(waits))

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

    return model, dict(routes=routes, starts=starts, ends=ends, prefill_ends=prefill_ends, reuses=reuses, releases=releases,
                       evicts=evicts, providers=providers, ttfts=ttfts, quantile=quantile)

class GapLogger(cp_model.CpSolverSolutionCallback):
    def on_solution_callback(self):
        obj, bound = self.ObjectiveValue(), self.BestObjectiveBound()
        print(f"[{self.WallTime():.1f}s] obj={obj:.0f} bound={bound:.0f} gap={(obj - bound) / max(abs(obj), 1):.1%}")

def solve(model, vars, objective="mean", num_workers=0):
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = num_workers
    status = solver.Solve(model, GapLogger())
    print(solver.StatusName(status))

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if objective == "mean":
            ttfts = vars["ttfts"]
            print("Avg TTFT: ", sum(solver.Value(t) for t in ttfts) / len(ttfts))
        else:
            print(f"P{objective * 100:g} TTFT: ", solver.Value(vars["quantile"]))

        if status != cp_model.OPTIMAL:
            obj, bound = solver.ObjectiveValue(), solver.BestObjectiveBound()
            print(f"Gap: {(obj - bound) / max(abs(obj), 1):.1%}")
    return solver, status

def dump_schedule(config, trace, blk_size, solver, status, vars, prefill_latency):
    value = solver.Value
    requests = []
    for j, v in enumerate(trace):
        src = next((i for i in range(len(trace)) if value(vars["reuses"][i][j])), None)
        requests.append(dict(
            id=j, cache=next(c for c, r in enumerate(vars["routes"][j]) if value(r)), prev=v.prev, think=v.think,
            isl=v.isl, osl=v.osl, release=value(vars["releases"][j]), start=value(vars["starts"][j]),
            prefill_end=value(vars["prefill_ends"][j]), end=value(vars["ends"][j]), src=src,
            hit=0 if src is None else shared_tokens(trace[src], v, blk_size), prefill_full=prefill_latency(v.isl),
            ttft=value(vars["ttfts"][j]), evict=value(vars["evicts"][j]) if value(vars["providers"][j]) else None))
    path = f"schedule_{config}.json"
    with open(path, "w") as f:
        json.dump(dict(params=prefill_latency.config, decode_ms=prefill_latency.decode_ms, blk_size=blk_size,
                       status=solver.StatusName(status), obj=solver.ObjectiveValue(), bound=solver.BestObjectiveBound(),
                       requests=requests), f, indent=1)
    print("\nWrote", path)

if __name__ == "__main__":
    # num_requests = 128
    # trace, blk_size = from_mooncake("traces/mooncake_conversation.jsonl", n=num_requests)

    num_requests = 128
    num_caches = 4
    sessions = 48 * num_caches
    # 900 s is the average session length for Weka
    stagger_s = 900 // sessions
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
        config = f"{num_caches}c_{sessions}s_{len(trace)}r_{stagger_s}st_{default_latency.config.split('-')[0]}"
        schedules = compare(trace, blk_size, solver, vars, default_latency, objective)
        swimlane(trace, blk_size, schedules, f"swimlane_{config}.png", default_latency)
        dump_schedule(config, trace, blk_size, solver, status, vars, default_latency)
