import json
from dataclasses import dataclass
from math import ceil
from typing import Any, TypedDict
from ortools.sat.python import cp_model
from mocker import Mocker, MockerLatency, MockerUsage, Objective, soldr_ttft, summarize
from plot import Plotter, Schedules
from workload import Loader, Request, Trace, shared_tokens

Expr = cp_model.LinearExpr | int

@dataclass
class Config:
    params: str = "llama3.1-405b_h200x8"
    trace: str = "traces/weka_256k.jsonl"
    num_caches: int = 4
    sessions_per_cache: int = 4
    num_requests: int | None = None
    stagger_s: float = 5
    objective: Objective = "mean"
    horizon: int = 10_000_000
    # KV capacity of params if None
    mem: int | None = None

    @property
    def sessions(self) -> int:
        return self.sessions_per_cache * self.num_caches

    def name(self, trace: Trace) -> str:
        model = self.params.split("-")[0]
        return f"{self.num_caches}c_{self.sessions}s_{len(trace.requests)}r_{self.stagger_s}st_{model}"

class Vars(TypedDict):
    routes: list[list[cp_model.IntVar]]
    starts: list[cp_model.IntVar]
    ends: list[cp_model.IntVar]
    prefill_ends: list[cp_model.IntVar]
    hits: list[cp_model.IntVar]
    releases: list[Expr]
    ttfts: list[cp_model.IntVar]
    quantile: cp_model.IntVar | None

# Reduction in prefill time of v if it gets a cache hit from u
def saving(u: Request, v: Request, blk_size: int, prefill_latency: MockerLatency) -> int:
    return prefill_latency(v.isl) - prefill_latency(v.isl, shared_tokens(u, v, blk_size))

def sc_size(u: Request) -> int:
    return u.isl + u.osl

class GapLogger(cp_model.CpSolverSolutionCallback):
    def on_solution_callback(self) -> None:
        obj, bound = self.ObjectiveValue(), self.BestObjectiveBound()
        print(f"[{self.WallTime():.1f}s] obj={obj:.0f} bound={bound:.0f} gap={(obj - bound) / max(abs(obj), 1):.1%}")

@dataclass
class Solution:
    solver: cp_model.CpSolver
    status: cp_model.CpSolverStatus
    vars: Vars

    @property
    def found(self) -> bool:
        return self.status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

class Soldr:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.latency = MockerLatency.load(cfg.params)
        loader = Loader(self.latency.decode_ms)
        # self.trace = loader.mooncake("traces/mooncake_conversation.jsonl", n=128)
        self.trace = loader.weka(cfg.trace, sessions=cfg.sessions, n=cfg.num_requests, stagger_s=cfg.stagger_s)
        self.name = cfg.name(self.trace)
        self.mocker = Mocker(self.latency, self.trace.blk_size)

    def log_stats(self) -> None:
        cfg, reqs = self.cfg, self.trace.requests
        print("Total requests: ", len(reqs))
        # Sessions that survived the truncation due to num_requests
        total_sessions = sum(u.prev is None for u in reqs)
        print("Total sessions: ", total_sessions)
        print(f"Avg session length: {len(reqs) / total_sessions:.1f} requests")
        print(f"Session stagger: {cfg.stagger_s} s")
        usage = MockerUsage(self.trace, cfg.num_caches, self.latency)
        usage.peak_working_set()
        usage.compute_usage()

    def build_model(self) -> tuple[cp_model.CpModel, Vars]:
        reqs, blk_size, prefill_latency = self.trace.requests, self.trace.blk_size, self.latency
        horizon, objective = self.cfg.horizon, self.cfg.objective
        mem = self.latency.kv_capacity if self.cfg.mem is None else self.cfg.mem
        N, C = len(reqs), self.cfg.num_caches
        assert mem >= max([u.isl + u.osl for u in reqs])
        for u in reqs:
            assert ceil(u.isl / blk_size) == len(u.hash_ids)

        model = cp_model.CpModel()

        routes = [[model.new_bool_var(f"route[{i},{c}]") for c in range(C)] for i in range(N)]

        # For symmetry breaking
        for j in range(min(N, C)):
            for c in range(j + 1, C):
                model.add(routes[j][c] == False)

        for i, u in enumerate(reqs):
            model.add_exactly_one(routes[i][c] for c in range(C))

        starts = [model.new_int_var(0, horizon, f"start[{i}]") for i in range(N)]
        ends = [model.new_int_var(0, horizon, f"end[{i}]") for i in range(N)]
        prefill_ends = [model.new_int_var(1, horizon, f"prefill_end[{i}]") for i in range(N)]

        # For Mooncake a request is released at its arrival timestamp as given in
        # the trace. For Weka, it's released think_time after the end of its
        # predecessor.
        releases: list[Expr] = []
        for u in reqs:
            if u.prev is None:
                assert u.arrival is not None
                releases.append(u.arrival)
            else:
                releases.append(ends[u.prev] + u.think)
        for i in range(N):
            model.add(starts[i] >= releases[i])

        # The SC is not directly present in the constraints. For a given session,
        # when a request is running, the SC is just its cache usage. When no request
        # is running, we persist the last request's cache usage until the next
        # request starts.

        # hits[j] means j gets a full cache hit from its SC
        hits = [model.new_constant(0)] * N
        intervals: list[list[cp_model.IntervalVar]] = [[] for c in range(C)]
        sizes: list[list[int]] = [[] for c in range(C)]
        for j, v in enumerate(reqs):
            if v.prev is None or saving(reqs[v.prev], v, blk_size, prefill_latency) == 0:
                continue
            hits[j] = model.new_bool_var(f"hit[{j}]")
            gap = model.new_int_var(v.think, horizon, f"gap[{j}]")
            model.add(gap == starts[j] - ends[v.prev])

            # keeps[c] means the SC waits for j on c
            keeps = [model.new_bool_var(f"keep[{j},{c}]") for c in range(C)]
            model.add(sum(keeps) == hits[j])
            for c in range(C):
                model.add_implication(keeps[c], routes[v.prev][c])
                model.add_implication(keeps[c], routes[j][c])
                sc = model.new_optional_interval_var(ends[v.prev], gap, starts[j], keeps[c], f"sc[{j},{c}]")
                intervals[c].append(sc)
                sizes[c].append(sc_size(reqs[v.prev]))

        prefills: list[list[cp_model.IntervalVar]] = [[] for c in range(C)]
        for j, v in enumerate(reqs):
            full = prefill_latency(v.isl)
            prefill: Expr
            if v.prev is None:
                prefill = full
            else:
                prefill = model.new_int_var(1, full, f"prefill[{j}]")
                model.add(prefill == full - hits[j] * saving(reqs[v.prev], v, blk_size, prefill_latency))
            duration = prefill + v.osl
            model.add(prefill_ends[j] == starts[j] + prefill)
            for c in range(C):
                intervals[c].append(model.new_optional_interval_var(starts[j], duration, ends[j], routes[j][c], f"lifetime[{j},{c}]"))
                sizes[c].append(v.isl + v.osl)
                prefills[c].append(model.new_optional_interval_var(starts[j], prefill, prefill_ends[j], routes[j][c], f"prefill[{j},{c}]"))

        for c in range(C):
            model.add_no_overlap(prefills[c])
            model.add_cumulative(intervals[c], sizes[c], mem)

        ttfts = []
        for i, u in enumerate(reqs):
            ttft = model.new_int_var(0, horizon, f"ttft[{i}]")
            # Weird, we can equivalently write ttft = prefill_ends[i] - releases[i] but that confuses the solver...
            model.add(ttft == ends[i] - u.osl - releases[i])
            ttfts.append(ttft)

        quantile = None
        if objective == "mean":
            model.minimize(sum(ttfts))
        else:
            k = ceil(objective * N)
            quantile = model.new_int_var(0, horizon, "quantile")
            lates = [model.new_bool_var(f"late[{i}]") for i in range(N)]
            for ttft, late in zip(ttfts, lates):
                model.add(ttft <= quantile).OnlyEnforceIf(late.Not())
            model.add(sum(lates) <= N - k)
            model.minimize(quantile)

        return model, Vars(routes=routes, starts=starts, ends=ends, prefill_ends=prefill_ends, hits=hits,
                           releases=releases, ttfts=ttfts, quantile=quantile)

    def solve(self, num_workers: int = 0) -> Solution:
        cfg = self.cfg
        model, vars = self.build_model()
        print("Built model, beginning solve...")
        solver = cp_model.CpSolver()
        solver.parameters.num_workers = num_workers
        sol = Solution(solver, solver.Solve(model, GapLogger()), vars)
        print(solver.StatusName(sol.status))

        if sol.found:
            if cfg.objective == "mean":
                ttfts = vars["ttfts"]
                print("Avg TTFT: ", sum(solver.Value(t) for t in ttfts) / len(ttfts))
            else:
                assert vars["quantile"] is not None
                print(f"P{cfg.objective * 100:g} TTFT: ", solver.Value(vars["quantile"]))

            if sol.status != cp_model.OPTIMAL:
                obj, bound = solver.ObjectiveValue(), solver.BestObjectiveBound()
                print(f"Gap: {(obj - bound) / max(abs(obj), 1):.1%}")
        return sol

    # Replays Soldr's schedule and Dynamo's own in the Mocker
    def compare(self, sol: Solution, mocker_runs: int = 3) -> Schedules:
        solver, vars, mocker, objective = sol.solver, sol.vars, self.mocker, self.cfg.objective
        reqs, dt = self.trace.requests, self.latency.decode_ms
        rows = mocker.rows(reqs, sessions=False)
        route = [next(c for c, r in enumerate(rs) if solver.Value(r)) for rs in vars["routes"]]
        submit = [solver.Value(s) * dt for s in vars["starts"]]
        release = [solver.Value(r) * dt for r in vars["releases"]]
        soldr_mocker = [0.0] * len(rows)
        # Soldr's schedule as Mocker replays it. Mocker's clock starts at each worker's first submission.
        soldr: list[dict[str, Any]] = [{} for _ in rows]
        for c in set(route):
            ix = sorted((j for j in range(len(rows)) if route[j] == c), key=lambda j: (submit[j], j))
            for j, q in zip(ix, mocker.run([rows[j] | {"timestamp": submit[j]} for j in ix], 1)):
                soldr_mocker[j] = submit[j] - release[j] + q["ttft_ms"]
                t0 = submit[j] - q["arrival_time_ms"]
                soldr[j] = dict(worker=c, release=release[j], start=t0 + q["first_admit_ms"],
                                prefill_end=t0 + q["first_token_ms"], end=t0 + q["terminal_time_ms"],
                                reused=q["reused_input_tokens"])
        schedules = {"Soldr": soldr}

        rows = mocker.rows(reqs)

        score = lambda recs: (soldr_ttft(rows, recs, self.latency, objective),
                              summarize([q["ttft_ms"] / dt for q in recs], objective))
        from dynamo._core import KvRouterConfig
        # Mean (Soldr model, Mocker) score over mocker_runs replays. The first replay is kept as the schedule to plot.
        def dynamo(name: str, **kwargs: Any) -> tuple[str, float, float]:
            runs = [mocker.run(rows, self.cfg.num_caches, **kwargs) for _ in range(mocker_runs)]
            schedules[name] = [dict(worker=q["decode_worker_idx"], release=q["arrival_time_ms"],
                                    start=q["first_admit_ms"], prefill_end=q["first_token_ms"],
                                    end=q["terminal_time_ms"], reused=q["reused_input_tokens"]) for q in runs[0]]
            scores = [score(recs) for recs in runs]
            return name, sum(s for s, _ in scores) / mocker_runs, sum(m for _, m in scores) / mocker_runs
        soldr_soldr = summarize([solver.Value(t) for t in vars["ttfts"]], objective)
        table = [("Soldr", soldr_soldr, summarize([t / dt for t in soldr_mocker], objective)),
                 dynamo("Dynamo"),
                 dynamo("Dynamo (w/o active tracking)", router_config=KvRouterConfig(router_track_active_blocks=False))]
        label = "Avg TTFT" if objective == "mean" else f"P{objective * 100:g} TTFT"
        print(f"{label:28s} {'Soldr':>7s} {'Mocker':>7s}")
        for name, model_ttft, mocker_ttft in table:
            print(f"{name:28s} {model_ttft:7.1f} {mocker_ttft:7.1f}")
        return schedules

    def visualize(self, schedules: Schedules) -> None:
        plotter = Plotter(self.trace, schedules, self.latency)
        plotter.swimlane(f"swimlane_{self.name}.png")
        plotter.cache_lanes(f"caches_{self.name}.png")

    def dump_schedule(self, sol: Solution) -> None:
        reqs, blk_size = self.trace.requests, self.trace.blk_size
        solver, vars, value = sol.solver, sol.vars, sol.solver.Value
        succ = {v.prev: j for j, v in enumerate(reqs) if v.prev is not None}
        requests = []
        for j, v in enumerate(reqs):
            src = v.prev if value(vars["hits"][j]) else None
            # The SC that j leaves is dropped when j ends, unless its successor uses it
            kept = j in succ and value(vars["hits"][succ[j]])
            requests.append(dict(
                id=j, cache=next(c for c, r in enumerate(vars["routes"][j]) if value(r)), prev=v.prev, think=v.think,
                isl=v.isl, osl=v.osl, release=value(vars["releases"][j]), start=value(vars["starts"][j]),
                prefill_end=value(vars["prefill_ends"][j]), end=value(vars["ends"][j]), src=src,
                hit=0 if src is None else shared_tokens(reqs[src], v, blk_size), prefill_full=self.latency(v.isl),
                ttft=value(vars["ttfts"][j]), evict=value(vars["starts"][succ[j]]) if kept else None))
        path = f"schedule_{self.name}.json"
        with open(path, "w") as f:
            json.dump(dict(params=self.latency.config, decode_ms=self.latency.decode_ms, blk_size=blk_size,
                           status=solver.StatusName(sol.status), obj=solver.ObjectiveValue(),
                           bound=solver.BestObjectiveBound(), requests=requests), f, indent=1)
        print("\nWrote", path)

    def run(self) -> None:
        self.log_stats()
        sol = self.solve()
        if sol.found:
            self.visualize(self.compare(sol))
            self.dump_schedule(sol)

if __name__ == "__main__":
    Soldr(Config()).run()
