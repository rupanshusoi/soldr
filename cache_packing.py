import json
from dataclasses import dataclass, replace
from itertools import islice
from math import ceil
from typing import Literal, TypedDict
from ortools.sat.python import cp_model
from calibrate import (MockerLatency, cache_lanes, compare, compute_usage, mocker_recs, peak_working_set, shared_tokens,
                       swimlane)

Objective = Literal["mean"] | float
Expr = cp_model.LinearExpr | int

@dataclass
class Request:
    isl: int
    osl: int
    arrival: int | None
    hash_ids: list[int]
    prev: int | None = None
    think: int = 0

@dataclass
class Trace:
    requests: list[Request]
    blk_size: int

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

@dataclass
class Loader:
    ms_per_step: int

    def mooncake(self, path: str, n: int | None = None) -> Trace:
        with open(path) as f:
            rows = [json.loads(line) for line in islice(f, n)]
        t0 = rows[0]["timestamp"]
        step = self.ms_per_step
        trace = [Request(r["input_length"], r["output_length"], (r["timestamp"] - t0) // step, r["hash_ids"])
                 for r in rows]
        return Trace(trace, 512)

    # Arrival timestamps for Weka are ignored, we use only think_time. Session starts are staggered.
    def weka(self, path: str, sessions: int, n: int | None = None, stagger_s: float = 0,
             idle_cap_s: float = 300) -> Trace:
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
        trace: list[Request] = []
        est: list[float] = []
        offset = 0
        for s, session in enumerate(chosen):
            t = s * stagger_s
            for k, r in enumerate(session["requests"]):
                # Cap think time. This appears to be a standard transformation for
                # Weka and helps decrease the horizon for us.
                think = 0 if k == 0 else min(r["think_time"], idle_cap_s)
                t += think
                est.append(t)
                arrival = round(s * stagger_s * 1000) // self.ms_per_step if k == 0 else None
                prev = None if k == 0 else len(trace) - 1
                trace.append(Request(r["in"], max(1, r["out"]), arrival, [h + offset for h in r["hash_ids"]], prev,
                                     round(think * 1000) // self.ms_per_step))
                t += r["api_time"]
            offset += 1 + max(h for r in session["requests"] for h in r["hash_ids"])

        assert n is None or len(trace) >= n
        order = sorted(range(len(trace)), key=lambda i: est[i])[:n]
        pos = {i: k for k, i in enumerate(order)}
        prev_of = lambda u: None if u.prev is None else pos[u.prev]
        return Trace([replace(trace[i], prev=prev_of(trace[i])) for i in order], blk_sizes.pop())

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

    def log_stats(self) -> None:
        cfg, reqs = self.cfg, self.trace.requests
        print("Total requests: ", len(reqs))
        # Sessions that survived the truncation due to num_requests
        total_sessions = sum(u.prev is None for u in reqs)
        print("Total sessions: ", total_sessions)
        print(f"Avg session length: {len(reqs) / total_sessions:.1f} requests")
        print(f"Session stagger: {cfg.stagger_s} s")
        recs = mocker_recs(reqs, self.trace.blk_size, cfg.num_caches, self.latency)
        peak_working_set(reqs, recs, cfg.num_caches, self.latency)
        compute_usage(recs, cfg.num_caches)

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
    def compare(self, sol: Solution) -> dict[str, list[dict]]:
        return compare(self.trace.requests, self.trace.blk_size, sol.solver, sol.vars, self.latency, self.cfg.objective)

    def visualize(self, schedules: dict[str, list[dict]]) -> None:
        reqs, blk_size = self.trace.requests, self.trace.blk_size
        swimlane(reqs, blk_size, schedules, f"swimlane_{self.name}.png", self.latency)
        cache_lanes(reqs, blk_size, schedules, f"caches_{self.name}.png", self.latency)

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
