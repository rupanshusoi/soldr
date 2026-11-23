import json
import random
import time
from dataclasses import dataclass
from math import ceil
from typing import Any, Protocol
from ortools.sat.python import cp_model
import soldr
from mocker import Mocker, MockerLatency, MockerUsage
from plot import Plotter, Schedules
from sim import Event, Live, Sim, Timing
from soldr import Expr, saving
from workload import Loader, Trace, sessions_of

@dataclass
class Config(soldr.Config):
    sessions_per_cache: int = 4
    num_requests: int | None = 64
    # The solver sees this many unsent turns of each session, and the sessions that start within lookahead_s
    lookahead_turns: int = 2
    lookahead_s: float = 60
    # Time limit of one solve. The simulated clock is stopped during a solve.
    solve_s: float = 2
    solver_workers: int = 8
    # KV router tie breaks are random, so it gets this many seeds
    router_runs: int = 5
    # Seconds between progress lines of the online solver's run
    log_s: float = 5
    # Also replays both KV routers in Mocker this many times, to check the simulator
    mocker_runs: int = 0

class Policy(Protocol):
    # Called whenever something happened. Returns when to be called again at the latest.
    def wake(self, loop: "Loop", events: list[Event]) -> float | None: ...
    # For the progress log
    def progress(self) -> str: ...

# Sends a trace through the simulator closed-loop, with a policy that decides where and when each request runs
class Loop:
    def __init__(self, trace: Trace, p: MockerLatency, workers: int, timing: Timing) -> None:
        self.reqs, self.p, self.dt = trace.requests, p, p.decode_ms
        self.sim = Sim(trace, p, workers, timing)
        self.succ = {u.prev: j for j, u in enumerate(self.reqs) if u.prev is not None}
        # In ms. Turns after the first are released when their predecessor ends.
        self.release = {j: float(u.arrival * self.dt) for j, u in enumerate(self.reqs) if u.arrival is not None}
        self.unsent = set(range(len(self.reqs)))

    # Released requests that are yet to be sent, oldest first
    def waiting(self) -> list[int]:
        now = self.sim.now
        return sorted((j for j in self.unsent if self.release.get(j, now + 1) <= now), key=lambda j: (self.release[j], j))

    def send(self, j: int, c: int) -> None:
        self.unsent.remove(j)
        self.sim.submit(j, c, self.release[j])

    # Prints a progress line every log_s seconds if given
    def run(self, policy: Policy, log_s: float | None = None) -> list[dict[str, Any]]:
        sim, done = self.sim, 0
        t0 = logged = time.time()
        timer = policy.wake(self, [])
        while done < len(self.reqs):
            if log_s is not None and time.time() - logged >= log_s:
                logged = time.time()
                print(f"[{logged - t0:4.0f}s] {done} requests done{policy.progress()}", flush=True)
            times = [t for j in self.unsent if (t := self.release.get(j, 0)) > sim.now]
            if timer is not None and timer > sim.now:
                times.append(timer)
            until = min(times, default=float("inf"))
            events = sim.step(until)
            assert events or until < float("inf"), "nothing is running and nothing is due"
            for e in events:
                if e.kind == "end":
                    done += 1
                    if (v := self.succ.get(e.id)) is not None:
                        self.release[v] = sim.now + self.reqs[v].think * self.dt
            timer = policy.wake(self, events)
        return [sim.recs[j] for j in range(len(self.reqs))]

# Dynamo's KV router with its default config: the cost of a worker is the prefill it would have queued, less what the
# request finds cached there, plus the KV of its active requests, all in blocks. Freed KV is left to the worker's LRU.
# track_active_blocks is the router_track_active_blocks flag: without it the KV of active requests costs nothing.
class KvRouter:
    def __init__(self, seed: int, track_active_blocks: bool = True) -> None:
        self.rng = random.Random(seed)
        self.track_active_blocks = track_active_blocks
        # New tokens of each request that is yet to give its first token, as estimated when it was routed
        self.prefill: dict[int, int] = {}

    def progress(self) -> str:
        return ""

    def wake(self, loop: Loop, events: list[Event]) -> None:
        sim, blk = loop.sim, loop.sim.blk_size
        for e in events:
            if e.kind == "first_token":
                del self.prefill[e.id]
        for j in loop.waiting():
            isl, costs = loop.reqs[j].isl, []
            for c, w in enumerate(sim.workers):
                active = [*w.running, *w.waiting]
                queued = sum(self.prefill.get(r.id, 0) for r in active)
                decode = sum(ceil(r.isl / blk) for r in active) if self.track_active_blocks else 0
                costs.append(max(0, (queued + isl) / blk - sim.hit(c, j) // blk) + decode)
            c = self.rng.choice([c for c, cost in enumerate(costs) if cost == min(costs)])
            self.prefill[j] = isl - sim.hit(c, j)
            loop.send(j, c)

# Soldr's model over what is in flight and the requests in sight, solved again at every wake. Only what the solution
# does right now is carried out: the requests it starts at time 0 and the session caches it drops.
class OnlineSoldr:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.solves = 0
        self.optimal = 0
        self.wall = 0.0
        # Relative gap between the objective and its bound, of each solve that found a solution
        self.gaps: list[float] = []

    def progress(self) -> str:
        gap = sum(self.gaps) / max(1, len(self.gaps))
        return f", {self.optimal}/{self.solves} solves optimal, {gap:.0%} avg gap"

    # Unsent requests the solver may plan for
    def window(self, loop: Loop) -> list[int]:
        cfg, reqs, depth = self.cfg, loop.reqs, dict[int, int]()
        for j in sorted(loop.unsent):
            u = reqs[j]
            if u.prev is None:
                if loop.release[j] <= loop.sim.now + cfg.lookahead_s * 1000:
                    depth[j] = 1
            elif u.prev not in loop.unsent:
                depth[j] = 1
            elif depth.get(u.prev, cfg.lookahead_turns) < cfg.lookahead_turns:
                depth[j] = depth[u.prev] + 1
        return list(depth)

    def wake(self, loop: Loop, events: list[Event]) -> float | None:
        sim = loop.sim
        ended = False
        for e in events:
            if e.kind == "end":
                if e.id in loop.succ:
                    ended = True
                else:
                    # A session cache whose session is over
                    sim.evict(e.worker, sim.session[e.id])
        # Nothing to decide unless a request waits to start or a session cache was just left behind
        if not loop.waiting() and not ended:
            return None
        return self.solve(loop)

    def solve(self, loop: Loop) -> float | None:
        cfg, sim, reqs, p, dt = self.cfg, loop.sim, loop.reqs, loop.p, loop.dt
        now, blk_size, C = sim.now, sim.blk_size, len(sim.workers)
        mem = p.kv_capacity if cfg.mem is None else cfg.mem
        size = lambda j: sim.blocks(reqs[j].isl + reqs[j].osl)
        window = self.window(loop)

        # Time is in steps from now. Requests in flight are constants: (cache, steps of prefill left, end).
        live: dict[int, tuple[int, int, int]] = {}
        busy = [0] * C
        for c, w in enumerate(sim.workers):
            r: Live
            for r in [*w.running, *w.waiting]:
                if r.first_token is None:
                    left = p(r.isl, r.computed if r.admit is not None else sim.hit(c, r.id))
                    live[r.id] = (c, left, left + r.osl)
                    busy[c] += left
                else:
                    live[r.id] = (c, 0, max(1, r.osl - r.generated))
        # Cache that holds the session cache of each finished request
        idle = {e.id: c for c, w in enumerate(sim.workers) for e in w.idle.values()}

        horizon = max([end for _, _, end in live.values()], default=0) + sum(
            p(reqs[j].isl) + reqs[j].osl + reqs[j].think + max(0, ceil((loop.release.get(j, now) - now) / dt))
            for j in window)
        model = cp_model.CpModel()
        intervals: list[list[cp_model.IntervalVar]] = [[] for c in range(C)]
        sizes: list[list[int]] = [[] for c in range(C)]
        prefills: list[list[cp_model.IntervalVar]] = [[] for c in range(C)]
        for j, (c, _, end) in live.items():
            intervals[c].append(model.new_fixed_size_interval_var(0, end, f"live[{j}]"))
            sizes[c].append(size(j))

        routes: dict[int, list[cp_model.IntVar]] = {}
        starts: dict[int, cp_model.IntVar] = {}
        ends: dict[int, cp_model.IntVar] = {}
        hits: dict[int, cp_model.IntVar] = {}
        ttfts: list[Expr] = []
        for j in window:
            v = reqs[j]
            routes[j] = [model.new_bool_var(f"route[{j},{c}]") for c in range(C)]
            model.add_exactly_one(routes[j])
            starts[j] = model.new_int_var(0, horizon, f"start[{j}]")
            ends[j] = model.new_int_var(0, horizon, f"end[{j}]")
            for c in range(C):
                if busy[c]:
                    model.add(starts[j] >= busy[c]).only_enforce_if(routes[j][c])

            release: Expr
            if j in loop.release:
                # A request that is not out yet must not start at 0, one that waits is owed its wait so far
                t = loop.release[j]
                release = ceil((t - now) / dt) if t > now else -round((now - t) / dt)
            elif v.prev in live:
                release = live[v.prev][2] + v.think
            else:
                assert v.prev is not None
                release = ends[v.prev] + v.think
            model.add(starts[j] >= release)

            # The session cache of the predecessor, kept until j starts if j is to hit it
            full = p(v.isl)
            save = 0 if v.prev is None else saving(reqs[v.prev], v, blk_size, p)
            hits[j] = model.new_bool_var(f"hit[{j}]") if save else model.new_constant(0)
            if not save:
                pass
            elif v.prev in idle:
                c = idle[v.prev]
                model.add_implication(hits[j], routes[j][c])
                intervals[c].append(model.new_optional_interval_var(0, starts[j], starts[j], hits[j], f"sc[{j}]"))
                sizes[c].append(size(v.prev))
            elif v.prev in live or v.prev in ends:
                prev_end: Expr = live[v.prev][2] if v.prev in live else ends[v.prev]
                gap = model.new_int_var(v.think, horizon, f"gap[{j}]")
                model.add(gap == starts[j] - prev_end)
                keeps = [model.new_bool_var(f"keep[{j},{c}]") for c in range(C)]
                model.add(sum(keeps) == hits[j])
                for c in range(C):
                    if v.prev not in live:
                        model.add_implication(keeps[c], routes[v.prev][c])
                    elif c != live[v.prev][0]:
                        model.add(keeps[c] == 0)
                    model.add_implication(keeps[c], routes[j][c])
                    intervals[c].append(model.new_optional_interval_var(prev_end, gap, starts[j], keeps[c],
                                                                        f"sc[{j},{c}]"))
                    sizes[c].append(size(v.prev))
            else:
                # The session cache is gone
                model.add(hits[j] == 0)

            prefill = full - hits[j] * save
            prefill_end = model.new_int_var(0, horizon, f"prefill_end[{j}]")
            model.add(prefill_end == starts[j] + prefill)
            model.add(ends[j] == prefill_end + v.osl)
            for c in range(C):
                intervals[c].append(model.new_optional_interval_var(starts[j], prefill + v.osl, ends[j], routes[j][c],
                                                                    f"lifetime[{j},{c}]"))
                sizes[c].append(size(j))
                prefills[c].append(model.new_optional_interval_var(starts[j], prefill, prefill_end, routes[j][c],
                                                                   f"prefill[{j},{c}]"))
            ttfts.append(prefill_end - release)

        for c in range(C):
            model.add_no_overlap(prefills[c])
            model.add_cumulative(intervals[c], sizes[c], mem)
        model.minimize(sum(ttfts))

        solver = cp_model.CpSolver()
        solver.parameters.num_workers = cfg.solver_workers
        solver.parameters.max_time_in_seconds = cfg.solve_s
        # Otherwise Ctrl-C only ends the solve in progress, and the run goes on
        solver.parameters.catch_sigint_signal = False
        t0 = time.time()
        status = solver.solve(model)
        self.solves += 1
        self.wall += time.time() - t0
        self.optimal += status == cp_model.OPTIMAL
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return now + 1000
        if cost := solver.objective_value:
            self.gaps.append((cost - solver.best_objective_bound) / cost)
        else:
            self.gaps.append(0.0)

        value = solver.value
        for j in window:
            prev = reqs[j].prev
            if prev in idle and not value(hits[j]):
                sim.evict(idle[prev], sim.session[prev])
        delayed = []
        for j in loop.waiting():
            if value(starts[j]) == 0:
                loop.send(j, next(c for c in range(C) if value(routes[j][c])))
            else:
                delayed.append(value(starts[j]))
        # Events wake us when a delayed request can go, the timer is for a delay the solver chose for another reason
        return now + min(delayed) * dt if delayed else None

# A scheduler by hand that rations KV memory the way Online Soldr turned out to. It is told think times and output
# lengths. A request goes to the worker where its first token is nearest, and only once that worker has room for all of
# it and no prefill in progress. Room is made by dropping session caches, cheapest first by the prefill they save per
# token-second they would sit idle, and only caches that sit idle for longer than it takes to rebuild them.
class Rationer:
    def __init__(self, patience: float = 1, aging: float = 0) -> None:
        # A cache is dropped for a request if its session is away for more than patience times its rebuild time
        self.patience = patience
        # Each ms a request has waited counts as this many ms that the caches in its way are away
        self.aging = aging

    def progress(self) -> str:
        return ""

    def wake(self, loop: Loop, events: list[Event]) -> None:
        sim, reqs, p, dt = loop.sim, loop.reqs, loop.p, loop.dt
        now, C = sim.now, len(sim.workers)
        size = lambda j: sim.blocks(reqs[j].isl + reqs[j].osl)
        for e in events:
            if e.kind == "end" and e.id not in loop.succ:
                sim.evict(e.worker, sim.session[e.id])
        waiting = loop.waiting()
        if not waiting:
            return None
        # ms of prefill that the requests on each worker are yet to do
        backlog = [sum(p.prefill_ms(r.isl, r.computed if r.admit is not None else sim.hit(c, r.id))
                       for r in [*w.running, *w.waiting] if r.first_token is None) for c, w in enumerate(sim.workers)]
        free = [p.kv_capacity - sum(size(r.id) for r in [*w.running, *w.waiting]) - sum(e.tokens for e in w.idle.values())
                for w in sim.workers]

        # (ms its session is still away, ms to rebuild it) of an idle cache
        def away(k: int) -> tuple[float, float]:
            v = loop.succ[k]
            return max(0.0, loop.release[v] - now), saving(reqs[k], reqs[v], sim.blk_size, p) * dt

        # Cheapest way to run j on c: (ms to its first token, sessions whose caches to drop), if it can be made to fit
        def option(j: int, c: int) -> tuple[float, list[int]] | None:
            idle, session = sim.workers[c].idle, sim.session[j]
            need = size(j) - (idle[session].tokens if session in idle else 0) - free[c]
            cost, victims = backlog[c] + p.prefill_ms(reqs[j].isl, sim.hit(c, j)), list[int]()
            scores = {s: away(e.id) for s, e in idle.items() if s != session}
            for s in sorted(scores, key=lambda s: scores[s][1] / (idle[s].tokens * max(scores[s][0], 1))):
                if need <= 0:
                    break
                gone, rebuild = scores[s]
                if gone + self.aging * (now - loop.release[j]) > self.patience * rebuild:
                    victims.append(s)
                    need -= idle[s].tokens
                    cost += rebuild
            return (cost, victims) if need <= 0 else None

        def send(j: int, c: int, victims: list[int]) -> None:
            session = sim.session[j]
            for s in victims:
                free[c] += sim.workers[c].idle[s].tokens
                sim.evict(c, s)
            for d, w in enumerate(sim.workers):
                if session in w.idle:
                    free[d] += w.idle[session].tokens
                    # Left behind for another worker
                    if d != c:
                        sim.evict(d, session)
            free[c] -= size(j)
            backlog[c] += p.prefill_ms(reqs[j].isl, sim.hit(c, j))
            loop.send(j, c)

        # Nearest first token first. A request whose best worker is still prefilling is held for it.
        while waiting:
            best: tuple[float, int, int, list[int]] | None = None
            for j in waiting:
                for c in range(C):
                    if (o := option(j, c)) is not None and (best is None or o[0] < best[0]):
                        best = (o[0], j, c, o[1])
            if best is None:
                break
            _, j, c, victims = best
            waiting.remove(j)
            if backlog[c] == 0:
                send(j, c, victims)
        # Nothing runs and what waits fits nowhere: the first to wait takes the emptiest worker, at any cost
        if not any(w.running or w.waiting for w in sim.workers) and (waiting := loop.waiting()):
            j, c = waiting[0], max(range(C), key=lambda c: free[c])
            idle, victims = sim.workers[c].idle, list[int]()
            for s in sorted((s for s in idle if s != sim.session[j]), key=lambda s: -away(idle[s].id)[0]):
                if free[c] + sum(idle[x].tokens for x in victims) >= size(j):
                    break
                victims.append(s)
            send(j, c, victims)
        return None

def stats(recs: list[dict[str, Any]], trace: Trace) -> dict[str, float]:
    ttfts = sorted(q["prefill_end"] - q["release"] for q in recs)
    sessions: dict[int, list[dict[str, Any]]] = {}
    for s, q in zip(sessions_of(trace.requests), recs):
        sessions.setdefault(s, []).append(q)
    quantile = lambda f: ttfts[ceil(f * len(ttfts)) - 1]
    return dict(mean=sum(ttfts) / len(ttfts), p50=quantile(0.5), p90=quantile(0.9), p95=quantile(0.95), max=ttfts[-1],
                reused=sum(q["reused"] for q in recs) / sum(u.isl for u in trace.requests),
                e2e=sum(q["end"] - q["release"] for q in recs) / len(recs),
                # From the release of a session's first request to the end of its last, think times included
                session_e2e=sum(max(q["end"] for q in qs) - min(q["release"] for q in qs)
                                for qs in sessions.values()) / len(sessions),
                # Time between tokens, which a prefill on the same worker stretches
                itl=sum(q["end"] - q["prefill_end"] for q in recs) / sum(u.osl - 1 for u in trace.requests))

class Online:
    def __init__(self, cfg: Config) -> None:
        assert cfg.objective == "mean"
        self.cfg = cfg
        self.p = MockerLatency.load(cfg.params)
        self.trace = Loader(self.p.decode_ms).weka(cfg.trace, sessions=cfg.sessions, n=cfg.num_requests,
                                                   stagger_s=cfg.stagger_s)
        self.name = "online_" + cfg.name(self.trace)
        self.timing = Timing(self.p)

    def loop(self) -> Loop:
        return Loop(self.trace, self.p, self.cfg.num_caches, self.timing)

    def run(self) -> None:
        cfg, trace = self.cfg, self.trace
        total_sessions = sum(u.prev is None for u in trace.requests)
        print("Total requests: ", len(trace.requests))
        print("Total sessions: ", total_sessions)
        print(f"Avg session length: {len(trace.requests) / total_sessions:.1f} requests")
        print(f"Session stagger: {cfg.stagger_s} s")
        print("Caches: ", cfg.num_caches)
        # As soldr prints for Mocker's run, here of the default KV router in the simulator
        usage = MockerUsage(trace, cfg.num_caches, self.p, [dict(
            arrival_time_ms=q["release"], first_admit_ms=q["start"], first_token_ms=q["prefill_end"],
            terminal_time_ms=q["end"], decode_worker_idx=q["worker"]) for q in self.loop().run(KvRouter(0))])
        usage.peak_working_set()
        usage.compute_usage()
        schedules: Schedules = {}
        rows: dict[str, dict[str, float]] = {}

        policy, loop = OnlineSoldr(cfg), self.loop()
        schedules["Online Soldr"] = loop.run(policy, cfg.log_s)
        rows["Online Soldr"] = stats(schedules["Online Soldr"], trace)
        schedules["Rationer"] = self.loop().run(Rationer())
        rows["Rationer"] = stats(schedules["Rationer"], trace)

        # The best run by mean TTFT is the one reported and plotted
        best = lambda runs: min(runs, key=lambda recs: stats(recs, trace)["mean"])
        from dynamo._core import KvRouterConfig
        mocker = Mocker(self.p, trace.blk_size)
        for label, track in [("Dynamo", True), ("Dynamo w/o tracking", False)]:
            name = f"{label} (best of {cfg.router_runs})"
            schedules[name] = best([self.loop().run(KvRouter(seed, track)) for seed in range(cfg.router_runs)])
            rows[name] = stats(schedules[name], trace)
            if cfg.mocker_runs:
                name = f"{label} in Mocker (best of {cfg.mocker_runs})"
                print(f"Replaying {label} in Mocker {cfg.mocker_runs} times", flush=True)
                router_config = KvRouterConfig(router_track_active_blocks=track)
                schedules[name] = best([[dict(worker=q["decode_worker_idx"], release=q["arrival_time_ms"],
                                              start=q["first_admit_ms"], prefill_end=q["first_token_ms"],
                                              end=q["terminal_time_ms"], reused=q["reused_input_tokens"])
                                         for q in mocker.run(mocker.rows(trace.requests), cfg.num_caches,
                                                             router_config=router_config)]
                                        for _ in range(cfg.mocker_runs)])
                rows[name] = stats(schedules[name], trace)

        width = max(map(len, rows))
        print(f"{'':{width}s} {'TTFT avg/p50/p90 (ms)':>22s} {'reuse':>7s} {'session E2E (s)':>16s}")
        for name, s in rows.items():
            ttft = f"{s['mean']:.0f}/{s['p50']:.0f}/{s['p90']:.0f}"
            print(f"{name:{width}s} {ttft:>22s} {s['reused']:7.1%} {s['session_e2e'] / 1000:16.1f}")

        print("Plotting", flush=True)
        plotter = Plotter(trace, schedules, self.p)
        plotter.swimlane(f"swimlane_{self.name}.png")
        plotter.cache_lanes(f"caches_{self.name}.png")
        path = f"{self.name}.json"
        with open(path, "w") as f:
            json.dump(dict(params=self.p.config, solves=policy.solves, optimal=policy.optimal, solver_s=policy.wall,
                           stats=rows, schedules=schedules), f, indent=1)
        print("Wrote", path)

if __name__ == "__main__":
    Online(Config()).run()
