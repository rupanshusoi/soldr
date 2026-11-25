import sys
import time
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
from scipy.optimize import linprog  # type: ignore[import-untyped]
from scipy.sparse import coo_matrix, csr_matrix, hstack  # type: ignore[import-untyped]
import soldr
from mocker import MockerLatency
from soldr import saving
from workload import Loader, Trace, sessions_of

Array = Any

# What an item takes of a cache, by its size as a fraction of the cache, in each memory resource. Items that fit a
# cache together take at most 1 of each: by size, and by count, as at most q - 1 items of more than 1/q fit. Richer
# families of such functions (Fekete and Schepers' dual feasible functions) did not raise the bound.
def shares() -> list[Callable[[float], float]]:
    return [lambda x: x] + [lambda x, q=q: (x * q > 1 + 1e-9) / (q - 1) for q in range(2, 6)]  # type: ignore[misc]

# A lower bound on the mean TTFT of soldr's global model, for traces that model is too large to solve.
#
# The model couples sessions only through what the caches can hold and prefill at a time. We relax that coupling:
# all caches are pooled, and what they can do at a time is only asked to hold on average over each piece of the
# horizon. With a price on every pooled resource in every piece (a Lagrange multiplier), the model falls apart into
# one problem per session, which a dynamic program over the session's turns solves exactly: when to start each turn,
# and whether to keep the session's KV for it. Any prices give a lower bound. Good prices come from a master LP over
# the session schedules found so far (Dantzig-Wolfe column generation).
#
# The pooled resources, each of which C caches have C of at a time:
#   memory    a request, or a kept session cache, of m tokens takes m / mem, and 1 / (q - 1) if it is over mem / q,
#             for q = 2, 3, ..., as a cache fits q - 1 of those
#   prefill   a prefilling request takes 1, as a cache prefills one request at a time

@dataclass
class Turn:
    # Steps of prefill with and without the session's KV, which is equal if there is nothing to reuse
    hit: int
    miss: int
    osl: int
    # Steps from the end of the turn before to the release of this one. For a first turn, its arrival.
    think: int
    # Tokens the turn holds, and leaves behind for the next
    size: int

@dataclass
class Schedule:
    # Step at which the session's last turn ends
    end: float
    # Use of each resource in each piece, in cache-steps
    usage: Array

class Bound:
    def __init__(self, trace: Trace, p: MockerLatency, caches: int, mem: int | None = None, grid: int = 25,
                 pieces: int = 150, horizon: int | None = None) -> None:
        reqs = trace.requests
        self.n, self.caches, self.dt = len(reqs), caches, p.decode_ms
        self.mem = p.kv_capacity if mem is None else mem
        chains: dict[int, list[int]] = {}
        for j, s in enumerate(sessions_of(reqs)):
            chains.setdefault(s, []).append(j)
        self.sessions: list[list[Turn]] = []
        for chain in chains.values():
            turns: list[Turn] = []
            for j in chain:
                u = reqs[j]
                assert u.prev is None or u.prev == chain[len(turns) - 1]
                save = 0 if u.prev is None else saving(reqs[u.prev], u, trace.blk_size, p)
                think = u.think if u.prev is not None else u.arrival
                assert think is not None
                turns.append(Turn(p(u.isl) - save, p(u.isl), u.osl, think, u.isl + u.osl))
            self.sessions.append(turns)
        # Sum of TTFTs is the sum over sessions of the end of the last turn, less this
        self.fixed = sum(t.osl + t.think for turns in self.sessions for t in turns)

        # Time is cut into cells of grid steps. Prices are constant over a piece, which is a whole number of cells,
        # and zero from the horizon on.
        if horizon is None:
            horizon = int(1.25 * max(sum(t.think + t.miss + t.osl for t in turns) for turns in self.sessions))
        self.g = grid
        self.cells_per_piece = max(1, -(-horizon // (grid * pieces)))
        self.pieces = pieces
        self.N = self.cells_per_piece * pieces
        self.steps = np.arange(self.N, dtype=np.float64) * grid
        # Resources: memory by each share function, then prefill
        self.fns = shares()
        self.R = len(self.fns) + 1
        self.weights = [self.weight(t.size) for turns in self.sessions for t in turns]

    # What tokens of KV take of each memory resource
    def weight(self, tokens: int) -> Array:
        return np.array([f(tokens / self.mem) for f in self.fns])

    # Prices per piece as what the dynamic programs use: per cell, and summed up to each cell
    def prepare(self, prices: Array) -> tuple[Array, Array]:
        rate = np.repeat(prices, self.cells_per_piece, axis=1)
        # A schedule on the grid stands for all that start up to a cell later, so each cell is priced at the lower
        # of its price and the next one's
        rate = np.minimum(rate, np.concatenate([rate[:, 1:], np.zeros((self.R, 1))], axis=1))
        total = np.concatenate([np.zeros((self.R, 1)), np.cumsum(rate, axis=1) * self.g], axis=1)
        return rate, total

    # total at offset steps after the start of every cell
    def at(self, rate: Array, total: Array, offset: int) -> Array:
        q, r = divmod(offset, self.g)
        out = np.full(self.N, total[self.N])
        if q < self.N:
            out[:self.N - q] = total[q:self.N] + r * rate[q:]
        return out

    # values q cells on. Past the horizon there are no prices: a turn that starts at step x ends the session at
    # x + tail.
    def shift(self, values: Array, q: int, tail: float) -> Array:
        out = np.empty(self.N)
        m = max(0, self.N - q)
        out[:m] = values[q:q + m]
        out[m:] = (np.arange(m, self.N) + q) * self.g + tail
        return out

    def shift_index(self, index: Array, q: int) -> Array:
        out = np.full(self.N, self.N, dtype=np.int32)
        m = max(0, self.N - q)
        out[:m] = index[q:q + m]
        return out

    # The cheapest schedule of a session at the given prices: the end of its last turn plus what it pays for what
    # it uses. Starts are whole cells. A turn may start at the cell its release falls in, and a kept cache is not
    # charged for the last cell before its turn starts, so that every schedule in steps costs at least as much as
    # some schedule on the grid.
    def session(self, s: int, rate: Array, total: Array, weights: list[Array], want: bool) -> tuple[float, Schedule | None]:
        turns, N, g = self.sessions[s], self.N, self.g
        n = len(turns)
        P, prate = total[-1], rate[-1]
        # Steps from the start of turn k to the end of the session when nothing waits
        tail_hit, tail_miss = [0.0] * (n + 1), [0.0] * (n + 1)
        for k in range(n - 1, -1, -1):
            rest = turns[k].osl + (turns[k + 1].think + tail_hit[k + 1] if k + 1 < n else 0)
            tail_hit[k], tail_miss[k] = turns[k].hit + rest, turns[k].miss + rest

        mem_rate = lambda k: weights[k] @ rate[:-1]
        mem_total = lambda k: weights[k] @ total[:-1]
        # Of the turn after the one being worked on: cost by start cell if it hits, least cost by release cell if
        # it misses, and least cost with what its kept cache pays up to the cell before it starts
        nxt_hit = nxt_miss = nxt_kept = None
        arg_miss = arg_kept = None
        back: list[dict[bool, tuple[Array, Array]]] = [{} for _ in range(n)]
        for k in range(n - 1, -1, -1):
            t = turns[k]
            krate, ktotal = mem_rate(k), mem_total(k)
            cost: dict[bool, Array] = {}
            for hit in ([True, False] if t.hit < t.miss else [False]):
                pre = t.hit if hit else t.miss
                run = pre + t.osl
                c = self.at(prate, P, pre) - P[:N] + self.at(krate, ktotal, run) - ktotal[:N]
                if k == n - 1:
                    c += self.steps + run
                else:
                    u = turns[k + 1]
                    q = (run + u.think) // g
                    assert nxt_miss is not None and arg_miss is not None
                    after = self.shift(nxt_miss, q, tail_miss[k + 1])
                    start, keep = self.shift_index(arg_miss, q), np.zeros(N, dtype=bool)
                    if u.hit < u.miss:
                        assert nxt_hit is not None and nxt_kept is not None and arg_kept is not None
                        released = self.at(krate, ktotal, run + u.think)
                        held = released - self.at(krate, ktotal, run)
                        options = [self.shift(nxt_hit, q, tail_hit[k + 1]), self.shift(nxt_hit, q + 1, tail_hit[k + 1]),
                                   self.shift(nxt_kept, q + 2, ktotal[N] + tail_hit[k + 1]) - released]
                        starts = [np.minimum(np.arange(N) + q, N), np.minimum(np.arange(N) + q + 1, N),
                                  self.shift_index(arg_kept, q + 2)]
                        for option, cells in zip(options, starts):
                            better = held + option < after
                            after = np.where(better, held + option, after)
                            start, keep = np.where(better, cells, start), keep | better
                    c += after
                    back[k][hit] = (start.astype(np.int32), keep)
                cost[hit] = c
            # Least over the cells from each one on
            def least(values: Array, tail: float) -> tuple[Array, Array]:
                rev = values[::-1]
                best = np.minimum.accumulate(rev)
                # The cell that attains it, or N if starting past the horizon is cheaper
                arg = np.maximum.accumulate(np.where(rev <= best, np.arange(N), 0))
                arg = np.where(best <= tail, N - 1 - arg, N)
                return np.minimum(best, tail)[::-1], arg[::-1].astype(np.int32)
            nxt_miss, arg_miss = least(cost[False], N * g + tail_miss[k])
            if True in cost:
                assert k > 0
                ptotal = mem_total(k - 1)
                before = np.concatenate([[0.0], ptotal[:N - 1]])
                nxt_hit = cost[True]
                nxt_kept, arg_kept = least(before + cost[True], ptotal[N] + N * g + tail_hit[k])

        cell = min(turns[0].think // g, N)
        assert nxt_miss is not None and arg_miss is not None
        value = float(nxt_miss[cell]) if cell < N else cell * g + tail_miss[0]
        if not want:
            return value, None

        # Walk the choices forward. Past the horizon nothing waits and nothing is charged.
        usage = np.zeros((self.R, self.pieces))
        piece = self.cells_per_piece * g
        def use(resource: Array, a: float, b: float) -> None:
            for i in range(int(a // piece), min(self.pieces, int(-(-b // piece)))):
                usage[:, i] += resource * (min(b, (i + 1) * piece) - max(a, i * piece))
        prefill = np.zeros(self.R)
        prefill[-1] = 1
        at = int(arg_miss[cell]) if cell < N else N
        hit, x = False, float(turns[0].think)
        for k, t in enumerate(turns):
            if at < N:
                x = float(at * g)
            pre = t.hit if hit else t.miss
            use(prefill, x, x + pre)
            use(np.append(weights[k], 0), x, x + pre + t.osl)
            end = x + pre + t.osl
            if k + 1 < n:
                if at < N:
                    nxt, kept = back[k][hit]
                    hit, at = bool(kept[at]), int(nxt[at])
                else:
                    hit = turns[k + 1].hit < turns[k + 1].miss
                release = end + turns[k + 1].think
                x = max(release, float(at * g)) if at < N else release
                if hit:
                    use(np.append(weights[k], 0), end, x)
        return value, Schedule(end, usage)

    def run(self, iterations: int = 200, tol: float = 0.01, smooth: float = 0.5, cap: float = 1000,
            log: bool = True) -> float:
        S, R, I = len(self.sessions), self.R, self.pieces
        offsets = np.cumsum([0] + [len(turns) for turns in self.sessions])
        weights = [self.weights[offsets[s]:offsets[s + 1]] for s in range(S)]
        have = self.caches * self.cells_per_piece * self.g
        to_ms = lambda total: (total - self.fixed) / self.n * self.dt

        def evaluate(prices: Array) -> tuple[float, list[Schedule]]:
            rate, total = self.prepare(prices)
            values, schedules = 0.0, []
            for s in range(S):
                value, schedule = self.session(s, rate, total, weights[s], True)
                assert schedule is not None
                values += value
                schedules.append(schedule)
            return values - prices.sum() * have, schedules

        t0 = time.time()
        best, schedules = evaluate(np.zeros((R, I)))
        floor, best_prices = best, np.zeros((R, I))
        columns: list[tuple[int, Schedule]] = [(s, c) for s, c in enumerate(schedules)]
        if log:
            print(f"No contention: {to_ms(floor):.0f} ms")
        for it in range(iterations):
            # Master: the cheapest mix of the schedules found so far that fits, with what does not fit bought at cap
            m = len(columns)
            rows = np.concatenate([np.flatnonzero(c.usage.ravel()) for _, c in columns])
            cols = np.concatenate([np.full(np.count_nonzero(c.usage), i) for i, (_, c) in enumerate(columns)])
            vals = np.concatenate([c.usage.ravel()[np.flatnonzero(c.usage.ravel())] for _, c in columns])
            use = coo_matrix((vals, (rows, cols)), shape=(R * I, m)).tocsr()
            slack = -csr_matrix(np.eye(R * I))
            pick = coo_matrix((np.ones(m), ([s for s, _ in columns], np.arange(m))), shape=(S, m)).tocsr()
            res = linprog(np.concatenate([[c.end for _, c in columns], np.full(R * I, cap * have)]),
                          A_ub=hstack([use, slack * have]), b_ub=np.full(R * I, have),
                          A_eq=hstack([pick, csr_matrix((S, R * I))]), b_eq=np.ones(S), bounds=(0, None), method="highs")
            assert res.status == 0, res.message
            master = res.fun
            prices = np.maximum(0, -res.ineqlin.marginals).reshape(R, I)
            # Prices between the best so far and the master's, which on their own swing too much
            prices = smooth * best_prices + (1 - smooth) * prices
            bound, schedules = evaluate(prices)
            if bound > best:
                best, best_prices = bound, prices
            columns += [(s, c) for s, c in enumerate(schedules)]
            unmet = float(res.x[m:].sum())
            if log:
                print(f"[{time.time() - t0:4.0f}s] {it + 1}: bound {to_ms(best):.0f} ms, master {to_ms(master):.0f} ms"
                      f"{f' ({unmet:.1f} bought)' if unmet > 1e-6 else ''}", flush=True)
            if unmet <= 1e-6 and master - best <= tol * (master - self.fixed):
                break
        self.floor_ms, self.bound_ms = to_ms(floor), to_ms(best)
        self.columns, self.have = columns, have
        return self.bound_ms

if __name__ == "__main__":
    cfg = soldr.Config()
    p = MockerLatency.load(cfg.params)
    trace = Loader(p.decode_ms).load(cfg.trace, sessions=cfg.sessions, n=cfg.num_requests, stagger_s=cfg.stagger_s,
                                     seed=cfg.seed)
    print(f"{len(trace.requests)} requests, {cfg.sessions} sessions, {cfg.num_caches} caches, seed {cfg.seed}")
    print(f"Lower bound on mean TTFT: {Bound(trace, p, cfg.num_caches, cfg.mem).run():.0f} ms")
