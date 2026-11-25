import random
import time
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from math import ceil
from typing import Any, Literal, Protocol
from mocker import MockerLatency
from workload import Trace, sessions_of, shared_tokens

# Mocker's timing of a forward pass
class Timing:
    def __init__(self, p: MockerLatency) -> None:
        from aisimulate_core.sdk.engine import EngineHandle
        aic = p.aic
        engine = EngineHandle.compile(aic["model"], aic["system"], "vllm", backend_version="0.24.0", tp_size=aic["tp"],
                                      gemm_quant_mode="bfloat16", fmha_quant_mode="bfloat16",
                                      kvcache_quant_mode="bfloat16", kv_block_size=p.kv_block)
        # new tokens after a prefix, for each of batch requests
        self.prefill = lru_cache(None)(lambda batch, new, prefix: engine.predict_prefill_latency(batch, prefix + new, prefix))
        # One token for each of batch requests
        self.decode = lru_cache(None)(lambda batch, context: max(1.0, engine.predict_decode_latency(batch, context, 2)))

# A request on a worker
@dataclass
class Live:
    id: int
    isl: int
    osl: int
    submit: float
    # Tokens whose KV is on the worker, reused or computed
    computed: int = 0
    generated: int = 0
    reused: int = 0
    admit: float | None = None
    first_token: float | None = None

# The KV a finished request leaves behind, which its session may reuse
@dataclass
class Idle:
    id: int
    tokens: int

@dataclass
class Event:
    kind: Literal["first_token", "end"]
    id: int
    worker: int

@dataclass
class Worker:
    waiting: deque[Live] = field(default_factory=deque)
    running: list[Live] = field(default_factory=list)
    # By session, least recently used first
    idle: dict[int, Idle] = field(default_factory=dict)
    # The pass in flight: when it ends and who gets a token from it
    pass_end: float | None = None
    emits: list[Live] = field(default_factory=list)

# A cluster of vLLM workers as Mocker simulates them, stepped from outside. Follows Mocker's vLLM scheduler: each forward
# pass spends a token budget on the running requests in admission order, then admits waiting ones. Every decoding
# request gets one token per pass, and the pass takes its prefill time plus its decode time, which is how a prefill
# slows down the decodes that share its worker.
class Sim:
    def __init__(self, trace: Trace, p: MockerLatency, workers: int, timing: Timing | None = None) -> None:
        self.reqs, self.blk_size, self.p = trace.requests, trace.blk_size, p
        self.session = sessions_of(self.reqs)
        self.timing = timing or Timing(p)
        self.workers = [Worker() for _ in range(workers)]
        self.now = 0.0
        self.recs: dict[int, dict[str, Any]] = {}
        # Should stay at 0 when the caller manages memory itself
        self.forced_evictions = 0
        self.preemptions = 0

    def blocks(self, tokens: int) -> int:
        return -(-tokens // self.p.kv_block) * self.p.kv_block

    def used(self, c: int) -> int:
        w = self.workers[c]
        return sum(self.blocks(r.computed) for r in w.running) + sum(e.tokens for e in w.idle.values())

    # Tokens of request j that the idle KV on worker c would save it
    def hit(self, c: int, j: int) -> int:
        v, best = self.reqs[j], 0
        for e in self.workers[c].idle.values():
            best = max(best, min(shared_tokens(self.reqs[e.id], v, self.blk_size), e.tokens))
        return min(best, v.isl - 1) // self.p.kv_block * self.p.kv_block

    def submit(self, j: int, c: int, release: float) -> None:
        u, w = self.reqs[j], self.workers[c]
        w.waiting.append(Live(j, u.isl, u.osl, self.now))
        self.recs[j] = dict(worker=c, release=release, submit=self.now)
        if w.pass_end is None:
            self.start_pass(c)

    def evict(self, c: int, session: int) -> None:
        del self.workers[c].idle[session]

    # Makes room for a request to grow, first from idle KV and then by preempting the newest running request. False
    # if the request itself got preempted.
    def reserve(self, c: int, r: Live, tokens: int) -> bool:
        w = self.workers[c]
        need = self.used(c) + self.blocks(r.computed + tokens) - self.blocks(r.computed) - self.p.kv_capacity
        while need > 0 and w.idle:
            session, e = next(iter(w.idle.items()))
            # vLLM reclaims a freed sequence from its tail
            cut = min(e.tokens, self.blocks(need))
            e.tokens -= cut
            need -= cut
            self.forced_evictions += 1
            if e.tokens == 0:
                del w.idle[session]
        while need > 0:
            victim = w.running.pop()
            need -= self.blocks(victim.computed)
            victim.computed = 0
            self.preemptions += 1
            w.waiting.appendleft(victim)
            if victim is r:
                return False
        return True

    def start_pass(self, c: int) -> None:
        w, now = self.workers[c], self.now
        budget = self.p.chunk
        # (new tokens, prefix) of each request that prefills in this pass
        prefills: list[tuple[int, int]] = []
        scheduled: set[int] = set()

        def schedule(r: Live) -> bool:
            nonlocal budget
            tokens = min(r.isl + r.generated - r.computed, budget)
            if not self.reserve(c, r, tokens):
                return False
            prompt = min(r.computed + tokens, r.isl) - min(r.computed, r.isl)
            if prompt:
                prefills.append((prompt, r.computed))
                scheduled.add(r.id)
            r.computed += tokens
            budget -= tokens
            return True

        k = 0
        while k < len(w.running) and budget > 0:
            r = w.running[k]
            if schedule(r):
                k += 1
        while w.waiting and budget > 0:
            r = w.waiting[0]
            fresh = r.computed == 0 and r.generated == 0
            reused = self.hit(c, r.id) if fresh else 0
            # vLLM admits a request only if what the running ones hold leaves room for all of its prompt, the part
            # it reuses included: idle KV is free until a request takes it
            held = sum(self.blocks(x.computed) for x in w.running)
            if self.blocks(r.isl + r.generated) > self.p.kv_capacity - held:
                break
            w.waiting.popleft()
            if r.admit is None:
                r.admit, r.reused = now, reused
                self.recs[r.id].update(start=now, reused=reused)
            # The request takes over the KV of its session
            if fresh:
                w.idle.pop(self.session[r.id], None)
                r.computed = reused
            w.running.append(r)
            if not schedule(r):
                break

        ready = [r for r in w.running if r.computed >= r.isl + r.generated]
        # The first token comes out of the pass that finishes the prefill
        decodes = [r for r in ready if not (r.generated == 0 and r.id in scheduled)]
        duration = 0.0
        if prefills:
            n = len(prefills)
            duration += self.timing.prefill(n, sum(new for new, _ in prefills) // n, sum(pre for _, pre in prefills) // n)
        if decodes:
            duration += self.timing.decode(len(decodes), sum(r.isl + r.generated for r in decodes) // len(decodes))
        if not ready and not prefills:
            w.pass_end = None
            return
        w.pass_end, w.emits = now + duration, ready

    def end_pass(self, c: int) -> list[Event]:
        w, now = self.workers[c], self.now
        events = []
        for r in w.emits:
            r.generated += 1
            if r.first_token is None:
                r.first_token = now
                self.recs[r.id]["prefill_end"] = now
                events.append(Event("first_token", r.id, c))
            if r.generated == r.osl:
                w.running.remove(r)
                w.idle[self.session[r.id]] = Idle(r.id, self.blocks(r.computed))
                self.recs[r.id]["end"] = now
                events.append(Event("end", r.id, c))
        w.pass_end, w.emits = None, []
        self.start_pass(c)
        return events

    # Runs to the first pass that ends with events at or before until, or to until if there is none
    def step(self, until: float) -> list[Event]:
        while True:
            ends = [(w.pass_end, c) for c, w in enumerate(self.workers) if w.pass_end is not None and w.pass_end <= until]
            if not ends:
                self.now = max(self.now, until)
                return []
            self.now, c = min(ends)
            if events := self.end_pass(c):
                return events

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
