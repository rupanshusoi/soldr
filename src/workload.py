import json
import random
from dataclasses import dataclass, replace
from itertools import islice

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
    # Requests of a raw session that share their first block form a sub-session. Sub-sessions are our sessions.
    # The raw sessions are picked at random, anew on every call unless a seed is given.
    def weka(self, path: str, sessions: int, n: int | None = None, stagger_s: float = 0,
             idle_cap_s: float = 300, seed: int | None = None) -> Trace:
        with open(path) as f:
            # Ignore sessions with subagents
            eligible = [session for session in map(json.loads, f)
                        if all(r["type"] != "subagent" for r in session["requests"])]
        assert len(eligible) >= sessions, f"only {len(eligible)} sessions without subagents"
        chosen = random.Random(seed).sample(eligible, sessions)
        blk_sizes = {p["block_size"] for p in chosen}
        assert len(blk_sizes) == 1

        # est is a release time estimate using the recorded API times, used to pick the first n requests
        trace: list[Request] = []
        est: list[float] = []
        offset = 0
        for s, session in enumerate(chosen):
            t = s * stagger_s
            # Last request of each sub-session and when it ends on the est timeline
            last: dict[int, tuple[int, float]] = {}
            for k, r in enumerate(session["requests"]):
                # Cap think time. This appears to be a standard transformation for
                # Weka and helps decrease the horizon for us.
                think = 0 if k == 0 else min(r["think_time"], idle_cap_s)
                t += think
                est.append(t)
                sub = r["hash_ids"][0]
                prev, arrival = None, None
                if sub not in last:
                    # A sub-session starts at its offset from the start of its raw session
                    arrival, think = round(t * 1000) // self.ms_per_step, 0
                else:
                    prev, end = last[sub]
                    # Requests of other sub-sessions ran in between
                    if prev != len(trace) - 1:
                        think = t - end
                trace.append(Request(r["in"], max(1, r["out"]), arrival, [h + offset for h in r["hash_ids"]], prev,
                                     round(think * 1000) // self.ms_per_step))
                t += r["api_time"]
                last[sub] = (len(trace) - 1, t)
            offset += 1 + max(h for r in session["requests"] for h in r["hash_ids"])

        assert n is None or len(trace) >= n
        order = sorted(range(len(trace)), key=lambda i: est[i])[:n]
        pos = {i: k for k, i in enumerate(order)}
        prev_of = lambda u: None if u.prev is None else pos[u.prev]
        return Trace([replace(trace[i], prev=prev_of(trace[i])) for i in order], blk_sizes.pop())

def shared_tokens(u: Request, v: Request, blk_size: int) -> int:
    count = 0
    for b1, b2 in zip(u.hash_ids, v.hash_ids):
        if b1 != b2:
            break
        count += 1
    return min(count * blk_size, u.isl, v.isl)

# Session (its first request) of each request
def sessions_of(reqs: list[Request]) -> list[int]:
    session: list[int] = []
    for j, u in enumerate(reqs):
        session.append(j if u.prev is None else session[u.prev])
    return session

def prev_shared(trace: Trace) -> list[int]:
    reqs = trace.requests
    return [0 if v.prev is None else shared_tokens(reqs[v.prev], v, trace.blk_size) for v in reqs]
