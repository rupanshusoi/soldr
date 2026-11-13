import json
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

# For each request, the tokens it shares with each earlier request of its session
def earlier_shared(trace: Trace) -> list[dict[int, int]]:
    reqs = trace.requests
    session = sessions_of(reqs)
    return [{i: shared_tokens(reqs[i], v, trace.blk_size) for i in range(j) if session[i] == session[j]}
            for j, v in enumerate(reqs)]
