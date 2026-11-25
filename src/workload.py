import glob
import json
import os
import random
from dataclasses import dataclass, replace
from itertools import islice
from typing import Any

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

        return self.first(trace, est, n, blk_sizes.pop())

    # The first n requests by est, which estimates when each is released
    def first(self, trace: list[Request], est: list[float], n: int | None, blk_size: int) -> Trace:
        assert n is None or len(trace) >= n
        order = sorted(range(len(trace)), key=lambda i: est[i])[:n]
        pos = {i: k for k, i in enumerate(order)}
        prev_of = lambda u: None if u.prev is None else pos[u.prev]
        return Trace([replace(trace[i], prev=prev_of(trace[i])) for i in order], blk_size)

    # The FreeInference agentic trace of Harvard MadSys, at the root of its release. As for Weka: recorded arrivals
    # are ignored, session starts are staggered, think times are capped and the sessions are picked at random. A
    # session is all the requests of a recorded session in order. Sessions with subagents are left out, and so are
    # those with a prompt over max_isl tokens, which the trace has up to 1.8M of.
    # The trace gives the prefix of a prompt in blocks of 16 tokens with ids that sessions share. We join them into
    # blocks of blk_size and keep the ids of sessions apart, as for Weka, so a session only reuses its own KV.
    def freeinference(self, path: str, sessions: int, n: int | None = None, stagger_s: float = 0,
                      idle_cap_s: float = 300, seed: int | None = None, max_isl: int = 256_000,
                      blk_size: int = 64) -> Trace:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
        weeks = sorted(glob.glob(os.path.join(path, "release", "*", "processed")))
        assert weeks, f"no FreeInference release under {path}"
        week_of: dict[str, str] = {}
        excluded: set[str] = set()
        for week in weeks:
            with open(os.path.join(week, "..", "manifest.json")) as f:
                per_blk = blk_size // json.load(f)["block_tokens"]
            for s in pq.read_table(os.path.join(week, "sessions.parquet")).to_pylist():
                week_of[s["session_id"]] = week
                # Subagents and the sessions that spawned them
                if s["parent_session_id"] is not None:
                    excluded |= {s["session_id"], s["parent_session_id"]}
            columns = ["session_id", "prompt_tokens", "waited_for"]
            for r in pq.read_table(os.path.join(week, "requests.parquet"), columns=columns).to_pylist():
                if r["waited_for"] == "subagent" or r["prompt_tokens"] > max_isl:
                    excluded.add(r["session_id"])
        eligible = sorted(week_of.keys() - excluded)
        assert len(eligible) >= sessions, f"only {len(eligible)} sessions without subagents"
        chosen = random.Random(seed).sample(eligible, sessions)

        rows: dict[str, list[dict[str, Any]]] = {s: [] for s in chosen}
        columns = ["session_id", "step", "start_ms", "end_ms", "completion_tokens", "block_ids"]
        for week in sorted({week_of[s] for s in chosen}):
            ids = [s for s in chosen if week_of[s] == week]
            table = pq.read_table(os.path.join(week, "requests.parquet"), columns=columns,
                                  filters=[("session_id", "in", ids)])
            for r in table.to_pylist():
                rows[r["session_id"]].append(r)

        trace: list[Request] = []
        est: list[float] = []
        # Id of each block of blk_size tokens, by the ids the trace gives its parts
        block: dict[tuple[int, tuple[int, ...]], int] = {}
        for s, session in enumerate(chosen):
            t = s * stagger_s
            prev = None
            for k, r in enumerate(sorted(rows[session], key=lambda r: r["step"])):
                parts = r["block_ids"]
                # The partial last block is dropped, as the trace does
                hash_ids = [block.setdefault((s, tuple(parts[i:i + per_blk])), len(block))
                            for i in range(0, max(per_blk, len(parts) - len(parts) % per_blk), per_blk)]
                think = 0.0 if prev is None else min(max(0, r["start_ms"] - prev["end_ms"]) / 1000, idle_cap_s)
                t += think
                est.append(t)
                arrival = round(t * 1000) // self.ms_per_step if prev is None else None
                trace.append(Request(len(hash_ids) * blk_size, max(1, r["completion_tokens"]), arrival, hash_ids,
                                     None if prev is None else len(trace) - 1,
                                     round(think * 1000) // self.ms_per_step))
                t += (r["end_ms"] - r["start_ms"]) / 1000
                prev = r
        return self.first(trace, est, n, blk_size)

    # By where the trace is: FreeInference is a directory, Weka a file
    def load(self, path: str, sessions: int, n: int | None = None, stagger_s: float = 0,
             seed: int | None = None) -> Trace:
        loader = self.freeinference if os.path.isdir(path) else self.weka
        return loader(path, sessions, n=n, stagger_s=stagger_s, seed=seed)

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
