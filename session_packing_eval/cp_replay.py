# Replay cache_packing's dumped 4c/64s/512r schedule in Mocker the way calibrate.compare does (open loop per cache)
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from session_packing import *
trace, blk = from_weka("traces/weka_256k.jsonl", sessions=64, n=512, stagger_s=14)
d = json.load(open("schedule_4c_64s_512r_14st_qwen3.json"))
R = d["requests"]
assert all(r["isl"] == u.isl for r, u in zip(R, trace))
dt = default_latency.decode_ms
print("cache_packing (Soldr model)     ", stats([r["ttft"] for r in R]))
out = [0.0] * len(R)
for c in {r["cache"] for r in R}:
    ix = sorted((j for j in range(len(R)) if R[j]["cache"] == c), key=lambda j: (R[j]["start"], j))
    rows = [{"timestamp": R[j]["start"] * dt, "input_length": trace[j].isl, "output_length": trace[j].osl,
             "hash_ids": trace[j].hash_ids} for j in ix]
    for j, q in zip(ix, run_mocker(rows, 1, blk, default_latency)):
        out[j] = R[j]["start"] - R[j]["release"] + q["ttft_ms"] / dt
print("cache_packing (Mocker open-loop)", stats(out))
groups = sessions_of(trace)
split = sum(len({R[j]["cache"] for j in js}) > 1 for js in groups)
print("sessions split across caches:", split, "/", len(groups),
      " requests moved off prev's cache:", sum(r["prev"] is not None and r["cache"] != R[r["prev"]]["cache"] for r in R))
print("requests delayed past release:", sum(r["start"] > r["release"] for r in R), " with hit < best possible:",
      sum(r["hit"] < max([shared_tokens(trace[i], trace[r['id']], blk) for i in range(r['id'])] or [0]) for r in R))
