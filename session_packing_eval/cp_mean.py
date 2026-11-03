# cache_packing on the same instance with the mean objective, time-limited, then the usual Mocker comparison
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cache_packing import *
from session_packing import stats
trace, blk = from_weka("traces/weka_256k.jsonl", sessions=64, n=512, stagger_s=14)
model, vars = build_model(trace, blk, num_caches=4, mem=default_latency.kv_capacity, horizon=10_000_000,
                          prefill_latency=default_latency, objective="mean")
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 600
status = solver.Solve(model, GapLogger())
print(solver.StatusName(status), "bound avg", solver.BestObjectiveBound() / len(trace))
print("cache_packing mean (Soldr model)", stats([solver.Value(t) for t in vars["ttfts"]]))
compare(trace, blk, solver, vars, default_latency, "mean")
