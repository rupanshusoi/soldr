# session_packing: handoff notes

`session_packing.py` is a session-level alternative to `cache_packing.py` for Weka traces. The only decision is which
cache serves each session, plus an optional single eviction per session. The model has no per-request variables.

## Model

- Each session gets a fixed **uncontended timeline**. Request k+1 starts at `end_k + think`. Each request takes its
  largest prefix hit from an earlier, still-resident request in the session.
- That timeline fixes the session's **KV profile**: a step function of the blocks that must be resident. A block is
  live while a request that contains it runs, and until the last request that hits on it starts. Hash blocks are
  deduplicated and running requests add their `osl` tokens.
- **Eviction option o** drops the session's KV after its request k. Requests after the gap lose their hits from before
  it, which changes their timeline, TTFTs and profile. The candidate gaps are the `evict_candidates=3` longest think
  gaps.
- `assign[s][c][o]` (exactly one per session) gates fixed optional intervals in one `AddCumulative` per cache.
- The global eviction budget `sum(assign[..][..][o>0]) <= max_evictions` is a hyperparameter. `__main__` runs an outer
  loop over budgets 0, 1, 2, 4, …, keeps doubling until a budget is feasible, then tries one more.
- **Objective:** the sum of estimated TTFTs (mean TTFT), in a single objective. It has two parts:
  - the uncontended prefill TTFT of the chosen eviction option, so an eviction is charged its real TTFT cost;
  - pairwise contention. For each pair of sessions on the same cache, it charges the FIFO delay their overlapping
    prefill intervals would cause (`same[a,b]` bools, linearized).
- **Ignored:** queueing delay does not shift a session's later requests, contention is only a pairwise estimate, and
  decode interference is not modelled.

## Evaluation

| Method | What it does |
|---|---|
| `simulate()` | Soldr cost model under a session routing: FIFO prefill per cache, closed loop with think times. |
| `replay_routing()` | Mocker, **exact** for session-sticky routing. Each cache's sessions are replayed closed-loop on a 1-worker Mocker, which is exact because sessions never share KV. |
| `replay_kv_router()` | Mocker's own kv_router on 4 workers, closed loop. |
| Baseline | Sticky round robin (session s → cache s mod C). |

## How to run (from the repo root, with `.venv/bin/python`)

```bash
.venv/bin/python -u session_packing.py            # same instance as cache_packing.py: 4 caches, 64 sessions, 512 req, 14 s stagger
```

```bash
.venv/bin/python -u session_packing.py 4          # stress test: per-cache KV capacity divided by 4
```

```bash
.venv/bin/python -u session_packing_eval/cp_replay.py   # replays cache_packing's P90-optimal schedule in Mocker (needs local untracked schedule_4c_64s_512r_14st_qwen3.json)
```

```bash
.venv/bin/python -u session_packing_eval/cp_mean.py     # cache_packing with the mean objective, 600 s limit, then calibrate.compare
```

Use `-u`, because output piped through grep is otherwise buffered and you can't see progress. Lines starting with `[`
are the CP-SAT solution log.

## Results so far

All values are TTFT in 6 ms timesteps on the 4c/64s/512r instance. That instance has only 41 sessions, because the
512-request cutoff truncates it.

| | mean | p50 | p90 | p99 |
|---|---|---|---|---|
| Uncontended lower bound (each request's own prefill) | 36.4 | 12 | 109 | 263 |
| session_packing, Soldr sim (OPTIMAL in seconds, est. = bound) | 36.4 | 12 | 109 | 263 |
| **session_packing, Mocker** | **37.4** | **12.6** | **111.4** | **266** |
| Sticky round robin, Mocker | 40.0 | 13.5 | 113.5 | 309 |
| Mocker kv_router, 3 runs | 65–76 | 22–26 | 185–198 | 405–683 |
| cache_packing P90-optimal schedule, Soldr model | 131.2 | 98 | 109 | 1594 |
| cache_packing P90-optimal schedule, Mocker open-loop | 113.1 | 67 | 114 | 1599 |
| cache_packing mean objective (stopped at 105 s, incumbent) | ~281 | | | |

The cache_packing mean-objective run reached that incumbent with a bound of 1.7 and a 99% gap when it was stopped. It
has not been rerun to completion.

### Stress tests

- **mem/4 (583k per cache):** feasible with 0 evictions and still contention-free in the model. The Mocker mean was
  40.9, but see TODO 1: Mocker still used full capacity.
- **mem/6 (389k per cache):** INFEASIBLE at budgets 0, 1, 2, 4 and 8. The run was stopped during budget 16.
  - Peak concurrent KV across all sessions is 2.19M.
  - 4 × 389k = 1.56M, so this level needs real eviction.
  - One eviction per session, limited to the top-3 think gaps, frees only one gap's hold. That seems too weak.

## Patterns observed

- **This instance is not memory-bound.** Peak concurrent KV across *all* 41 sessions is 2.19M, which is less than
  *one* cache (2.33M). Memory constraints and evictions never bind at real capacity, so the problem is pure contention
  avoidance. With 79 conflicting session pairs and 4 caches, it is a graph colouring that has a zero-conflict solution.
- **cache_packing's proven-optimal P90 of 109 equals the uncontended P90.** The P90 is set by cold first-turn prefills
  that no routing can avoid.
  - The P90 objective lets cache_packing freely degrade everything under the threshold. Its p50 is 98 steps
    (session_packing gets 12).
  - It splits 35 of 41 sessions across caches.
  - It delays 317 of 512 requests past release.
  - It gives 251 requests a smaller hit than they could have gotten.
- **Minimizing evictions vs minimizing TTFT.** These are not the same objective. The objective charges each eviction
  its exact TTFT cost, so minimizing TTFT already prefers fewer evictions. Evictions are only worth taking when they
  free memory that lets sessions avoid contention. The eviction budget therefore works as a feasibility and robustness
  knob, not as a second objective.
- **kv_router mis-routes sessions.** It moves a session off its previous worker 133–182 times out of 471 follow-up
  turns, which loses the prefix hit. Even sticky round robin beats it by about 1.7× on mean TTFT.
- **Mocker TTFT for zero-conflict routings varies by a few steps.** At full memory the result was 37.4, and with a
  different zero-conflict routing it was 40.9. Contention that the pairwise estimate misses (chunked prefill and
  decode batching) is the next-order effect. Maybe use the conflict cost as a tie-breaker, or estimate overlap with a
  slack margin.

## TODO, in priority order

1. **Mocker capacity in stress mode.** `replay_routing` and `replay_kv_router` build Mocker from `default_latency`, so
   they always use the full `kv_capacity` even when `mem` is reduced. Pass `dataclasses.replace(p, kv_capacity=mem)`,
   then rerun `session_packing.py 4`.
2. **Finish the cache_packing mean-objective comparison** (`cp_mean.py`). Report its Soldr and Mocker mean TTFT next to
   session_packing's 37.4.
3. **Scale.** Real capacity makes the 512-request instance trivial. Try the full sessions:
   - Use `from_weka(..., sessions=64, n=None)`, which is 3,433 requests.
   - Try 128 sessions on 8 caches (7,018 requests).
   - Only 218 subagent-free sessions exist, so `sessions` must be ≤ 218.
   - Time the model build: `contention()` is O(n_a·n_b) Python per session pair and may need a sorted sweep.
   - Also try lower capacity or a shorter stagger so memory binds.
4. **Eviction model at mem/6.**
   - Allow all gaps (`evict_candidates=None` → all).
   - Allow more than one eviction per session.
   - Allow partial eviction that keeps the prefix the next request needs.
   - Check whether mem/6 becomes feasible, and at what budget.
5. Report everything in the results table format above: mean, p50, p90 and p99 in timesteps, for Soldr sim, Mocker
   replay, sticky round robin and kv_router.
