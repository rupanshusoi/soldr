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
.venv/bin/python -u session_packing.py                                        # same instance as cache_packing.py: 4 caches, 64 sessions, 512 req, 14 s stagger
```

```bash
.venv/bin/python -u session_packing.py --mem-div 4                            # stress test: per-cache KV capacity divided by 4 (solver and Mocker)
```

```bash
.venv/bin/python -u session_packing.py --requests 0                           # all 64 sessions in full (3,433 requests), ~4 min
```

```bash
.venv/bin/python -u session_packing.py --requests 0 --sessions 128 --caches 8 # 7,018 requests, ~4 min
```

Other flags are `--stagger` (in seconds), `--time-limit` (per eviction budget, default 60 s) and `--kv-router-runs`.

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

### Full sessions, mean TTFT (prelim, 60 s solves)

The "Mocker alone" row is Mocker's own no-contention floor: every session is replayed by itself on its own worker.

| | 64 sessions / 4 caches (3,433 req) | 128 sessions / 8 caches (7,018 req) |
|---|---|---|
| Peak concurrent KV, in caches' worth | 2.68 of 4 | 5.35 of 8 |
| Soldr uncontended lower bound | 25.5 | 26.8 |
| Mocker alone (no contention) | – | **25.8** |
| session_packing, Soldr sim | 25.8 | 26.9 |
| **session_packing, Mocker** | **28.6** | **29.4–30.5** (two solves) |
| Sticky round robin, Mocker | 30.4 | 30.4 |
| Mocker kv_router | 134.0 | 208.0 |

Both solves ended FEASIBLE, with the model's gap under 0.5%. kv_router moved a session off its previous worker 1,250
times (64 sessions) and 3,694 times (128 sessions).

- **The model misses most of the real contention at scale.** With 128 sessions on 8 caches, the Soldr model puts
  session_packing at the floor (26.9) and round robin 2.4 steps above it.
  - In Mocker, both routings sit about 4 steps above Mocker's own floor (25.8).
  - The floor is below the Soldr bound, so the calibration is not biased upward and the excess is contention.
  - Per cache, the Mocker excess is +2–4 steps and fairly uniform. It does not track lost cache hits, which are
    small and occur under both routings.
  - Likely cause: decode interference. vLLM mixes prefill chunks with decode tokens in an 8,192-token budget, and the
    model only charges FIFO overlap between prefills. The two near-optimal solves differed by 1 step in Mocker, which
    is also the noise level of the comparison.
- **Most of the gain over kv_router comes from keeping each session on one cache.** Sticky round robin gets most of
  it. session_packing adds 0–2 steps (0–6%) on top in Mocker, against about 4.6 steps of total headroom to the floor.

### Memory-bound case: 218 sessions / 8 caches (11,846 req)

- **Budget 0 is provably infeasible.** Peak concurrent KV is 10.0 caches' worth against 8 caches.
- CP-SAT returned UNKNOWN in 30 s. The run was stopped during budget 1, because the remaining budgets would take about
  15 min and likely also end UNKNOWN.
- **The eviction model can't express this regime.** At most one eviction per session, in one of its top-3 think gaps,
  cannot free 2 caches' worth at a single peak instant. Memory-bound operation needs most sessions evicted during most
  of their long gaps.
- **Suggested fix:** replace "evict at gap g" with per-session **TTL options**. Evict during every think gap longer
  than τ, with τ from a small set such as {∞, 120 s, 60 s, 30 s, 10 s}. This still uses only per-session option
  variables.
  - It is a real policy, which makes it easy to replay.
  - The budget becomes the total number of re-prefills, or the objective just charges them.
  - CP-SAT will also need a greedy hint to find any feasible solution at this size.

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

1. ~~Mocker capacity in stress mode~~ Done: `--mem-div` now applies to Mocker too. The table above has not been rerun
   since this fix.
2. **Finish the cache_packing mean-objective comparison** (`cp_mean.py`) on the 512-request instance only.
   cache_packing can't build the full-session instances (O(N²)).
3. **Contention model:** add decode-load interference to the pairwise cost, then check whether Mocker's ~4-step excess
   over the floor drops.
4. **TTL eviction options** for the 218-session memory-bound case (see above).
5. ~~Scale~~ Partly done (see above). Earlier notes: Real capacity makes the 512-request instance trivial. Try the full sessions:
   - Use `from_weka(..., sessions=64, n=None)`, which is 3,433 requests.
   - Try 128 sessions on 8 caches (7,018 requests).
   - Only 218 subagent-free sessions exist, so `sessions` must be ≤ 218.
   - Time the model build: `contention()` is O(n_a·n_b) Python per session pair and may need a sorted sweep.
   - Also try lower capacity or a shorter stagger so memory binds.
6. **Eviction model at mem/6** (superseded by TTL options).
   - Allow all gaps (`evict_candidates=None` → all).
   - Allow more than one eviction per session.
   - Allow partial eviction that keeps the prefix the next request needs.
   - Check whether mem/6 becomes feasible, and at what budget.
7. Report everything in the results table format above: mean, p50, p90 and p99 in timesteps, for Soldr sim, Mocker
   replay, sticky round robin and kv_router.
