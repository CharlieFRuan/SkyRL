# Re-creating the Qwen3.5-9B CalibForge RL run on JUWELS Booster (A100-40GB), 2026-11-19 to 2026-11-24

Status: **finished and shut down.** The training run was stopped at 120 trained steps on 2026-11-23, nine TB2.1
evals were run, and everything on JUWELS was stopped before the Booster maintenance window. This file is the single
entry point for that effort: what was re-created, how, what came out, what broke, and where the earlier
Qwen3.5-9B context lives. It is written so that an agent with no other context can continue from it.

This file is on its own orphan branch (`charlie/qwen35-juwels-repro-1123`); the Qwen3.5-9B code branch
`charlie/calibforge-rl-1115` is unchanged.

Sources: the dated log on the Lila head node (`~/hackskyrl/ops/calibforge_rl_1110/PLAN.md`, cited as (Lnnnn) = line
number), which holds every report the JUWELS session sent, and the JUWELS session's own final account (appendix A).
Numbers are copied from those. "+-" or "SE" is one standard error over the 89 tasks.

---

## 1. Bottom line

- **The Qwen3.5-9B result replicates on a second cluster and a different GPU type.** Same recipe, same task set, same
  sandbox provider, different hardware (JUWELS Booster A100-40GB instead of Lila's B200): TB2.1 goes from a base of
  0.2472 to 0.32-0.37 by the end of epoch 2, against 0.3461 for the Lila reference. Pooled over 15 passes of the
  end-of-epoch-2 checkpoint, JUWELS reads 0.3461, the same number.
- **Training-side curves match**: mean kept-batch reward 0.440 over the first 50 steps against 0.430 for the
  reference; entropy falls from 0.48 to about 0.30 in both.
- **Two measurement effects were found and sized**: a 5-pass TB2.1 eval repeats to about +-0.02; slow serving
  during an eval costs about 0.03 (agent timeouts).
- **A third epoch does not help and breaks the policy for a while**: from about step 104 the model starts sampling
  its end-of-turn token in the middle of a sentence, the agent answers "No valid JSON found", and trials loop to the
  250-turn limit. It peaks at step 115 (27% of training trials stuck; 42% of TB2.1 trials) and is receding by 120.
  Two epochs (about 104 steps) is the recipe.

## 2. Status on 2026-11-24 (answer to "are the runs cancelled, are the nodes in maintenance")

From the JUWELS session's status report of 06:47 CET 11-24 (appendix A1, clock read on JUWELS). Not checked
independently from Lila; the public JSC status page could not be read by an automated fetch.

- Training: stopped at 120 trained steps at 19:29 CET 11-23; not to be resumed unless the owner asks.
- Evals: all nine finished (last one, checkpoint 100, on 11-24 about 04:00 CET).
- Slurm: squeue is empty (nothing running, pending or held); the last node hold was cancelled at 04:15 CET 11-24.
- Sandboxes: 0 with a JUWELS run label on Dune and on both Daytona orgs; the login-node proxy is stopped.
- Maintenance: JUWELS Booster is down from 2026-11-24 09:00 CET to 2026-11-30 09:00 CET (software updates and a
  Slurm upgrade to 25.11), per the login message of the day and the Slurm reservation `maint_booster`. At 06:47 CET
  on 11-24 it had not started. The JUWELS session may be unreachable in that window.
- Nothing was uploaded from JUWELS. **The 26 checkpoints (117 GB each) exist only on JUWELS scratch, which removes
  files not used for 90 days (late February 2027).**

## 3. Where the earlier Qwen3.5-9B context is

Already uploaded (public fork `CharlieFRuan/SkyRL`, branch `charlie/calibforge-rl-1115`, commit `aada336d`):

- `handoff/BRIEF.md`: the ground rules (shared sandbox keys, labels, kill by PID, no secrets), the stack, how to
  launch, how to evaluate. https://github.com/CharlieFRuan/SkyRL/blob/charlie/calibforge-rl-1115/handoff/BRIEF.md
- `handoff/ops/calibforge_rl_1110/FINDINGS.md`: Qwen3.5-9B findings as of 11-15 (runs, TB2.1 eval table rows 1-22,
  looping at low temperature, infra findings).
  https://github.com/CharlieFRuan/SkyRL/blob/charlie/calibforge-rl-1115/handoff/ops/calibforge_rl_1110/FINDINGS.md
- `handoff/ops/calibforge_rl_1110/run_rl.sh`: the Qwen fully-async launcher, plus `eval_agg.py`, `eval_caps.py`,
  `loop_stats.py`, `ckpt_backup_watch.sh` and the other helper scripts in that directory.
- `handoff/ops/RUN_NOTES.md`, `handoff/ops/evals_1104/` (eval settings research and results).

Only on the Lila head node (`~/hackskyrl/ops/calibforge_rl_1110/`, not uploaded):

- `PLAN.md`: the full dated log (2,700 lines), including everything after 11-15.
- `FINDINGS.md`: the current version; section 2b rows 23-26 are the four 5-pass evals of the reference run (11-16),
  reproduced in section 6 below.
- `launch_eval_1116.sh`: how the reference 5-pass evals were launched (training entrypoint with `RESUME_PATH`,
  `EVAL_BEFORE_TRAIN=true`, `EVAL_SAMPLES=5`, `PROVIDER=dune`, 110 concurrent).
- Trial archives of the four reference evals in the Lila artifact bucket
  (`.../charlie/traces/qwen35-cf-v2-daytona-b32-1116-<eval25|eval50|eval75|evalfinal>/`). TB traces are private.
- `make_daytona_tasks.py` / `make_daytona_tasks_v2.py` and the converted task set `calibforge_v2_daytona3`
  (2,598 tasks as "base snapshot + setup script" for the shared Daytona RL org). Not pushed; the JUWELS session
  received the conversion script by message and reproduced the same counts.

The sibling summary for the Nemotron-3.5-Lightning work (stopped 11-23) is on the orphan branch
`charlie/nemotron35-rl-summary-1123`, file `NEMOTRON35_LIGHTNING_RL_SUMMARY.md`.

## 4. What was being re-created: the Lila reference run

| | |
|---|---|
| Runs | `qwen35-cf-v2-daytona-b32-lr2e-6-1115` (W&B `c0gc5h63`, steps 1-30) and its resume `...-1115-r30` (W&B `0gu2cvvl`, steps 31-104); W&B project `sky-posttraining-uc-berkeley/skycap-calibforge-fully-async` |
| Model | Qwen3.5-9B, from base |
| Recipe | fully async GRPO, batch 32 task groups x 16 trials, max staleness 3, LR 2e-6, DPPO 0.15, prompt_mean, no std normalisation, zero-variance groups filtered; sampling 1.0 / 1.0; training context 131,072; one reply capped at 16,384 tokens; agent timeout 3,600 s; no summarization; thinking kept; timeouts and context-exceeded endings trained as reward 0 |
| Agent / sandboxes | in-sandbox Terminus 2 through SkyCap's sandbox relay; shared Daytona RL org; task set `calibforge_v2_daytona3` (2,598 tasks) |
| Length | 2 epochs = 104 trained steps (epoch 0 ended after 52 trained steps, epoch 1 after 104; a zero-variance filter keeps about 0.65 of the groups, so an epoch is about 52 steps, not 81) |
| Training curve | kept-batch reward by 5-step window 0.390 0.403 0.398 0.438 0.433 (steps 1-25), 0.433 / 0.458 / 0.475 (31-45); raw reward 0.23-0.36 (steps 2-10) -> ~0.40 (17-25) -> 0.42-0.50 (40-104); entropy 0.45-0.49 -> ~0.30 |
| TB2.1, 5 passes on Dune (11-16) | step 25: 0.2787; 50: 0.2989; 75: 0.3281; end of epoch 2: **0.3461** (pass@5 0.5281). Base, earlier 3-pass evals: 0.2135-0.2547 |

An earlier Dune variant of the same recipe exists too (`qwen35-cf-v2-insb-3600-cap16k-t1-b32-lr2e-6-1113`, W&B
`hwzf6cq7`; base eval 0.2247, step 25 0.2772; ended by a Dune outage at step 52). JUWELS' first attempt copied
that one.

## 5. Timeline of the JUWELS effort

| When (PST / CET) | What |
|---|---|
| 11-19 20:41 PST | The JUWELS session asks Lila for the recipe. Sent: fork, branch, launcher and launch line, recipe defaults, data layout, environment knobs, the two unpushed sandbox patches (apt mirror, tmux guard), shared-key etiquette (L2264-L2268). |
| 11-20 03:32-04:28 PST | **Short reproduction on Dune**: 28 A100-40GB nodes (trainer on 8 nodes = 4 x Megatron TP4 x CP2 with CPU optimizer offload; 20 vLLM engines x TP4; 500 trials in flight). Base TB2.1 x3: 0.2322 (pass@3 0.3596). Steps 1-2: reward 0.475 / 0.357, entropy 0.47, grad norm 0.036, logprob gap 0.010, policy_train about 910 s, step about 1,390 s, keep rate 0.65-0.70. A checkpoint is 117 GB and writes in 31 s; resume works. W&B `zdrrfumb`. Sizing note: on A100-40GB, 500 trials in flight needs about 20 TP4 engine nodes (L2293, L2300). |
| 11-20 21:37 CET | **Long run on Dune** `qwen35-cf-v2-juw-3600-cap16k-t1-b32-lr2e-6-1120` launched on the owner's go (28 nodes, 500 in flight, no in-run evals, checkpoint every 5 steps, daily restart because of the 24 h job limit) (L2338). |
| 11-21 03:30-05:00 CET | Dune DNS failures inside sandboxes and then creation failures under the combined load of three clusters: 250 setup failures per minute; the run stopped at its step-15 checkpoint; a 500-sandbox relaunch failed again (L2379, L2385). |
| 11-21 ~05:12-05:50 CET | Owner: move to the Daytona RL org, start fresh, at most two runs in total on Daytona (Lila's one + JUWELS' one). Keys handed over on the owner's approval; probes on both Daytona orgs passed; task conversion reproduced exactly (2,606 kept / 182 skipped, 2,598 after 8 dropped, same per-snapshot counts) (L2382-L2393). |
| 11-21 06:16 CET | **The run of record** `qwen35-cf-v2-juw-daytona-b32-lr2e-6-1121` (W&B `lt61lhgd`) launched from base on the Daytona RL org after a clean ramp test: `calibforge_v2_daytona3`, 500 in flight, 3 trial starts per second, no in-run evals, checkpoint every 5 steps (L2394, L2395). |
| 11-21 to 11-23 | Training with a daily restart about 21:00 CET. Step 28 on 11-21 18:32 CET, 45 at the 11-22 restart, epoch 0 ended after 51 trained steps (checkpoint 52), 80 steps on 11-22 21:5x CET, 102 on 11-23 08:25 UTC, 116 on 11-23 17:37 CET. About 23 minutes per step (L2409-L2622). |
| 11-23 18:35 CET | Looping failure found in the third epoch (section 8). |
| 11-23 18:49 CET | TB2.1 x5 evals on Dune started (owner: "2 epochs was enough. Let's run eval ... 5 reps"). |
| 11-23 19:29 CET | Training stopped at 120 trained steps; 500 sandboxes deleted by label. |
| 11-23 21:39 CET to 11-24 ~04:00 CET | Nine evals reported (section 6). |
| 11-24 04:17 CET | Everything shut down for the maintenance (section 2). |

## 6. Results

### 6a. Training side, JUWELS vs the Lila reference

| Steps | JUWELS kept-batch reward | Reference | JUWELS entropy | Reference entropy |
|---|---|---|---|---|
| 1-25, by 5-step window | 0.447 0.436 0.416 0.425 0.433 | 0.390 0.403 0.398 0.438 0.433 | 0.485 -> 0.32-0.34 | 0.45-0.49 -> 0.38-0.43 |
| 31-35 / 36-40 / 41-45 | 0.466 / 0.461 / 0.455 | 0.433 / 0.458 / 0.475 | 0.29-0.36 | 0.35-0.37 |
| mean over the first 50 | 0.440 | 0.430 | 0.30-0.33 at 46-51 | ~0.30-0.35 |
| mean over 80 / 100 / 120 | 0.460 / 0.465 / 0.472 | (run ended at 104) | 0.29-0.33 in epoch 2; 0.39-0.46 in epoch 3 | ~0.30 |

- Grad norm 0.036-0.055 and trainer-vs-rollout logprob gap 0.010-0.012 through step 108, the same as the reference.
- JUWELS' entropy fell somewhat faster than the reference's in the first 30 steps; rewards stayed level.
- Infra was clean on Daytona: 1 environment failure in the first 25,237 trials; context-exceeded 17-32% of finished
  trials (15-30% per step is normal for this recipe); agent timeouts about 1%.

### 6b. TB2.1, 5 passes each (89 tasks x 5 = 445 trials), sandboxes on Dune, same settings as the reference evals

Sampling 1.0 / 1.0, no summarization, 16,384 tokens per reply, 3,600 s per trial, about 110 trials in flight per
eval. Replies cut at the 16,384 cap: 0 in every JUWELS eval. Setup failures: 0 in every eval.

| Checkpoint (trained steps) | JUWELS avg | JUWELS pass@5 | Lila reference avg / pass@5 | Agent timeouts | Trials at >= 240 of 250 turns | "No valid JSON" share of turns | Model call |
|---|---|---|---|---|---|---|---|
| base (run 11-21) | 0.2472 | 0.4270 | (3-pass base evals 0.2135-0.2547) | 9.8% | - | 0.69% | - |
| 25 | 0.2629 | 0.4831 | 0.2787 / 0.4831 | 18.8% | 4.9% | 0.94% | 5.25 s |
| 50 | 0.3236 | 0.5169 | 0.2989 / 0.4831 | 20.8% | 8.3% | 1.17% | 5.51 s |
| 75 | 0.3236 | 0.4382 | 0.3281 / 0.4944 | 12.5% | 3.3% | 1.14% | 5.80 s |
| 100 | 0.3303 | 0.5169 | - | 8.5% | 10.5% | 2.97% | 9.51 s (slow serving) |
| 104 = end of epoch 2, first eval (5 engines) | 0.3236 | 0.5281 | **0.3461 / 0.5281** | 8.3% | 9.2% | 5.42% | 8.67 s (slow) |
| 104, same-config repeat (5 engines) | 0.3483 | 0.5056 | | 6.1% | 11.9% | 6.59% | 8.92 s (slow) |
| 104, 12 engines (the reference's serving speed) | **0.3663** | 0.5169 | | 4.5% | 11.6% | 6.69% | 5.42 s |
| 110 | 0.3213 | 0.5056 | - | 5.1% | 23.9% | 23.68% | 8.3 s (slow) |
| 115 | 0.3056 | 0.4831 | - | 0.9% | 41.7% | 46.78% | 5.36 s |
| 120 | 0.3236 | 0.4831 | - | 3.1% | 19.1% | 12.80% | 7.1 s (slow) |

Reference values for the same diagnostics, measured from the 11-16 trial archives: trials at >= 240 turns 4.9 / 6.7 /
6.0 / 6.0% at 25 / 50 / 75 / final; parse errors 0.47 / 0.60 / 0.67 / 0.32% of turns; agent timeouts 29 / 30 / 26 /
25 trials (6-7%); model call 6.18 / 7.85 / 6.44 / 5.38 s (L2656, L2661).

### 6c. How to read the eval table

- **Replication.** JUWELS minus reference at 25 / 50 / 75 / 104 (first evals): -0.016 / +0.025 / -0.005 / -0.023,
  all within about one SE of a 5-pass eval. Three evals of checkpoint 104 pooled (15 passes): 0.3461, with 54 of 89
  tasks solved at least once (L2684, L2688).
- **Eval noise.** Same checkpoint, same configuration, run twice: 0.3236 and 0.3483; paired per task +0.025, SE
  0.020. A single 5-pass eval is good to about +-0.02. (Four JUWELS evals landed on exactly 144 solved of 445; that
  was chance.)
- **Serving speed.** 12 engines against 5 engines on the same checkpoint: +0.030 (SE 0.015) against the mean of the
  two 5-engine evals; model calls 5.4 s against 8.7-8.9 s, 105 against 63-65 tokens/s. Slower model calls push more
  trials over the 3,600 s limit. The evals of 100, 110 and 120 ran slow and probably read about 0.03 low; 25, 50,
  75 and 115 ran at full speed.
- **Agent timeouts at the early checkpoints are the policy, not the infrastructure.** The base model asks for long
  waits per tool step (10.2 s); on JUWELS that habit lasted through step 75 (11.6-12.5 s) and dropped to 7.2 s only
  by 104, while the reference run had already cut it by step 25 (8.7 -> 4.5 s over the run). Tool steps took the
  requested wait plus 0.03-0.08 s, so sandbox speed was not the cause. With timed-out trials excluded, JUWELS is
  level with the reference: 0.321 / 0.406 / 0.366 / 0.351 against 0.297 / 0.320 / 0.347 / 0.363 (L2660-L2675).

## 7. What differed from the Lila run

| | Lila reference | JUWELS |
|---|---|---|
| GPUs | Lila: an Anyscale Ray cluster of 8-GPU nodes (layout in `handoff/BRIEF.md` and `run_rl.sh`) | JUWELS Booster, A100-40GB x 4 per node, 28 nodes |
| Trainer | as set by `run_rl.sh` (not re-read for this file) | 8 nodes = 4 replicas x (Megatron TP4 x CP2), Adam state offloaded to host memory: a 128k-token row does not fit 40 GB with TP4 alone |
| Rollout engines | vLLM engines on the remaining nodes | 20 vLLM engines x TP4 (8 and 10 engines collapsed from KV-cache eviction at 500 trials in flight) |
| Trials in flight | cap 900 | 500 |
| Step time | 15-22 min (training pass 12-13.5 min) | mean 30 min incl. restarts and epoch ends; typical 23-35 min (training pass 11-23 min) |
| Job length | no limit | 24 h Slurm limit: daily restart from the last checkpoint |
| Sandboxes, training | Daytona RL org | Dune first (abandoned after the 11-21 outages), then the Daytona RL org |
| Sandboxes, evals | Dune | Dune |
| Eval serving | one node per eval: trainer on 2 GPUs + 3 engines x TP2; 92-115 tokens/s | 5 engines per eval (61-87 tokens/s) or 12 engines (105 tokens/s) |
| Length | 2 epochs (104 steps) | 2 epochs + 16 steps of a third (120 steps) |

Exact layout, memory limits, costs and the full run list with W&B ids: appendix A2 and A3.

## 8. The third-epoch looping failure (found on JUWELS; the reference run stopped before it could appear)

- **Symptom.** From about policy step 104 the model samples `<|im_end|>` in the middle of a sentence. The reply ends
  after 5-25 tokens with finish_reason "stop" (not the token cap, not serving); Terminus answers "No valid JSON
  found"; the exchange repeats to the 250-turn limit. Stuck trials score 0.01-0.09.
- **Measured on one trajectory at step 117**: the first cut sampled the end token at logprob -6.0 (0.25%) and
  recovered; the persistent loop began near reply 59 (prompt 37.7k tokens), after which the end token had 2-25%
  probability at each position (L2638).
- **It ramps by policy step, it is not a step change** (31,774 training records, steps 81-117; "cut" = replies whose
  final end token was sampled at logprob < -1, "stuck" = trajectories with >= 240 replies) (L2640):

| Policy step | 81-100 | 104 | 105 | 106 | 107 | 109 | 111 | 113 | 115 | 116 | 117 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cut, % of replies | 0.04-1.1 | 2.6 | 5.1 | 9.4 | 12.5 | 13.8 | 23.0 | 31.4 | **35.3** | 27.6 | 17.9 |
| stuck, % of trials | 0-1.2 | 1.3 | 2.4 | 4.9 | 7.6 | 7.8 | 13.5 | 24.4 | **26.7** | 13.9 | 10.4 |

- **Trainer signals lag it**: grad norm 0.140 / 0.119 / 0.081 at steps 116-118 (usual 0.03-0.05), logprob gap 0.012
  -> 0.017 -> 0.025 (run maximum before: 0.012), entropy 0.40 -> 0.46 (0.32 in epoch 2); kept-batch reward stayed at
  0.43-0.51 because stuck trials sit in low-variance groups. By 119-120 the signals were partly back (gap 0.019 /
  0.016, grad norm 0.050 / 0.063).
- **Nothing changed at the onset**: no relaunch (the last one was at step 81), same engines, sampling and template.
- **At eval** the same failure shows as the turn-limit and parse-error columns of the table in 6b: 104 is already
  affected (9-12% at the turn limit, 5-7% parse errors against 0.3-0.7% in the reference), 115 is the peak.
- **Not established**: the cause. Candidates not tested: reward-0 training of stuck trials reinforcing short
  replies in low-variance groups; repeated visits of the same tasks (third pass); a possible precursor in the
  reference run's final checkpoint, where replies of 80 tokens or fewer rose to 5.2% of turns without parse errors.
- **Check on Nemotron**: no such loop in any Nemotron run (trials at >= 240 turns stayed at 0.1-1.0%).

## 9. Infrastructure findings from this effort

- **Daytona session calls can hang forever.** Harbor's Daytona environment calls `process.create_session` (and
  `get_session_command`) without a request timeout; a trial wedged on an unresponsive sandbox cannot be freed by the
  trial timeout, and one wedged trial keeps an epoch from ending (the first epoch boundary on JUWELS stalled about
  40 minutes on 4 trials; Lila later lost 60 minutes the same way). JUWELS' local fix, `bound_session_calls`: a
  120 s bound on both calls; live from 11-22 21:36 CET, no false trips in about 20 hours. Not pushed (L2490, L2549).
- **The Daytona RL org's sandbox listing lags by hours**: label censuses over-count and delete-by-label returns
  not-found for sandboxes that are already gone; creates, execs and deletes themselves are fine (L2579).
- **Dune does not hold the combined load of three clusters**: DNS failures inside sandboxes and creation timeouts
  on 11-21 with about 2,000 sandboxes on the key; one run should ramp at a time, about one start per second (L2385).
- **One training task's setup script broke from upstream package drift** (a pip-tools release needing a newer pip
  than the image has); drop or pin it before the next full-set run.
- **Shared-org etiquette**: at most two training runs in total on the Daytona RL org (owner's rule of 11-21); label
  every sandbox with owner and run; delete only by both labels; no image builds or new snapshots on the RL org;
  TB2.1 cannot run on the RL org, so evals went to Dune.
- **Checkpoints** are 117 GB each on JUWELS; kept: every multiple of 5 up to 120, plus 52 and 104 (the two epoch
  ends). Scratch only, 90-day removal; paths in appendix A2.
- **JUWELS specifics**: compute nodes have no internet (a CONNECT proxy on a login node carries sandbox, W&B and
  storage traffic); the venv is unpacked to /dev/shm on every node; a 3-epoch run writes millions of trial files
  and exhausted the first scratch project's inode quota on 11-22 (finished trial dirs are now packed into tar
  archives); 24 h job limit, each daily restart costs the in-flight rollouts (the next step took 72-83 min).

## 10. Open items, and what to do first when the cluster is back (after 2026-11-30 09:00 CET)

1. Nothing needs restarting: the run is complete for its purpose. Do not resume training past 120 unless the owner
   asks; if a longer run is wanted, stop at the end of epoch 2 or add a stop rule on the share of trials at the turn
   limit and on parse errors.
2. **Copy the checkpoints worth keeping off JUWELS scratch before late February 2027** (90-day removal; nothing was
   uploaded). The end-of-epoch-2 checkpoint (`global_step_104`) is the one to keep and build on; 100 is the last
   one before the looping shows in training.
3. Re-run the evals of 100, 110 and 120 with 12 engines if a clean third-epoch curve is wanted (they were served
   slowly, worth about 0.03). For any comparison with the Lila numbers use 12 engine nodes per eval.
4. Push the fixes that are still local: `bound_session_calls` (JUWELS), the Daytona task conversion scripts and the
   two sandbox patches (Lila). Commit or push only on the owner's word.
5. Open question for the owner: whether the two-run limit on the Daytona RL org also covers evals on the Daytona
   eval org (never answered; JUWELS ran its evals on Dune instead).
6. Investigate the third-epoch looping failure before any recipe with more than two epochs.
7. The owner's stop at 120 steps reached JUWELS only as a relay through Lila; the JUWELS session acted on it but
   notes that it was never typed there. First steps on JUWELS after the maintenance are listed in appendix A6.

---

## Appendix A. The JUWELS session's own account (written 2026-11-24 06:47 CET, in three parts)

Copied as received. `$WORK`, `$SCR` and `$SCR_OLD` are placeholders for the project's work and scratch directories on
JUWELS; the same text is saved there as `$WORK/JUWELS_QWEN_REPRO.md` (19 KB). No keys, hosts, addresses, account
names or task names are included.

### A1. Status at 2026-11-24 06:47 CET (clock read on JUWELS)

- **Jobs**: none. squeue is empty (nothing running, pending or held). The last hold was cancelled at 04:15 on Nov 24;
  no Booster node is allocated or reserved by us. sacct: every job since Nov 18 is CANCELLED (by the session) or
  FAILED (one 0-second dev job on Nov 20).
- **Maintenance**: Booster is in maintenance from 2026-11-24 09:00 to 2026-11-30 09:00 CET (software updates and a
  Slurm upgrade to 25.11). Sources: the login message of the day and the Slurm reservation `maint_booster` with
  exactly those start and end times. At 06:47 it had not started yet (186 nodes allocated to other users, 133 idle).
  Breaking change announced: with Slurm 25.11 a job with only `--gpus-per-task` fails; `--gres=gpu:X` is needed. The
  hold script already uses `--gres=gpu:4` and no srun line uses `--gpus-per-task`.
- **Training**: stopped at 120 trained steps on Nov 23 19:29 (instruction relayed through the Lila session: two
  epochs are enough, run the evals). Not to be relaunched unless the owner asks.
- **Sandboxes**: 0 with a JUWELS run label on Dune (count after the last cleanup, 04:15) and 0 on both Daytona orgs
  (census 04:15).
- **Login-node proxy**: stopped at 04:15.
- **Uploads**: none. No checkpoint or dataset was uploaded from JUWELS (no HF upload); metrics are on W&B only.

### A2. The reproduction

**Cluster layout**

| Part | JUWELS Booster | Why it differs from the B200 setup |
|---|---|---|
| Node | 4 x A100-40GB, 48 cores, 503 GiB | |
| Training run | 28 nodes = 112 GPUs, 24 h job limit | |
| Trainer | 8 nodes = 4 replicas, each Megatron TP4 x CP2 (2 nodes per replica), Adam state offloaded to host memory (optimizer_cpu_offload, fraction 1.0, precision-aware optimizer) | A 128k-token row does not fit 40 GB with TP4 alone (36 GB peak and growing); TP4 x CP2 peaks at 26.8 GB per GPU. The launcher's Qwen3.5 family has no optimizer offload because B200s do not need it |
| Engines | 20 nodes, one vLLM engine with TP4 per node, gpu_memory_utilization 0.80, 262,144 context | With 500 trajectories in flight the KV cache must hold about 500 live contexts. 8 engines at 0.65 and 10 engines at 0.80 both collapsed after 20-30 min (prefix eviction); 20 engines ran at 23-28% KV use, 97% prefix-cache hits, nothing waiting |
| Eval runs | 7 nodes each (2 trainer nodes that only load the checkpoint + 5 engines), or 14 nodes (2 + 12 engines); 110 trials in flight | Separate steps on a hold |
| Network | Compute nodes have no internet; an outbound CONNECT proxy on a login node carries sandbox, W&B and storage traffic | |
| Storage | venv unpacked to /dev/shm on every node (imports from the parallel file system take 83 s per process, 12 s from RAM); run dirs on scratch | The first scratch project ran out of inodes on Nov 22 (a 3-epoch run writes millions of trial files); run dirs moved, finished trial dirs are packed into tar archives |

Environment deviations from uv.lock: vllm-router wheel retagged for glibc 2.34; nvidia-cuda-nvcc / crt / nvvm pinned
to 13.0.88 (the first training step fails otherwise); ray[default] for GPU metrics.

**Recipe (the Lila Qwen3.5-9B batch-32 recipe, unchanged)**

| Setting | Value |
|---|---|
| Model | Qwen3.5-9B (public weights) |
| Trainer | fully async, `run_fully_async_terminus.sh` through `juwels/run_rl_juwels.sh` |
| Batch | 32 groups x 16 trajectories, sample_full_batch, zero-variance groups dropped |
| LR | 2e-6 |
| Loss | DPPO (delta_low = delta_high = 0.15), prompt-mean loss reduction, as in the launcher |
| Max staleness | 3 steps |
| In flight | 500 trajectories, 3 started per second |
| Context | trainer 131,072 tokens; engines 262,144 |
| Reply cap | 16,384 tokens per turn |
| Agent | Terminus (installed in the sandbox), 250 turns max, 3,600 s agent limit, no summarization, timeouts trained on |
| Sampling | temperature 1.0, top_p 1.0, top_k off, min_p 0 (train and eval) |
| Epochs | 3 planned; stopped in the third |
| Checkpoints | every 5 steps, plus epoch ends |
| In-run evals | none (owner's choice); checkpoints evaluated afterwards |
| Sandboxes | steps 1-15 of the first long run on Dune; the reported run on the Daytona RL org (converted task set: base snapshot + per-task setup script, no image builds) |
| Task set | CalibForge v2: 2,788 tasks on Dune; the Daytona conversion has 2,598. Nominal 81 steps per epoch, but about 37% of groups are dropped by sample_full_batch, so an epoch ends after about 51-52 steps |
| Eval | Terminal-Bench 2.1 (89 tasks) x 5 passes on Dune, through the SkyRL entrypoint (EVAL_BEFORE_TRAIN=true, resume from the checkpoint) |

**Runs (W&B project `sky-posttraining-uc-berkeley/skycap-calibforge-fully-async`)**

| Run | W&B id(s) | Dates (CET) | Steps | Checkpoints | Why it ended |
|---|---|---|---|---|---|
| `qwen35-cf-v2-juw-smoke2-1120`, `...-b32-lr2e-6-e2e-1120`, `-e2e2-`, `-e2e3-` | 4i11a0yu, 2ibxu4nj, q1a93c2g, 9smy5i4v | Nov 20 | 0-1 | none kept | Bring-up: engine count and memory layout (8 and 10 engines collapsed) |
| `qwen35-cf-v2-juw-b32-lr2e-6-e2e4-1120` | zdrrfumb, ekvqlkmg | Nov 20 | 3 (step-0 TB2.1 x3: 0.2322, pass@3 0.3596) | global_step_2 | Validation of the recipe on 28 nodes incl. checkpoint + resume; done |
| `qwen35-cf-v2-juw-3600-cap16k-t1-b32-lr2e-6-1120` (Dune) | s90fdctg, wsc6gsv9 | Nov 20 21:37 - Nov 21 05:12 | 15 (mean reward 0.422; Lila 0.420) | 5, 10, 15 | Dune sandbox DNS outage on Nov 21 from 04:29; owner: switch to Daytona, start fresh |
| `qwen35-cf-v2-juw-daytona-ramp300-1121`, `-dtn-smoke5-`, `-dtn-smoke5b-`, `-dtneval-smoke3-`, `-dtneval-smoke3b-` | n716dibk, ri6mxc07, 4roapcxa, 14gq9hpo, 381yty15 | Nov 21 | 0 | none | Daytona smokes (relay over preview URLs through the proxy) |
| **`qwen35-cf-v2-juw-daytona-b32-lr2e-6-1121`** (Daytona; the reported run) | **lt61lhgd** (steps 1-45), eu12glcn (46-80), f135co7d (restart that hit the inode quota), 8oi0b4ee (81-120) | Nov 21 06:16 - Nov 23 19:29 | **120** | every 5 from 5 to 120, plus 52 and 104 (epoch ends; global_step_104 = 103 trained steps = end of epoch 2). 117 GB each, 26 in all | Stopped on the owner's relayed instruction; Booster maintenance from Nov 24 |

Checkpoint paths: `$SCR/runs/qwen35-cf-v2-juw-daytona-b32-lr2e-6-1121/ckpts/global_step_N` (N <= 80 are symlinks into
`$SCR_OLD/runs/<same run>/ckpts/`). **Scratch only; the site removes scratch files not used for 90 days.**

**Evals (TB2.1 x5 on Dune; run dirs `$SCR/runs/<run>`, base under `$SCR_OLD/runs` + a tar archive)**

| Run | W&B id | Checkpoint | Engines | Finished |
|---|---|---|---|---|
| `qwen35-cf-v2-juw-eval-base-x5-1120` | 6ag9xhev | base | 12 | Nov 21 |
| `qwen35-cf-v2-juw-eval-step25-x5-1123` | p1grpmkw | 25 | 5 | Nov 23 21:39 |
| `...-step50-x5-1123` | fork0zup | 50 | 5 | Nov 23 22:48 |
| `...-step75-x5-1123` | rz5moazz | 75 | 5 | Nov 23 22:53 |
| `...-step100-x5-1123` | zi5j2ds9 | 100 | 5 | Nov 24 04:14 |
| `...-step104-x5-1123` | x8uya1hq | 104 | 5 | Nov 23 21:36 |
| `...-step104-r2-x5-1123` | ibvowikc | 104 (same-config repeat) | 5 | Nov 24 01:22 |
| `...-step104-e12-x5-1123` | aq12u40u | 104 (serving-speed test) | 12 | Nov 24 02:45 |
| `...-step110-x5-1123` | vq1bd1oz | 110 | 5 | Nov 24 01:12 |
| `...-step115-x5-1123` | 0mb258o6 | 115 | 5 | Nov 23 23:32 |
| `...-step120-x5-1123` | nx6eqk3c | 120 | 5 | Nov 23 23:55 |

### A3. Differences from the Lila run and what they cost

| | JUWELS (112 A100-40GB) | Lila |
|---|---|---|
| Step time | mean 30 min over 120 steps (wall time incl. restarts and epoch ends); typical step 23-35 min | 15-22 min |
| Training pass | 11-23 min (670-1,360 s; grows with batch length, 49k -> 90k tokens per trajectory), 25-36k tok/s | 12-13.5 min |
| Rollout wait per step | 2-17 min normally | |
| Restarts | 24 h job limit: a new hold and relaunch every day (Nov 22 01:43 at step 45, Nov 22 21:23 at step 80). Each costs the in-flight rollouts: the next step took 83 min and 72 min. One extra relaunch when the scratch project ran out of inodes | none forced |
| Epoch ends | the epoch-boundary wait plus the first step of the new epoch took 2 h 03 min and 1 h 26 min | |
| Sandboxes | Dune for the first 15 steps (outage), then the Daytona RL org for the reported run; all through a login-node proxy | Daytona |
| Eval serving | 5 engines per eval: 5.3-5.8 s per model call at checkpoints 25 / 50 / 75 / 115, but 8.3-9.5 s (61-65 tok/s) at 100 / 104 / 110 and 7.1 s at 120. 12 engines: 5.42 s at 105 tok/s | 5.38 s, 105 tok/s |
| Cost of slow eval serving | about +0.030 avg (SE 0.015) at checkpoint 104 between 5 and 12 engines, through agent timeouts (37 and 27 vs 20) | |
| Compute | about 61 h x 28 nodes for the 120 steps; the eval hold used about 270 node-hours (28 nodes for 9.2 h, then 8 for 1.5 h) | |

### A4. Results

**TB2.1 x5 per checkpoint** (Dune, 89 tasks, 445 trials; avg = mean per-task score; a trial that hit the 1 h agent
limit counts 0)

| Checkpoint | Engines | avg | pass@5 | Lila avg / pass@5 | Agent timeouts | Out of context | Cap hits (16,384) | Trials at >= 240 turns | "No valid JSON found" share of turns | Model call (tok/s) | Wait asked per tool step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 12 | 0.2472 | 0.4270 | - | 44 | n/a | 0 | 4.5% | 0.69% | 6.98 s (84) | 10.2 s |
| 25 | 5 | 0.2629 | 0.4831 | 0.2787 / 0.4831 | 84 | 5 | 0 | 4.9% | 0.94% | 5.25 s (87) | 12.5 s |
| 50 | 5 | 0.3236 | 0.5169 | 0.2989 / 0.4831 | 93 | 4 | 0 | 8.3% | 1.17% | 5.51 s (81) | 11.6 s |
| 75 | 5 | 0.3236 | 0.4382 | 0.3281 / 0.4944 | 56 | 0 | 0 | 3.3% | 1.14% | 5.80 s (86) | 11.6 s |
| 100 | 5 | 0.3303 | 0.5169 | - | 38 | 10 | 0 | 10.5% | 2.97% | 9.51 s (61) | 7.1 s |
| 104 (end of epoch 2) | 5 | 0.3236 | 0.5281 | 0.3461 / 0.5281 (final) | 37 | 4 | 0 | 9.2% | 5.42% | 8.7 s (65) | 7.2 s |
| 104 repeat | 5 | 0.3483 | 0.5056 | same | 27 | 3 | 0 | 11.9% | 6.59% | 8.92 s (63) | 6.9 s |
| 104, matched serving speed | 12 | **0.3663** | 0.5169 | same | 20 | 6 | 0 | 11.6% | 6.69% | 5.42 s (105) | 7.1 s |
| 110 | 5 | 0.3213 | 0.5056 | - | 23 | 0 | 0 | 23.9% | 23.7% | 8.27 s (61) | 6.5 s |
| 115 | 5 | 0.3056 | 0.4831 | - | 4 | 0 | 0 | 41.7% | 46.8% | 5.36 s (65) | 6.1 s |
| 120 | 5 | 0.3236 | 0.4831 | - | 14 | 2 | 0 | 19.1% | 12.8% | 7.10 s (67) | 5.9 s |

- Setup failures: 0 in every eval. The Lila reference evals: agent timeouts 29 / 30 / 26 / 25, trials at the turn
  limit 4.9-6.7%, "No valid JSON found" 0.3-0.7% of turns at the four checkpoints.
- Noise: two same-config 5-pass evals of checkpoint 104 differ by +0.0247 (SE 0.0197). Pooled 15 passes at 104:
  0.3461. The equal averages 0.3236 at 50 / 75 / 104 / 120 are real (144 solved trials of 445 each time, different
  tasks).
- Timeouts at 25-75 come from the policy asking for long waits per tool step (the base model's habit, 10.2 s);
  sandbox overhead beyond the requested wait is 0.01-0.08 s per tool step.
- Reading: the Lila curve (base to about 0.33-0.35 after two epochs) is replicated within 5-pass noise; at matched
  serving speed the JUWELS end-of-epoch-2 checkpoint is above Lila's final. The third epoch added no score.

**Training side (kept-batch mean reward, 10-step windows)**

| Steps | Reward | Lila, same steps | Entropy | Grad norm | Rollout-trainer logprob gap | Step time | Tokens per trajectory |
|---|---|---|---|---|---|---|---|
| 1-10 | 0.441 | 0.396 | 0.437 | 0.045 | 0.0107 | 1,591 s | 53.9k |
| 11-20 | 0.421 | 0.418 | 0.368 | 0.044 | 0.0102 | 1,443 s | 58.5k |
| 21-30 | 0.421 | 0.447 | 0.344 | 0.040 | 0.0101 | 1,525 s | 62.8k |
| 31-40 | 0.463 | 0.446 | 0.348 | 0.039 | 0.0106 | 1,482 s | 72.1k |
| 41-50 | 0.454 | 0.444 | 0.322 | 0.033 | 0.0096 | 1,990 s | 82.2k |
| 51-60 | 0.504 | - | 0.303 | 0.048 | 0.0091 | 2,063 s | 78.1k |
| 61-70 | 0.480 | - | 0.298 | 0.031 | 0.0093 | 1,716 s | 83.4k |
| 71-80 | 0.495 | - | 0.324 | 0.032 | 0.0100 | 1,788 s | 78.2k |
| 81-90 | 0.505 | - | 0.343 | 0.032 | 0.0104 | 1,875 s | 76.8k |
| 91-100 | 0.465 | - | 0.340 | 0.029 | 0.0103 | 1,836 s | 89.2k |
| 101-110 | 0.538 | - | 0.379 | 0.033 | 0.0111 | 1,975 s | 72.1k |
| 111-120 | 0.483 | - | 0.399 | 0.062 | 0.0145 | 1,980 s | 51.6k |

(The "Lila, same steps" column is from the per-step CSV of the Lila Daytona run, steps 1-50.) Mean over 120 steps
0.472 (Lila 0.430 over its first 50 steps, about 0.45 over 51-104). Epoch means: first (1-51) 0.439, second (52-103)
0.493, third so far (104-120) 0.509. Entropy fell to about 0.30 in the second epoch and rose again from step 81, to
0.40 in the third epoch; grad norm 0.14 / 0.12 / 0.08 at steps 116-118 and the logprob gap rose to 0.017-0.025 at
117-119.

**Third-epoch looping failure**

From policy step 104 on the model increasingly samples its end token in the middle of a reply (finish reason
"stop"). Terminus rejects the reply ("No valid JSON found"), the retry is cut again, and the trial runs to the
250-turn limit with reward near 0. Per policy step in training (share of replies cut / share of trials at >= 240
turns; final counts, slightly different from the interim table in section 8):

| Policy steps | Replies cut | Trials at >= 240 turns |
|---|---|---|
| 81-100 | 0.04-0.8% | 0-1.2% |
| 104 | 2.8% | 1.5% |
| 105-109 | 6-12% | 2-7% |
| 110-112 | 15-26% | 9-19% |
| 113-115 | 31-32% | 21-24% (peak) |
| 116 | 26.8% | 13.6% |
| 117 | 18.7% | 8.7% |

At eval (table above) the same shape: 9-12% of trials at the turn limit at 104, 24% at 110, 42% at 115, 19% at 120;
checkpoint 100 already shows 10.5% of trials and 3.0% of turns. The trials that do not loop keep improving (about
0.51 mean at 115, 0.40 at 120, 0.36 at 104), so the policy got better on what it completes while the looping removed
up to 42% of the trials. It does not depend on serving speed (11.6% with 12 engines). Cause not established;
candidates are the entropy rise of the third epoch and trained cut replies reinforcing themselves (the end-token
logprob goes from -6 at a first cut to -1 to -4 once cut replies are in the context).

### A5. Code and script changes on JUWELS

SkyRL checkout: branch `charlie/calibforge-rl-1115`, HEAD `aada336d`, remote = the public fork. Nothing committed or
pushed from JUWELS; 9 files are modified in the working tree (189 insertions, 24 deletions). No diff file was
exported (no outgoing transfer from JUWELS besides these messages); the changes are small enough to re-apply from
this table, and the two that came from Lila / JUPITER exist there too.

| File | Change |
|---|---|
| `harbor_skycap/fast_exec.py` | `bound_session_calls()`: 120 s limit (`SKYCAP_SESSION_CALL_TIMEOUT_SEC`, 0 disables) wrapped around the provider SDK's `create_session` / `get_session_command`. Harbor passes these calls no timeout; a sandbox whose toolbox stopped answering held its trial forever, and with sample_full_batch the epoch then never ended |
| `harbor_skycap/outage_guard.py`, `harbor_generator.py` | Environment failures (tmux missing after the agent install, sandbox start / agent setup timeout, relay stub not started, runner install failed, agent install timeout with exit 124) go back to the gate uncounted, at most `SKYCAP_ENV_RETRIES=3` times per trial (JUPITER's patch, extended by the message-matched cases) |
| `harbor_skycap/terminus_installed.py` | TmuxMissingError after the agent install instead of a silent 0-reward trial (JUPITER's patch) |
| `harbor_skycap/daytona_setup.py` | `SKYCAP_APT_MIRROR`: point Ubuntu sources at a mirror before the task setup (Lila's patch) |
| `harbor_skycap/sandbox_relay/manager.py` | `trust_env=True` so the relay manager's HTTP session uses the proxy |
| `skyrl/backends/skyrl_train/workers/worker.py` | NUMA: prefer local memory (numa_set_preferred) instead of a strict bind (a strict bind got checkpoint loading OOM-killed) |
| `harbor_skycap/run_fully_async_terminus.sh`, `run_sync_terminus.sh` | MODEL_FAMILY switch (qwen3.5 / nemotron-3.5) and env knobs for GPUs per node (4 here, was fixed at 8) and Megatron context-parallel / expert-parallel sizes (were fixed at 1) |

(`harbor_skycap/` = `examples/train_integrations/harbor_skycap/`.)

JUWELS-only scripts in `$WORK` (not in any repo):

| Script | Purpose |
|---|---|
| `juwels/hold_nodes.sbatch`, `calibforge_juw.sh`, `ray_cluster.sh`, `run_rl_juwels.sh` | hold nodes, start Ray inside the hold, launch training (PROVIDER=dune / daytona / daytona_eval, RESUME_PATH) |
| `juwels/eval_ckpt_juw.sh`, `eval_launch_juw.sh` | checkpoint eval on separate nodes of a hold (SUFFIX, NODES_CSV) |
| `juwels/eval_report.py`, `eval_timing.py`, `eval_summary.py`, `ops/eval_caps.py` | per-eval report: score, endings, timeouts, cap hits, turn-limit share, "No valid JSON found" share, seconds per model call / tool step |
| `juwels/step_table.sh`, `trial_stats.py`, `rl_status.sh`, `engine_trend.sh`, `pack_trials.sh` | per-step table, trial statistics, engine load, packing trial dirs into tar archives |
| `ops/dune_cleanup.py`, `dune_probe.py`, `make_daytona_tasks_v2.py`, `build_nemo_env.sh`, `stage_venv_tar.sh` | sandbox cleanup by owner + run label, provider probe, Daytona task conversion, env build, venv staging |
| `relay/login_services.sh` | the login-node CONNECT proxy (start / stop / status) |
| looping diagnostics | session scratch scripts (early-stop share per policy step from the SkyCap records; trials at the turn limit by start hour), not kept in `$WORK`; their results are in `FINDINGS.md` there and in A4 |

### A6. Open, and what to do first after Nov 30 (the JUWELS session's list)

1. The owner has not confirmed in the JUWELS session itself the relayed instruction to stop at 120 steps (the
   session acted on it; training stays stopped).
2. Decide whether to continue: more training from global_step_100 / 104 (before the looping) or 120, with a change
   against the early end token (not designed yet), under a new run name containing "juw". Not started.
3. Matched-speed evals (12 engines) exist only for checkpoint 104; 25 / 50 / 75 / 100 / 110-120 ran with 5 engines.
   For comparisons with the Lila numbers use 12 engine nodes per eval.
4. Local SkyRL patches are uncommitted; commit or push only when the owner asks.
5. First steps after the maintenance: read `$WORK/PLAN.md` (tail) and `$WORK/FINDINGS.md`; sbatch a hold
   (`--gres=gpu:4` is already there); start the proxy (the login node may differ; it writes a new proxy env file);
   provider probe; then `juwels/eval_launch_juw.sh <hold> <offset> <count> <N> 110`, or a training launch.
6. **Checkpoints are on scratch only (117 GB each, 26 of them); copy the ones worth keeping before the 90-day limit
   (late February 2027).**

### A7. Log and result files on JUWELS

| File | Size | Content |
|---|---|---|
| `$WORK/PLAN.md` | 152 KB, 535 lines | dated log of every action, Nov 16 - Nov 24 |
| `$WORK/FINDINGS.md` | 51 KB, 90 lines | results table (one row per finding / eval / 5-step training window) |
| `$WORK/READY.md` | 7.6 KB | the validated launch recipe of Nov 20 |
| `$WORK/JUWELS_QWEN_REPRO.md` | 19 KB | this account |
| `$SCR/logs/<run>.log`, `.jobs`, `_step.log` | per run | driver log (contains trajectory text: private), job notes |

The JUWELS session's memory notes hold the absolute roots, the Slurm account and the cluster quirks; they stay on
JUWELS.
