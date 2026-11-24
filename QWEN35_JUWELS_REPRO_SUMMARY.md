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

From the JUWELS session's reports of 04:17 CET 11-24 (L2690) and its reply in appendix A; not checked independently
from Lila, and the public JSC status page could not be read by an automated fetch.

- Training: stopped at 120 trained steps at 19:29 CET 11-23; not to be resumed unless the owner asks.
- Evals: all nine finished (last one, checkpoint 100, on 11-24 about 04:00 CET).
- Slurm: the node hold was cancelled; nothing of this project is running or queued.
- Sandboxes: 0 with a JUWELS run label on Dune and on both Daytona orgs; the login-node proxy is stopped.
- Maintenance: JUWELS Booster is announced down from 2026-11-24 09:00 CET to 2026-11-30 09:00 CET (as reported by
  the JUWELS session). The JUWELS session may be unreachable in that window. Checkpoints and logs stay on the
  JUWELS filesystem.

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
| Trainer | as set by `run_rl.sh` (not re-read for this file) | 8 nodes = 4 x (Megatron TP4 x CP2), CPU optimizer offload |
| Rollout engines | vLLM engines on the remaining nodes | 20 vLLM engines x TP4 |
| Trials in flight | cap 900 | 500 |
| Step time | not re-measured here (W&B timing series of `c0gc5h63` / `0gu2cvvl`) | about 23 min (policy_train about 15 min) |
| Job length | no limit | 24 h Slurm limit: daily restart from the last checkpoint |
| Sandboxes, training | Daytona RL org | Dune first (abandoned after the 11-21 outages), then the Daytona RL org |
| Sandboxes, evals | Dune | Dune |
| Eval serving | one node per eval: trainer on 2 GPUs + 3 engines x TP2; 92-115 tokens/s | 5 engines per eval (61-87 tokens/s) or 12 engines (105 tokens/s) |
| Length | 2 epochs (104 steps) | 2 epochs + 16 steps of a third (120 steps) |

Exact layout, memory limits and launch lines on JUWELS: appendix A.

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
  ends). Paths: appendix A.

## 10. Open items, and what to do first when the cluster is back (after 2026-11-30 09:00 CET)

1. Nothing needs restarting: the run is complete for its purpose. Do not resume training past 120 unless the owner
   asks; if a longer run is wanted, stop at the end of epoch 2 or add a stop rule on the share of trials at the turn
   limit and on parse errors.
2. Decide where the checkpoints go (they exist only on the JUWELS filesystem unless appendix A says otherwise). The
   end-of-epoch-2 checkpoint (104) is the one to keep and build on.
3. Re-run the evals of 100, 110 and 120 with 12 engines if a clean third-epoch curve is wanted (they were served
   slowly, worth about 0.03).
4. Push the fixes that are still local: `bound_session_calls` (JUWELS), the Daytona task conversion scripts and the
   two sandbox patches (Lila). Commit or push only on the owner's word.
5. Open question for the owner: whether the two-run limit on the Daytona RL org also covers evals on the Daytona
   eval org (never answered; JUWELS ran its evals on Dune instead).
6. Investigate the third-epoch looping failure before any recipe with more than two epochs.

---

## Appendix A. The JUWELS session's own account

Requested from the JUWELS session on 2026-11-24 06:46 CET (status re-check, exact cluster layout and launch lines,
run names with W&B ids for the evals, checkpoint paths, code and script changes with their git state, its own log
paths). Not received when this version was pushed; the Booster maintenance starts at 09:00 CET the same day, so it
may arrive only after 2026-11-30. This section is replaced when it arrives. Everything above comes from the reports
the JUWELS session sent to Lila while the work was running.
