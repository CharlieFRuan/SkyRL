# Qwen3.5-9B on CalibForge v2: key findings of the 11-10 / 11-11 RL run

Running notes are in `PLAN.md` (dated log, heartbeats, the end-of-run fix list). This file is the short version:
what we learned, with the numbers, and what is still open. Last updated 2026-11-13 08:45 PST.

**2026-11-12 02:45 PST: the trainer node was terminated by the provider. Run `r75` died at step ~97 and every
checkpoint (25 / 50 / 75) was lost with the node's local disk; none had been uploaded. The numbers below survive
(W&B, this file); the weights do not. The in-run eval at step 100 never happened.**

## 1. The runs

| Run | What | Status |
|---|---|---|
| `qwen35-cf-v2-insb-3600-cap16k-nosum-1110b` (W&B `yjkpuxep`) | steps 0-76, eval at 0.6 / 0.95 / top_k 20 | stopped 11-11 19:29 at step 76; checkpoints lost 11-12 |
| `qwen35-cf-v2-insb-3600-cap16k-nosum-1111-r75` (W&B `x64cheop`) | resumed from step 75, eval at training sampling (1.0 / 1.0) | died at step ~97 with the node, 11-12 02:45 |
| `qwen35-cf-v2-insb-3600-cap16k-t1-1112` (W&B `ait0r6tf`) | fresh start from step 0 after the node loss, eval at 1.0 / 1.0; driver <node>; checkpoints backed up to the artifact bucket | **stopped 11-13 16:03 PST after step 126** (user); evals at 0 / 25 / 50 / 75 / 100 / 125; checkpoints 25 / 50 / 75 / 100 / 112 (epoch 1) / 125 in `$ANYSCALE_ARTIFACT_STORAGE/charlie/ckpts/<run>/`; traces (53,731 trials, no terminal recordings, 14.8 GB) in `.../charlie/traces/<run>/` |
| `qwen35-cf-v2-insb-3600-cap16k-t1-b32-lr2e-6-1113` (W&B `hwzf6cq7`) | from base again: **batch 32** (group 16, staleness 3, 128 generation workers), **LR 2e-6**, MAX_CONCURRENCY 1600, TRAJECTORIES_PER_SECOND 3; everything else as `t1-1112`; eval before train (second base eval at 1.0, to gauge noise) | launched 11-13 16:06 PST; step 0 eval 0.2247, step 25 eval 0.2772; **Dune API outage 06:00-10:01 PST 11-14**: steps 36 and 38-52 trained on empty batches, step-50 "eval" 0.0 is not a result; stopped 08:31 at step 52; checkpoints 25 (last clean) and 50 in S3 |
| `qwen35-cf-v2-insb-3600-cap16k-t1-b32-lr2e-6-1114-r25` (W&B `za91stsl`) | resumes the run above from its checkpoint 25 (1,267 task groups consumed, 467 of them filtered); same settings; outage guard on (`SKYCAP_OUTAGE_GUARD=1`), no eval before train | launched 11-14 10:07 PST; Dune refused the key (403) from 10:14; six empty steps; stopped 10:52, no checkpoint (not a result) |
| `qwen35-cf-v2-daytona-b32-lr2e-6-1114` (W&B `r0kd17zc`) | **from base, on the shared Daytona org**: same recipe as the b32 Dune run (batch 32, group 16, staleness 3, LR 2e-6); 2,606 of the 2,788 tasks as base snapshot + setup script, 1 vCPU / 4 GiB (4 / 8 for ~9%) / 10 GiB; cap 900; **no in-run evals** (TB2.1 cannot run there: evaluate uploaded checkpoints on Dune later); checkpoints every 5 steps, uploads 25 x N | launched 11-14 17:30 PST; stopped at step 25 (user); checkpoint 25 eval on the eval org 0.2659 / pass@3 0.3933 vs base 0.2547 / 0.3596 (+0.011, within noise): 114 of its 301 agent timeouts were trials stuck on a model call the preview relay lost, trained as reward 0 |
| `qwen35-cf-v2-daytona-b32-lr2e-6-1115` (W&B `c0gc5h63`) | **from base again on the shared Daytona org, with the fixes**: acknowledged relay delivery (lost poll responses are redelivered), stall rule (an agent timeout with a model call > 900 s is an error, not reward 0), outage guard, setup limit 1,800 s; task set `calibforge_v2_daytona3` (2,598 tasks; second scan `cf-v2-daytona-scan-1115`: 0 stalled of 25 timeouts, 0.2% pre-agent failures, all 4 dropped); otherwise the same recipe (batch 32, group 16, staleness 3, LR 2e-6, cap 900, no in-run evals, checkpoints every 5 steps, uploads 25 x N) | launched 11-15 05:05 PST (training from 05:09); stopped 14:14 at step 30 for the checkpoint 25 eval (user): **0.2809 / pass@3 0.4157** vs base 0.2547 and 0.2172 on the same org; checkpoint 25 in S3 |
| `qwen35-cf-v2-daytona-b32-lr2e-6-1115-r30` (W&B `0gu2cvvl`) | resumes the run above from its checkpoint 30; same settings, plus `SKYCAP_CREATE_TIMEOUT_SEC=120` (a create that hangs on a broken Daytona runner is abandoned after 120 s and retried) | resumed 11-15 16:38 PST |

Settings common to both: fully async, DPPO 0.15, LR 1e-6, 16 tasks x 16 samples per step, staleness 3; training
sampling 1.0 / 1.0; Terminus 2 inside the sandbox (Dune, 4 vCPU / 8 GB / 20 GB); 3,600 s agent timeout; training
context 131,072 and eval context 262,144; thinking kept; no summarization; one reply capped at 16,384 tokens; eval =
Terminal-Bench 2.1 x 3 samples (267 trials) every 25 steps.

## 2. Eval and training results

Eval at 0.6 / 0.95 / top_k 20 (run `1110b`):

| | Step 0 | Step 25 | Step 50 | Step 75 |
|---|---|---|---|---|
| avg_score | 0.232 | 0.210 | 0.217 | 0.225 |
| pass@3 | 0.404 | 0.337 | 0.326 | 0.360 |
| Out of context (of 267) | 60 | 75 | 87 | 92 |
| Replies cut at the 16,384 cap, % of calls | 3.9% | 6.2% | 7.5% | 6.3% |
| Trials with a capped reply | 94 | 118 | 117 | 121 |
| Mean of trials without a capped reply | 0.330 | 0.351 | 0.377 | 0.375 |

- A second step-0 eval of the same base model (the launch before `1110b`) gave 0.210 / pass@3 0.303, so one eval's
  noise is about +-0.025 on the average and up to 0.10 on pass@3.
- The friend's curve (other branch, eval settings unknown): TB2.1 ~0.25 -> ~0.38 over 300 steps, i.e. ~0.03 per
  75 steps, the size of our noise.

Same step-75 checkpoint, eval at training sampling (run `r75`, eval before training):

| | Step 75 at 0.6 | Step 75 at 1.0 |
|---|---|---|
| avg_score | 0.225 | **0.296** |
| pass@3 | 0.360 | **0.427** |
| Out of context | 92 | 11 |
| Agent timeouts at 3,600 s | 18 | 33 |
| Replies cut at the cap | 1,184 of 18,691 calls (6.33%) | 7 of 26,697 calls (0.03%) |

- **Base model at 1.0 / 1.0** (step 0 of the fresh run `t1-1112`, 11-12):
  **0.2135, pass@3 0.337**; 2 out of context, 19 agent timeouts, 0.05% of calls capped. So at training sampling the
  eval rose from 0.213 (base) to 0.296 (step 75 of the lost run): **+0.083 in 75 steps**, against ~+0.03 on the
  friend's curve over the same span and a noise of ~0.025 per eval. Caveat: the two numbers come from different
  runs (same code and settings; the step-75 weights are gone, so this cannot be re-measured).

Training reward (trials grouped by the hour they started, each hour ~120 different tasks x 16):
0.30-0.34 for the first five hours, 0.35-0.38 through the morning, 0.41-0.43 by steps 65-75. Tasks do not repeat
within the first epoch, so this is not memorization. Out-of-context endings in training: 14-27% per hour, no
clear trend.

## 2b. TB2.1 evals of Qwen3.5-9B (89 tasks x 3 = 267 trials each)

The in-run evals of the first full run of 11-07 (steps 20-100, at 0.6 without summarization: 0.17-0.25 with 43-46% of
trials out of context) are left out at the user's request; they are in `ops/calibforge_rl_1106/PLAN.md`.

"Out of context" = trial ended on the context limit (scored 0). "At cap" = replies cut at the per-turn limit (with no
cap: a reply that ran until the context was full). n/m = not measured (traces lost or never looked at).

| # | Date, run | Weights | Eval sampling | Summarize | Per-turn cap | Agent, sandbox, timeout | avg | pass@3 | Out of context | At cap: % of calls / trials | Agent timeouts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 11-05 `skycap-qwen-tb-dune` | base | 0.6 / 0.95 / 20 | on | none | cluster, 4 vCPU / 8 GB, max(7,200 s, own); thinking dropped (template) | 0.253 (0.270 with timeouts re-run once) | 0.410 (0.443) | n/m (a full context is summarized) | n/m | 60 |
| 2 | 11-05 `base-qwen-tb-dune` (stock Harbor, no SkyCap) | base | 0.6 / 0.95 / 20 | on | none | as 1 | 0.238 (0.252) | 0.333 (0.375) | n/m | n/m | 57 |
| 3 | 11-06 base eval before training (`-fa-1106-1723` step 0) | base | 0.6 / 0.95 / 20 | on | none | cluster, 4 / 8, 3,600 s; thinking kept | 0.277 per trial (W&B per row: 0.240) | 0.438 (0.337) | n/m (80 trials summarized) | n/m | 67 |
| 4 | 11-10 `-insb-3600-1110` step 0 | base | 0.6 / 0.95 / 20 | off | none | in sandbox, each task's own size (mostly 1 vCPU), 3,600 s, 262k | 0.195 | 0.315 | 106 (40%) | ~1% of calls / ~99 trials (one runaway reply each) | 20 |
| 5 | 11-10 `-cap16k-nosum-1110` step 0 (265 trials) | base | 0.6 / 0.95 / 20 | off | 16,384 | in sandbox, 4 / 8 / 20 GB, 3,600 s, 262k | 0.210 | 0.303 | 62 (23%) | 5.8% / 87+ | 20 |
| 6 | 11-10 `...-1110b` step 0 | base | 0.6 / 0.95 / 20 | off | 16,384 | as 5 | 0.232 | 0.404 | 60 (22%) | 3.9% / 94 | 21 |
| 7 | same, step 25 | step 25 | same | off | 16,384 | as 5 | 0.210 | 0.337 | 75 (28%) | 6.2% / 118 | 18 |
| 8 | same, step 50 | step 50 | same | off | 16,384 | as 5 | 0.217 | 0.326 | 87 (33%) | 7.5% / 117 | 14 |
| 9 | same, step 75 | step 75 | same | off | 16,384 | as 5 | 0.225 | 0.360 | 92 (34%) | 6.3% / 121 | 18 |
| 10 | 11-11 `...-1111-r75` step 75 | step 75 (same weights as 9) | **1.0 / 1.0, top_k off** | off | 16,384 | as 5 | **0.296** | 0.427 | 11 (4%) | 0.03% / 5 | 33 |
| 11 | 11-12 `...-t1-1112` step 0 | base | **1.0 / 1.0, top_k off** | off | 16,384 | as 5 | **0.2135** | 0.337 | 2 (1%) | 0.05% / 9 | 19 |
| 12 | same, step 25 (first in-run eval at 1.0) | step 25 | **1.0 / 1.0, top_k off** | off | 16,384 | as 5 | **0.2434** | 0.348 | 4 (1.5%) | 0.07% / 11 | 27 |
| 13 | same, step 50 (in-run) | step 50 | **1.0 / 1.0, top_k off** | off | 16,384 | as 5 | **0.2697** | 0.4045 | 11 (4.1%) | 0.08% / 9 | 23 |
| 14 | same, step 75 (in-run; one sample forced to 0, see PLAN 11-13 01:16) | step 75 | **1.0 / 1.0, top_k off** | off | 16,384 | as 5 | **0.2959** | 0.4157 | 12 (4.5%) | 0.04% / 9 | 35 |
| 15 | same, step 100 (in-run) | step 100 | **1.0 / 1.0, top_k off** | off | 16,384 | as 5 | **0.3258** | 0.4382 | 15 (5.6%) | 0.03% / 10 | 37 |
| 16 | same, step 125 (in-run; epoch 2, tasks repeat from step 113) | step 125 | **1.0 / 1.0, top_k off** | off | 16,384 | as 5 | **0.3146** | 0.4494 | 12 (4.5%) | 0.01% / 4 | 17 |
| 17 | 11-13 `...-t1-b32-lr2e-6-1113` step 0 (second base eval at 1.0, same setup as row 11: noise check) | base | **1.0 / 1.0, top_k off** | off | 16,384 | as 5 | **0.2247** | 0.3371 | 2 (0.7%) | 0.07% / 10 | 19 |
| 18 | same run, step 25 (in-run; batch 32, LR 2e-6: 25 updates of 512 trajectories each) | step 25 | **1.0 / 1.0, top_k off** | off | 16,384 | as 5 | **0.2772** | 0.382 | 4 (1.5%) | 0.07% / 13 | 34 |
| 19 | 11-15 `...-daytona-b32-eval0-1115` step 0: **base on the second Daytona org** (own images, 4 vCPU / 8 GiB / 20 GiB, preview relay, 4 engines x 90 concurrent = the in-run evals' load per engine) | base | **1.0 / 1.0, top_k off** | off | 16,384 | in-sandbox Terminus, **Daytona (eval org)**, 3,600 s | **0.2547** | 0.3596 | 4 (1.5%) | 0.08% / 14 | 24 |
| 20 | 11-15 `...-daytona-b32-eval25-1115b`: checkpoint 25 of the Daytona training run (first, uncorrected run), same setup as 19 | step 25 (Daytona run) | **1.0 / 1.0, top_k off** | off | 16,384 | as 19 | **0.2659** | 0.3933 | 6 (2.2%) | 0.03% / 7 | 15 |
| 21 | 11-15 `...-daytona-b32-1115-eval0b` step 0: **base on the second Daytona org again** (run next to 22, to measure run-to-run noise), same setup as 19 | base | **1.0 / 1.0, top_k off** | off | 16,384 | as 19 | **0.2172** | 0.3146 | 4 (1.5%) | 0.09% / 17 | 18 |
| 22 | 11-15 `...-daytona-b32-1115-eval25`: checkpoint 25 of the second Daytona training run `...-daytona-b32-lr2e-6-1115` (with the relay / stall fixes), same setup as 19 | step 25 (Daytona run 2) | **1.0 / 1.0, top_k off** | off | 16,384 | as 19 | **0.2809** | 0.4157 | 13 (4.9%) | 0.09% / 17 | 14 |

Reading it:
- **At training sampling there is almost no looping**: rows 10 and 11 have 7 and 11 capped replies in 26,697 and
  20,455 calls and 11 and 2 out-of-context endings, against 60-125 out-of-context endings in every eval at 0.6
  without summarization (rows 4-9).
- **The base model without looping is 0.2135 (row 11), not the friend's ~0.25.** At 0.6 the base model's clean
  trials score higher (0.33 over 186 trials in row 6) than at 1.0 (0.24 over 241 trials in row 11), so removing
  the loops by raising the temperature gives back most of what it gains: 0.2135 at 1.0 against 0.21-0.23 at 0.6.
- **Our only base-model evals at or above 0.25 had summarization on** (rows 1-3: 0.253 / 0.270 / 0.277), with the
  agent on the cluster and 0.6 sampling.
- One eval's noise is about +-0.025 on avg (rows 5 and 6 are the same model and settings: 0.210 and 0.232), so
  0.2135 against 0.25 is about 1.5 standard errors: suggestive, not established.
- W&B's `eval/all/avg_score` counts rows, not trials, when a trajectory has several paths (row 3: 0.240 per row
  against 0.277 per trial). With summarization off the two agree.
- **Base model at 1.0 / 1.0, four evals: 0.2135 and 0.2247 on Dune (rows 11, 17), 0.2547 and 0.2172 on the
  Daytona eval org (rows 19, 21).** The two Daytona runs differ by 0.0375 (paired over tasks: se 0.026; 25 of 89
  tasks changed), more than the gap between the providers' means (0.236 vs 0.219). So the "Daytona base is higher
  than Dune" reading of row 19 alone is not supported; one eval is good to about +-0.03.
- **Checkpoint 25 of the second Daytona run (row 22): 0.2809**, +0.045 over the mean of the two base evals on
  the same org (+0.026 over row 19, +0.064 over row 21; paired se ~0.027 each) and +0.015 over the first Daytona
  run's checkpoint 25 (row 20). Same size as the Dune b32 run's gain at step 25 (0.2247 -> 0.2772, +0.053). It
  also runs out of context more often (13 trials vs 4) with longer replies (median 538 vs 350 tokens).

## 3. Looping at low temperature (the main finding)

- **What it is.** The model starts a reply and repeats a block of text until something stops it. With no per-turn
  cap, one such reply filled the whole 262k context (~24 min of an engine slot; 40% of eval trials in the first
  launch). With the 16,384 cap the reply is cut, Terminus tells the model its output was too long and asks again, and
  the trial goes on, usually into another loop a turn or two later.
- **It is a property of the sampling, not of the harness.** Run `1110b`: eval at 0.6 / top_k 20: 5.67% of 68,100
  calls cut at the full cap, 41% of trials with at least one. Training at 1.0: 0.02% of 157,000 calls (1-in-10
  sample), 1.1% of trials, flat over 20 hours. Base model, 126,000 calls at 1.0 with no cap: zero runaway replies.
- **It got worse with training at 0.6**: trials with a capped reply 35% -> 45%, out-of-context 22% -> 34% over 75
  steps, while trials without a capped reply improved 0.33 -> 0.38. Training samples at 1.0, so nothing penalizes
  what the policy does at 0.6. Policy entropy drifted from 0.42-0.55 to 0.33-0.44.
- **What the loops look like** (2,152 capped eval replies read, 2,019 with a clear repeating block): in the
  thinking in 98.6%; the repeated block starts a median 1% into the reply; its first sentence begins "Let me ..."
  20%, "Actually, ..." 21%, "Wait, ..." 3%, otherwise task-specific. A trial with a capped reply averages reward
  0.02-0.05. The few capped TRAINING replies are different: long code or data dumps, not reconsidering loops.
- **Examples**: `loop_examples_step75/` (three step-75 eval trials rendered as text; `.capped-N.txt` = full reply).
- a reference recipe's ApexAgents-SkyRL-Recipe evaluates at the training sampling (one shared `llm_kwargs`, temperature 1.0),
  caps a turn at 40-50k tokens and ends the trajectory with reward 0 when a capped reply has no tool call.

## 4. Checked and ruled out

- **Stale KV cache in eval**: every eval trial carries the cache salt of the weights it evaluates (`:1`, `:26`,
  `:51`, `:76`); Harbor's client puts `cache_salt` in the request body (`test_cache_salt_wire.py`); skycap forwards
  it; the trainer is paused during an eval, so one weight version serves it.
- **SkyCap after a capped reply**: the next call extends the truncated turn's sampled tokens (bridged=True, one
  path) in every case checked (1,165 of 1,165 followed capped turns by step 25).
- **Dune / sandbox limits as the cause of timeouts**: the first full run's timeouts were Harbor's exec round trips
  and a shared event loop (`ops/timeout_repro_1110/`), not the provider.
- **1 vCPU eval sandboxes** (first launch): timeouts were not concentrated there; eval now runs at 4 vCPU anyway.

## 5. Infra findings (fixed in place unless noted; proper fixes are the end-of-run list in `PLAN.md`)

- **Driver event loop froze 30-376 s**: Harbor tars and gzips agent logs synchronously on the loop, and re-uploads
  the in-sandbox agent's own logs (up to 1.4 GB of terminal recording). Fix `loop_offload.py`
  (`SKYCAP_OFFLOAD_TAR=1`, `SKYCAP_SKIP_LOG_REUPLOAD=1`): worst pause since ~4 s; step time 26 min -> 8-15 min;
  training errors 4.5% -> under 1%.
- **A trial ran 90 min past its 3,600 s limit** when the timeout landed inside a freeze: the in-sandbox agent now
  enforces the deadline itself (not yet exercised).
- **Verifier-retry check keyed on ctrf.json**: 25 training tasks never write it, so their failed attempts were
  retried and their groups discarded; patched in the data copy (`fix_no_ctrf.py`).
- **TB2.1 `qemu-startup` / `qemu-alpine-ssh` could never pass** (Debian bullseye packages moved to the archive);
  one-line fix in our eval copy (`fix_qemu_apt.sh`), in effect from step 0 of `1110b`.
- **Launcher bug**: `EVAL_SAMPLING="${EVAL_SAMPLING:-{...}}"` appended a stray `}` to any value set from outside;
  fixed in both launchers.
- **Throughput** is set by the staleness window (64 groups running or buffered) and by each group waiting for its
  slowest of 16 trials (group p50 859 s, p90 3,751 s; one trial p50 396 s): ~4-5 steps/h, engines mostly idle.
  Each eval pauses training for 65-100 min, sometimes on one retried trial. Not changed (user's settings).

## 6. Open, and what to watch next

1. ~~Does looping come back in an in-run eval at 1.0?~~ **No** (11-12, run `t1-1112`): the in-run step-25 eval had
   13 of 19,920 calls at the cap (0.07%) and 4 out-of-context endings, the same as the clean-state step-0 eval
   (11 of 20,705; 2). So in-run state is not a cause of the looping; the 0.6 / top_k 20 sampling was.
   Score at 1.0: 0.2135 (step 0) -> 0.2434 (25) -> 0.2697 (50) -> 0.2959 (75) -> 0.3258 (100): +0.112 in 100 steps
   (the friend's curve: ~+0.13 over 300), a steady +0.026 to +0.030 per 25 steps; at step 75 it was already three times one eval's noise, and the same as the lost run's step 75 (0.296), an independent run from a fresh start.
2. ~~Confirm the engines served step-75 weights in the resumed eval.~~ Checked 11-11 21:55: rollout-vs-trainer
   |logprob diff| is 0.0080 and 0.0082 on the first two steps after the resume, the same as every step of `1110b`
   (0.0077-0.0102), with no drop after the first in-run sync. Consistent with step-75 weights; not a proof, since
   the value base weights would give was never measured.
3. ~~Base model at 1.0 / 1.0.~~ Measured 11-12: 0.213 (section 2).
4. How a capped reply appears in the training rows, and whether to train on it (user asked to be reminded to check
   max_tokens truncation on SkyCap when reviewing results).
5. Staleness limit; checkpoint upload to S3 (checkpoints are on the trainer node's local disk only).

## 7. TODO after this run: recreate the friend's ~0.25 at step 0
- **After this run: concurrency** (user, 11-14): the runs keep only ~580-640 trials in flight under a 900-1,600
  cap, because the staleness window releases ~800 trials per ~930 s step; see PLAN.md fix item 13. Also to
  explain: trials on Daytona are ~16% shorter than on Dune for the same tasks.

Removing the looping (eval at training sampling) did not reproduce the friend's step-0 score: ours is 0.2135
(row 11 of the table in 2b), theirs ~0.25, and theirs then climbs to ~0.38. Find out why before reading our curve
against theirs.

1. Get the friend's exact eval setup: sampling, per-turn cap (they said 16k), summarization (they said none),
   thinking retention (kept), agent timeout, max turns, sandbox size and provider, where the agent runs, Terminus
   version and patches, TB2.1 commit, samples per task, and how their `avg_score` is computed (per row or per trial).
2. Compare with our three base-model evals that did reach 0.25-0.28 (rows 1-3): all had summarization on, the agent
   on the cluster and 0.6 sampling; rows 1-2 also dropped earlier thinking and allowed 7,200 s.
3. One-factor checks on the base model, each a 75-140 min eval: (a) 0.6 + summarization on, in sandbox, 3,600 s;
   (b) 1.0 + summarization on; (c) the cluster-side agent at 1.0 with the cap; (d) 7,200 s; (e) a repeat of row 11
   for the noise. Read the per-task differences, not only the mean.
4. Sampling in between: 0.6 without top_k, or 0.8, with a repetition or presence penalty, to keep the higher
   per-trial quality of 0.6 without the loops.
5. Until this is settled, compare our curve with the friend's by its gain over our own step 0, not by level.
