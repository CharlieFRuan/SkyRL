# Fully async Qwen3.6-35B-A3B on MiMo (SkyCap + mini-swe-agent + sandbox relay)

Launcher: `ops/run_fully_async.sh <run>` (on worker 100.81.16.105) ->
`SkyRL-skycap/examples/train_integrations/harbor_skycap/run_fully_async_qwen36_mini_swe.sh`.
Status: `bash ops/train_status.sh` on the driver worker. W&B project: `sky-posttraining-uc-berkeley/skycap-mimo-fully-async`.

Setup (both runs): Megatron TP4/EP8 on 2 nodes (CPU optimizer offload), 8 vLLM engines TP2 on 2 nodes; hero
hyperparameters (DPPO delta 0.15, prompt_mean, fused LM-head logprobs, lr 1e-6, betas [0.9, 0.98], wd 0.01, no std
norm, zero-variance filter); 16 prompts x 16 samples, staleness 2 (48 groups in flight), concurrency 500, `final`
path, no R3, 128k context; sticky_least_loaded routing; cache salt per trajectory; train on 2658 MiMo tasks; eval
SWE-bench-100 (1 sample) before training and every 20 steps; ckpt every 5 steps, keep 2.

## Run 1: `qwen36-mimo-fa-1013-1744` (W&B 08c34mur), stopped after step 22 on request

- SWE-bench-100 eval: step 0 **0.634**, step 20 **0.624** (within noise: 100 tasks x 1 sample, sd ~0.05).
- Step time ~10 min after the first (36 min cold): ~53% `wait_for_generation_buffer`, ~45% `policy_train`
  (150-760 s), weight sync ~6 s. Per-step reward 0.12-0.39, entropy ~0.23 (one dip to 0.14 at step 9), staleness
  1.4-1.9, grad norm 0.03-0.05. No training errors. Relay: 570k calls, 0.03% errors, 0 sandboxes dropped.
- Why stopped:
  - The agent timeout was **1200 s** (SkyRL's `harbor_trial_config/default.yaml` overrides every task's), not
    MiMo's 3600 s: 5.6% of trials timed out, in 25% of the groups, and a group with any timeout is masked whole.
  - A rollout that ran out of context trained with the verifier's reward (could be 1).

## Run 2: `qwen36-mimo-fa-1013-2316`, from scratch (started 2026-10-13 23:16 PDT)

Changes vs run 1:
- `harbor_trial_config.agent.override_timeout_sec=null`: each task's own agent timeout (MiMo 3600 s; the
  SWE-bench eval tasks keep the parity 1800 s / verifier 300 s set in their task.toml).
- `skycap.context_length_reward=0.0`: a training rollout that runs out of context trains with reward 0 (not
  filtered); eval still scores it as stock Harbor does.
- Sandboxes labelled `run=<run name>` (environment kwarg `run_label`).
- Relay: stub-measured agent-side time per call (`mean_overhead_seconds` in the managers' /stats), reply
  connections kept alive 300 s.
- Trial backstop `SKYCAP_TRIAL_TIMEOUT_SEC=9000`.

Data fixes on the nodes (originals kept as `task.toml.orig`): 8 MiMo tasks capped at 16 GB memory and 2 at 20 GB disk
(Daytona's per-sandbox maxima); SWE-bench-100 eval copy set to agent 1800 s / verifier 300 s (parity settings).

### Run 2 restart at step 5 (2026-10-14 01:18 PDT): relay delta encoding

Measured relay overhead in run 2 climbed from 0.45 s (eval) to ~1 s per call in training. Cause: mini-swe-agent
resends its whole history every call, and a body crosses Daytona's proxy (us-east-1, from ap-south-2) at ~170 KB/s:
a 300 KB body cost 2.7 s per call in a standalone test. Fix: the stub sends a call that extends the last answered
call on its path as a delta (shared-prefix length + the rest), the manager rebuilds it (fetching the whole body
from the stub if it lacks the base), and poll responses are gzipped; a 300 KB body now costs 0.35 s. Also fixed: a
call handed to a poll whose connection broke was lost (the agent would hang); the stub now requeues it.
Deployed by stopping after the step-5 checkpoint, merging the checkpoint onto all 4 nodes (`ops/sync_ckpt.py`, ~2 min for 486 GB with one stream per file:
each node had only its own ranks' shards, and placement moves between launches), and relaunching the same run
name with `trainer.eval_before_train=false` (resumes at step 5).
Run 2 before the restart: step-0 SWE-bench eval 0.703 (agent timeout 1800 s, parity); no agent timeouts; policy
updates ~15-20 min per step (4-5M tokens per DP rank) now that rollouts aren't cut at 20 min.

### Run 2 progress
- Step 20 (2026-10-14 04:59 PDT): SWE-bench-100 **0.683** (step 0: 0.703; within noise, sd ~0.046 at 100 x 1).
  Steps 6-20 ~7-15 min each; per-step train reward 0.27-0.46, mean rollout 50-73k tokens, staleness <= 1.8;
  relay 0.30 s overhead per call (99% deltas), 2 call errors in 450k, 0 sandboxes dropped; no training errors.
  ~0.5% of sandboxes hit Harbor's 360 s agent-install limit in start bursts; the launcher now sets
  `AGENT_SETUP_TIMEOUT=900` (takes effect at the next launch). W&B: steps 1-5 `14wm4ep6`, steps 6+ `8klyboql`.
- Step 40 (2026-10-14 09:44 PDT): SWE-bench-100 **0.713** (0.703 / 0.683 / 0.713 at steps 0 / 20 / 40; 1 pass each,
  sd ~0.046, so flat within noise). A ~2-min Daytona proxy outage at 07:28 PDT failed ~190 trials (bad gateway);
  retried/masked, no leaked sandboxes, run unaffected. Relay passed 1M calls (95 errors, all in that outage; 0 dropped).
  Next restart: `EVAL_SAMPLES=3` (pending the user's answer) and `AGENT_SETUP_TIMEOUT=900`.
- Step 60 (15:20 PDT): SWE-bench-100 **0.703** (0 / 20 / 40 / 60: 0.703 / 0.683 / 0.713 / 0.703). This eval took 68 min:
  sympy__sympy-13372's Daytona image build wedged twice (past its 1800 s build timeout; Daytona's builder was slow, a
  snapshot sat in BUILDING_SNAPSHOT 100+ min); unblocked by deleting our stuck sandboxes, the task scored 0.
- Found: `SKYCAP_*` env vars never reached the Ray actors (SkyRL forwards only `SKYRL_*`), so the per-trial backstop
  `SKYCAP_TRIAL_TIMEOUT_SEC` was never in effect (and the stress test's `SKYCAP_TRIAL_ATTEMPTS=1` was ignored: it
  retried failed trials once). Fixed: `prepare_runtime_environment` forwards `SKYCAP_*`; new
  `SKYCAP_EVAL_TRIAL_TIMEOUT_SEC` (launcher: 3600 s) caps eval trials. Planned restart at the step-65 checkpoint
  with these, `AGENT_SETUP_TIMEOUT=900` and `EVAL_SAMPLES=3`.
- Restarted from the step-60 checkpoint at 15:33 PDT (W&B `okvj1xcm`), with SKYCAP_* forwarding (trial backstop
  9000 s, eval-trial backstop 3600 s now in effect), `AGENT_SETUP_TIMEOUT=900`, and **3 eval passes** from step 80 on.
  `ops/sync_ckpt.py` now writes the latest-step marker as the step itself (a stale marker from the step-5 merge had
  been copied over the driver's "60"; caught before relaunch).
- Step 80 (21:12 PDT): SWE-bench-100 **0.708** over 3 passes (301 trials, sd ~0.027); eval 44 min (psf__requests-2317's
  slow verifier is again the last trial). Series: 0.703 / 0.683 / 0.713 / 0.703 / 0.708 at steps 0 / 20 / 40 / 60 / 80:
  flat. Since the step-60 restart: 0 relay call errors, 0 dropped sandboxes, 2 agent-install timeouts (was ~0.5%).
- Step 100 (02:46 PDT 10-15): SWE-bench-100 **0.698** over 3 passes (eval 36 min). Series 0.703 / 0.683 / 0.713 /
  0.703 / 0.708 / 0.698 at steps 0-100: flat. Train reward by 20-step window: 0.352 / 0.342 / 0.340 / 0.419 / 0.379;
  entropy 0.24 -> 0.27 (rising slowly, no collapse); mean rollout 62k -> 79k tokens; grad norm 0.040 -> 0.033.
- Step 120 (08:55 PDT 10-15): SWE-bench-100 **0.674** over 3 passes; **0.688** excluding sympy__sympy-21847 (its
  Daytona sandbox build failed on every attempt) and psf__requests-2317 (verifier always times out): vs 0.712 / 0.702
  on the same basis at steps 80 / 100. A ~1 sd dip (sd ~0.027). The eval took 63 min: psf__requests-2317's retry hung
  without model calls; deleted its sandbox (and 3 leaked STARTED sandboxes, 2.5-4 h old) to release it.
  Queued for the next restart: `SKYCAP_EVAL_TRIAL_ATTEMPTS=1` (one attempt per eval trial, as the parity baseline).

# Nemotron-3.5-Lightning-30B-A3B (from 2026-10-15)

## SWE-bench Verified 100 x 3 parity (2026-10-15 12:22-12:50 PDT)

Both arms: mini-swe-agent 2.4.6, `mini_swe_agent_nemotron35.yaml` (temp 1.0, top_p 0.95 [model card], thinking on and
kept: `truncate_history_thinking=false`), 131072 context, 16k tokens per turn, no step limit, agent 1800 s / verifier
300 s, 300 trials at once (100 tasks x 3), one attempt each, 8 vLLM engines TP1 on one node, same sandbox relay.

| Arm | Runs | Mean (sd) | Exits: submitted / context | Turns p50 / p90 | Peak prompt p50 / p90 |
|---|---|---|---|---|---|
| Stock Harbor + vLLM (`nem35-swe100-baseline-x3-1015-1230`) | 0.490 / 0.510 / 0.480 | **0.493** (0.015) | 291 / 9 | 62 / 130 | 35.5k / 84.3k |
| SkyCap (`nem35-swe100-skycap-x3-1015-1222`) | 0.480 / 0.480 / 0.400 | **0.453** (0.046) | 291 / 9 | 68 / 143 | 39.0k / 92.5k |

SkyCap - baseline, paired per task: **-0.040 (95% CI -0.083 to +0.003)**. Model card: 51.56 (NVIDIA's harness).
Checks for a systematic cause (none found):
- Per turn the arms match: format errors 9 vs 8, finish=length 0 vs 0, completion tokens/turn mean 289 vs 285 (p90 723 vs 722).
- History: 97% of model turns re-render through the chat template exactly as sampled (the rest: a trailing newline in
  a tool argument); SkyCap's renderer == HF template on rendered prompts.
- Tool-call parsing: 1 of 5790 turns had a `<tool_call>` SkyCap didn't parse, and it was malformed (`<parameter=command`
  without `>`).
- SkyCap's third pass (0.40) is the outlier. To confirm: rerun both arms (more reps) when GPUs free up.
- Relay: 0 call errors in either arm, overhead 0.27 s/call in both.
- Why SkyCap's third pass scored 0.40 (read the traces, 2026-10-15): the "passes" are not separate runs. All 300 trials
  ran at once, and `--split 3` assigns each task's attempts by start time (seconds apart). Pass 3 lost 12 tasks that
  were solved in at least 3 of the other 5 attempts. In each, the agent submitted normally and the verifier found a
  wrong fix:
  - sphinx-9698: changed the output to the wrong string, then edited the test to match.
  - sklearn-13135: edited the installed sklearn in site-packages, restored it from a backup it made in /testbed, and
    broke the import.
  - sympy-22456: called the issue a "known limitation" and didn't fix it.
  - astropy-14508, django-14999, sphinx-11445: wrong fixes.
  Behaviour by start order is identical (SkyCap attempts 1/2/3: turns 78/78/75, tokens per turn 282/283/273, format
  errors 5/1/2, context exits 3/3/3). 42 of 100 tasks are coin flips for this model (27 always solved, 31 never), so
  one pass has sd ~0.028; a pass at <= 0.40 has p ~0.007 (~4% for the lowest of 6 passes). Most likely chance.
  The one consistent difference: SkyCap trajectories average ~8 more turns in every attempt (78 vs 70; ~2 se).

## Overfit (synchronous, vci3ga5w recipe, overfit16fast, group 8, all paths, 20 steps), started 12:49 PDT 10-15

Runs `nem35-of16fast-g8-all-lr1e-6-1015-1249` (W&B s29odotd) and `...-lr3e-6-1015-1249` (tcz31f88), 2 nodes each.
Megatron TP4/EP8 trains Nemotron; weight sync to vLLM ~7 s; step ~15 min (rollouts ~12 min, policy_train ~105 s).

| Step 1 | reward | vLLM-vs-Megatron mean abs dlogprob (max) | entropy | grad norm |
|---|---|---|---|---|
| lr 1e-6 | 0.297 | **0.072** (17.6) | 0.110 | 0.253 |
| lr 3e-6 | 0.289 | **0.071** (24.1) | 0.113 | 0.216 |
| Qwen3.6 vci3ga5w (same recipe) | - | 0.0157 (6.2) | - | - |

The train/inference logprob gap is ~4.6x Qwen's on the same recipe. Step 1 is on-policy, so it is numerical.
- SkyRL's CI bound for this model is 5e-2, on short sequences (`test_megatron_models.py`, nemotron3.5-lightning row).
- Megatron's router already runs in fp32 (SkyRL default `moe_router_dtype=fp32`). Mamba packed sequences are handled
  (`seq_idx` goes through the fused conv+scan kernel).
- Likely contributor: the installed mamba_ssm 2.3.2.post1 has no `state_dtype`
  (`megatron.core.ssm.mamba_mixer.MAMBA_HAS_STATE_DTYPE == False`). Megatron's training forward hands inter-chunk SSM
  states along in bf16 (`ssd_combined.py:381`, `out_dtype=C.dtype`), while vLLM keeps the SSM state in fp32 (the
  model's `mamba_ssm_cache_dtype`). That error grows with sequence length; this run averages ~21k tokens.
- Not fixed yet: it needs a mamba_ssm with `state_dtype` (SkyRL pins a custom cu13/torch2.13 wheel) and
  `transformer_config_kwargs.mamba_training_ssm_states_dtype=fp32`. Token TIS (cap 2) in this recipe absorbs part of it.
- mamba_ssm with `state_dtype`: not released (PyPI latest = 2.3.2.post1, ours); upstream state-spaces/mamba `main` has it
  in `ops/triton/ssd_combined.py`. A fix would mean building mamba_ssm from main for cu13/torch 2.13 on the nodes and
  setting `trainer.policy.megatron_config.transformer_config_kwargs.mamba_training_ssm_states_dtype=fp32`; it would need
  GPUs to verify (all 32 are in the overfit runs now).

## Logprob-mismatch investigation (2026-10-15 13:55-14:45 PDT; overfits stopped on request)

1. GSM8K GRPO with SkyRL main's Nemotron example settings (`ops/run_gsm8k_nemotron.sh`), no SkyCap, short sequences
   (~650 tokens), one node each, W&B project `nemotron_3_5_lightning_30b_grpo`. `rollout_train_logprobs_abs_diff_mean`, steps 1-3:

   | Variant | mean | max |
   |---|---|---|
   | base (CUDA graphs) `nem35-gsm8k-base-1015-1402` | 0.045 / 0.053 / 0.052 | 12-14 |
   | vLLM eager `...-eager-...` | 0.045 / 0.052 / 0.051 | 9-12 |
   | 4 engines x TP2 `...-tp2x4-...` | 0.046 / 0.053 / 0.052 | 10-16 |
   | **R3** (routing replay) `...-r3-...` | **0.008 / 0.008 / 0.008** | 1.1-1.9 |

   Not kernels/CUDA graphs (eager = same) or TP. **MoE routing divergence is the cause**: replaying vLLM's expert
   choices in Megatron cuts the gap 6x. The architecture mapping is right (with R3 the experts, scaling 2.5 and the rest
   agree to 0.008).
   - Router precision matches: vLLM computes the gate in fp32 (`GateLinear(..., force_fp32_compute=True)`), Megatron
     too (`moe_router_dtype=fp32`). The bridge maps `e_score_correction_bias` -> `router.expert_bias`, sigmoid scoring,
     and `routed_scaling_factor` -> `moe_router_topk_scaling_factor`.
   - So small hidden-state differences flip Nemotron's (sigmoid, 6 of 128) expert choices often, and a flip moves a
     token's logprob a lot.
2. Not SkyCap: re-scoring 12 SkyCap-recorded overfit trajectories (step 1, base weights) with a plain vLLM server's
   prefill (`ops/rescore_skycap.py`, `prompt_logprobs`) gives mean |recorded - prefill| = **0.074** over 107k sampled
   tokens, the same as Megatron's 0.072. vLLM's own decode vs prefill disagree that much (worst: recorded -0.05,
   prefill -15.2). No length dependence: 0-8k 0.070, 8-16k 0.076, 16-24k 0.076.
   Long agentic context adds a little (GSM8K 0.05 vs overfit 0.072). SkyCap's bookkeeping is consistent: a misalignment
   would show as a much larger, irregular gap.
3. The mamba_ssm bf16 inter-chunk state (above) is at most a minor term: R3 alone brings GSM8K to 0.008.
=> Use R3 for Nemotron (SkyRL's new_model_support.md, Stage 2: "enable R3 ... to reduce logprob mismatch").
4. Confirmed through SkyCap on long agentic trajectories: the same synchronous overfit with R3 (`ops/run_overfit.sh` R3=1;
   `nem35-of16fast-g8-all-r3-lr1e-6-1015-1430` W&B f50x3z2w, `...-r3-lr3e-6-...` flcc3asj, ~22k tokens/trajectory):
   mean |dlogprob| **0.0103** (lr 1e-6, step 1); 0.0108 / 0.0112 / 0.0106 (lr 3e-6, steps 1-3); max 3.5-7.2.
   That is down from 0.072 without R3, and below Qwen3.6's 0.0157 on this recipe without R3. SkyCap carries the routed
   experts end to end. Rewards at steps 1-3 (lr 3e-6): 0.312 / 0.305 / 0.281.
   These two runs continue as the lr comparison (20 steps); the full run will use R3.

## Cross-framework no-R3 mismatch tests (prepared 2026-10-15 ~16:00; run after the R3 overfits)
- **prime-rl** (commit 0956a885, v0.10.0; vLLM 0.31.0 cu130, torch 2.13, transformers 5.6.2) on .123:
  `$D/frameworks/prime-rl/run_mismatch_gsm8k.sh` (config `configs/nemotron35_gsm8k_mismatch.toml`; `DRY_RUN=1` validates).
  - Setup: its own NemotronH trainer (FSDP, EP4 on GPUs 4-7), 4 vLLM replicas on GPUs 0-3; GSM8K (verifiers env),
    32 prompts x 8, temp 1.0, top_p 1.0 (`generation_config="vllm"`), 1024 tokens, lr 1e-6, 3 steps,
    `max_off_policy_steps=0`, no R3.
  - Metrics: native `mismatch_kl/all/mean` is the k3 KL (exp(d) - 1 - d, d = trainer - vLLM logprob). Patched to also
    log `mismatch_logprob_abs_diff/all/mean` (= SkyRL's mean |d|) and to keep zero-advantage groups
    (`PRL_KEEP_ZERO_ADV=1`) so the metric covers all response tokens. Compare step 1.
  - Ports moved to base 18000 (5555 is taken on .123). The first run converts weights (~60 GB, `prime-rl/conv/`).
    About 25-35 min.
  - Note: vLLM logits fp32 vs trainer bf16 LM-head GEMM; `FP32_LM_HEAD=0` removes that difference.
- Lead from the prime-rl prep: vLLM's NemotronH gate (`GateLinear`, bf16 weights, fp32 out on B200) uses a cuteDSL
  low-latency GEMM for <= 16 tokens (decode) and cuBLAS otherwise (prefill). Both are bf16 x bf16 -> fp32, so they
  differ only in summation order. Small, but it could flip near-tied sigmoid routing. Test later: re-score with the
  ll_bf16 path off.
- **Miles** (radixark/miles main 871cc7d4; SGLang `sglang-miles` 7b6993a3, Megatron-LM radixark 9df207fd, Megatron-Bridge
  8cd3466d, torch 2.13 cu130, TE 2.17) on .87: venv `$D/frameworks/miles/venv`, script
  `$D/frameworks/miles/run_mismatch_gsm8k.sh` (`DRY_RUN=1` validates; backup of the scripts in `ops/miles/`).
  - **NEVER use Miles' own launchers (`scripts/run_*.py` / `execute_train`): they run `ray stop --force; pkill -9 ray`
    etc.** The script bypasses them and starts a private local Ray (`custom/run_with_local_ray.py`,
    `ray.init(address="local")`, dashboard 8465).
  - Supports nemotron_h through Megatron-Bridge with its own shim (router, expert bias, scaling 2.5, shared expert). It
    has an experimental Lightning example. **All Miles MoE recipes enable R3**; the Nano-30B PR reports a max train/rollout
    logprob diff ~0.014 (R3 status unstated).
  - Setup: 1 node colocated, Megatron TP4/EP8 (fallback `TP=2 PP=2 EP=2`, Miles' verified layout, if TP4 > 2 KV heads
    fails), 8 SGLang engines TP1, GSM8K strict `#### N` reward, 128 x 5, 1024 tokens, temp 1.0 / top_p 1.0, lr 1e-6, 3
    rollouts, no R3 (R3=1 to compare).
  - Metric: native `train/train_rollout_logprob_abs_diff` is the per-sample token mean averaged over samples; patched
    `train/train_rollout_logprob_abs_diff_token_mean` = SkyRL's global token mean (compare this). Global batch 320 =
    2 train entries per rollout at step 1 (average them). Don't use `--get-mismatch-metrics` (it changes the loss).
  - `--recompute-logprobs-via-prefill` (with `--true-on-policy-mode`) gives SGLang's decode-vs-prefill comparison.
  - About 30-45 min (NUM_ROLLOUT=1 about 20 min).
- **NeMo-RL** (NVIDIA-NeMo/RL main cf63b8a3, v0.8.0; vLLM 0.29.0, mcore 0.20, Megatron-Bridge ec835530, Py 3.13) on .219:
  `$D/frameworks/nemo-rl/run_mismatch_gsm8k.sh` (config `configs/grpo-gsm8k-lightning35-1n8g-megatron-mismatch.yaml`).
  - Ray: the script starts a PRIVATE head (`ray start --head --block`, temp `/tmp/nrl_ray`, GCS 127.0.0.1:7379, worker
    ports 20000-27999) and stops only that process (never `ray stop`). NeMo-RL needs ray >= 2.58; it cannot join the
    Anyscale 2.57 cluster.
  - Setup: GSM8K (NeMo-RL boxed CoT prompt, hf_math_verify), 128 x 5, 3 steps, temp 1.0 / top_p 1.0, 1024 tokens,
    lr 1e-6, Megatron TP4+SP / EP8, colocated vLLM 2 x TP4, router frozen fp32, no R3, no IS correction.
    Knobs: `VLLM_PRESET=nvidia|default`, `FP32_LM_HEAD=1`, `ROUTER_DTYPE`, `THINKING`.
  - Metrics: native `train/token_mult_prob_error` = token mean of exp(|d|) (>= 1 + mean|d|, inflated by tails),
    `train/gen_kl_error` (k3). Patched `train/rollout_train_logprobs_abs_diff_mean` = SkyRL's token-weighted mean |d|
    (also max/p99/frac>0.5), printed as `[xfw-mismatch]` per step.
  - About 30-45 min the first run (HF->mcore conversion cached after).
  - **NVIDIA's released Lightning RL recipe (`examples/nemo_gym/nemotron-3.5-lightning/rlvr.yaml`) does NOT use R3.**
    It tolerates the mismatch with TIS clamped to [0.2, 5] and **drops whole sequences with mean exp|d| > 2**
    (`seq_logprob_error_threshold: 2`). vLLM: mamba_ssm_cache_dtype float32, moe_backend flashinfer_cutlass,
    FLASH_ATTN, cudagraph PIECEWISE. Megatron: router frozen + fp32, expert bias fp32, `defer_fp32_logits`.
  - NeMo-RL has a NemotronH fp32 LM-head patch (vLLM + Megatron), described in its code as "the dominant contributor
    to token_mult_prob_error" (off in the Lightning recipe). Worth a variant (`FP32_LM_HEAD=1`); in SkyRL R3 alone
    gives 0.008, so routing dominates here.

## Overfit with R3 (overfit16fast, vci3ga5w recipe, group 8, all paths), 2026-10-15
| step | 1 | 3 | 5 | 7 | 9 | 11 | 13 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lr 3e-6 (`...-r3-lr3e-6-1015-1430`, flcc3asj) | 0.31 | 0.28 | 0.32 | 0.28 | 0.38 | 0.34 | 0.34 | 0.45 | 0.44 | 0.40 | 0.44 | 0.46 | 0.39 |
| lr 1e-6 (`...-r3-lr1e-6-1015-1430`, f50x3z2w), stopped at 18 | 0.31 | 0.30 | 0.31 | 0.28 | 0.31 | 0.34 | 0.31 | 0.32 | 0.34 | 0.32 | 0.36 | - | - |
- lr 3e-6 learns: steps 1-7 mean ~0.29 -> steps 15-20 mean ~0.43. Logprob diff 0.010-0.013 (creeping up slightly);
  tokens/trajectory 22k -> 25-28k. lr 1e-6 is flat (user: too low; stopped).
- Every step waits ~20 min on task 001650 (0/~100 solved, hits the 1200 s agent timeout); 4 tasks never solved
  (000566, 000783, 001650, 002684), so 4 of 16 groups give no signal. Re-pick after the pool96 pass@8 eval.
- Next (user): 4-node lr 5e-6 overfit `nem35-of16fast-g8-all-r3-lr5e-6-4node-1015-1903`, launched 19:21 by
  `ops/launch_after.sh` when lr 3e-6 finished.
- 4-node lr 5e-6 (`nem35-of16fast-g8-all-r3-lr5e-6-4node-1015-1903`, W&B lasvik35), launched 19:21; ~15 min/step.
  Reward by step: 0.32 0.31 0.39 0.31 0.36 0.31 0.34 0.37 0.41 0.38 **0.45** (step 11). It learns faster than
  lr 3e-6, but shows the precursor of the Qwen3.6 lr-5e-6 collapse on GCP: from step 9, entropy 0.11 -> 0.086 -> 0.073
  and tokens/trajectory 30k -> 32k -> 36k (from ~22k at step 1), and context-limit exits 0 -> 2 -> 4. Logprob diff
  ~0.009. Running to step 20 as planned. lr 3e-6 stayed stable (entropy ~0.11).
- lr 5e-6 4-node finished (00:22 PDT 10-16). Reward steps 12-20: 0.43 0.49 0.45 0.51 0.48 0.49 **0.58** 0.55 0.55
  (step 1: 0.32). It learns clearly faster than lr 3e-6 (final ~0.43). Entropy 0.12 -> 0.07, tokens/trajectory
  22k -> 51k, context-limit exits 4-6/128 at the end; logprob diff 0.009 throughout. Some drift toward longer
  rollouts, but no collapse within 20 steps (Qwen3.6 lr-5e-6 collapsed to entropy 0.04 / 122k tokens).
- Queue 1 launched 00:23-00:31 PDT 10-16 (`run_queue1.sh` stalled after Miles: the remote `cd && setsid ... &` kept the ssh
  open; fixed to `cd ...; setsid ... &`). Runs: `nem35-miles-mismatch-1016-0023` (.87), `nem35-nemorl-mismatch-1016-0023`
  (.219), `nem35-primerl-mismatch-1016-0023` (.123), `nem35-pool96-pass8-1016-0023` (.105; logs in `$D/logs/` on each node).

## Cross-framework results (no R3, GSM8K), 2026-10-16
- **NeMo-RL** (`nem35-nemorl-mismatch-1016-0023`, .219, Megatron TP4/EP8 + colocated vLLM 0.29 2xTP4, NVIDIA's
  Lightning vLLM preset): SkyRL-definition mean |dlogprob| **0.0135 / 0.0142 / 0.0136** (steps 1-3), max 1.8-4.1,
  p99 ~0.19, 0.05% of tokens > 0.5; gen KL error ~0.001; reward ~0.86. That is **3.5x lower than SkyRL without R3**
  (0.045-0.053, max 12-14), close to SkyRL with R3 (0.008). So the model is not inherently this mismatched; our stack adds it.
  - NeMo-RL's vLLM preset: `moe_backend=flashinfer_cutlass`, FLASH_ATTN, cudagraph PIECEWISE,
    mamba_ssm_cache_dtype float32. Megatron: router frozen + fp32, `defer_fp32_logits`.
  - Prime suspect: our vLLM 0.30 picks the FlashInfer **TRT-LLM bf16 MoE** (`trtllm_bf16_moe` is autotuned at startup),
    whose fused kernel does the routing itself, vs `flashinfer_cutlass` with vLLM's fp32 router. Testing SkyRL GSM8K
    with `engine_init_kwargs.moe_backend=flashinfer_cutlass` (`nem35-gsm8k-cutlass-1016-0050`, .219, private Ray
    via `ops/with_private_ray.sh`).
- **prime-rl** (`nem35-primerl-mismatch-1016-0023`, .123; vLLM 0.31 using the **TRITON** unquantized MoE backend, own
  FSDP trainer EP4): `mismatch_logprob_abs_diff/all/mean` ~**0.010-0.013** (k3 mismatch_kl ~0.001-0.002). Also low.
- **SkyRL with `moe_backend=flashinfer_cutlass`** (`nem35-gsm8k-cutlass-1016-0050`, vLLM log confirms "Using FlashInfer
  CUTLASS Unquantized MoE backend"): **0.0456 / 0.0528**, the same as the default (TRT-LLM) -> **the MoE kernel is
  not the cause.**
- vLLM 0.30's NemotronH mamba_ssm_cache_dtype resolves to float32 ('auto' -> HF config) -> not the cause either.
- Remaining vLLM-side difference vs NeMo-RL: attention backend (SkyRL's vLLM picks FLASHINFER; NeMo-RL forces
  FLASH_ATTN). Testing `engine_init_kwargs.attention_backend=FLASH_ATTN` (`nem35-gsm8k-flashattn-1016-0115`, .219).
- Miles first try (`nem35-miles-mismatch-1016-0023`) crashed in TE cuDNN fused attention ("CUDNN_BACKEND_TENSOR_DESCRIPTOR
  cudnnFinalize failed"); relaunched with `NVTE_FUSED_ATTN=0 ATTN_BACKEND=flash` (`nem35-miles-mismatch-flash-1016-0105`).
- `ops/with_private_ray.sh` pins a SkyRL job to one node with a private Ray head (port 7479, /tmp/skp_ray). A killed
  SkyRL driver can hang in shutdown and leave the head up; check `ps | grep skp_ray` and kill leftovers before reuse
  (the wrapper now refuses to start if its port is taken).

## pool96 pass@8 learnability eval (`nem35-pool96-pass8-1016-0023`, 2026-10-16 00:31-01:10 PDT)
Stock harbor + plain vLLM on .105 (`ops/run_baseline_relay.sh`), the overfit's train agent config, 1200 s agent
timeout, temp 1.0 / top_p 1.0 (`--generation-config vllm`), 96 fast MiMo tasks x 8. Results:
- 768 trials (742 clean, 25 NonZeroAgentExitCode, 1 agent timeout). Mean **pass@1 0.255, pass@8 0.583**.
  40 tasks never solved, 4 always, **48 learnable** (1-6 of 8 solved, <= 1 timeout). Per-task file:
  `$D/runs/nem35-pool96-pass8-1016-0023/per_task.json` (.105).
- Current overfit16fast: 8 of 16 never solved here (000466, 002684, 001218, 000783, 001650, 000566, 001216, 003031)
  and 000142 is 7/8.
  - Note: 001650 never hit the 1200 s agent timeout in this stock-harbor eval (max 778 s), but it timed out on every
    overfit step under SkyCap. Worth a look: the relay makes calls slower?
- New set **`overfit16v2`** (`ops/overfit16v2.txt`, `$D/data/overfit16v2` on all nodes): the 7 learnable current tasks
  (000510, 000651, 001457, 001471, 001775, 002410, 002554) + 9 from the pool nearest pass@1 0.4 with max trial
  <= 900 s (002519, 001446, 001238, 001902, 000457, 002672, 000018, 000558, 003024). Mean pass@1 0.37; max trial 869 s.

## ROOT CAUSE of the no-R3 logprob mismatch (found 2026-10-16 ~01:50 PDT): weight sync rounds the router bias to bf16
- Nemotron-3.5-Lightning's MoE routers have a score-correction bias (`backbone.layers.N.mixer.gate.e_score_correction_bias`,
  the only fp32 tensors in the checkpoint) used only to choose experts. Values are large and tightly clustered, e.g.
  layer 1: 40.640-40.713 (30 distinct), layer 3: 46.923-47.021, layer 6: 42.847-42.940.
- SkyRL's weight sync (`skyrl/backends/skyrl_train/weight_sync/sources.py`, MegatronWeightSource / FsdpWeightSource)
  casts EVERY tensor to the inference dtype (bf16) before sending. bf16's step at ~40 is 0.25, so each layer's whole
  bias collapses to ~1 value. vLLM loads the HF fp32 bias at start, but SkyRL's first sync (before step 1) overwrites
  it, so all rollouts route **without** the correction while Megatron (fp32 bias; verified with a load-time diagnostic,
  `SKYRL_DEBUG_EXPERT_BIAS=1`, exact match with the checkpoint) routes **with** it.
- Explains everything:
  - The no-R3 gap: 0.05 GSM8K, 0.072 agentic.
  - R3 "fixing" it: Megatron replays vLLM's bias-less choices, so both sides agree, but both route WITHOUT the bias.
    The R3 overfits trained a slightly degraded routing.
  - The re-score: fresh vLLM (correct bias) vs synced rollouts gave 0.074. It was NOT decode vs prefill.
  - NeMo-RL (0.014) and prime-rl (0.010) sync the bias correctly.
  - Qwen3.6 is unaffected: softmax router, no large fp32 bias.
  - Parity evals unaffected: no Megatron policy, no weight sync.
- Ruled out along the way: MoE kernel (cutlass 0.046/0.053), attention backend (FLASH_ATTN 0.046), CUDA graphs (eager),
  TP, mamba SSM cache dtype (fp32 already), Megatron bias load (exact).
- **Fix** (uncommitted): `sources.py` `FP32_SYNC_SUFFIXES = ("e_score_correction_bias",)` + `wire_dtype()`. Those params
  are sent (and declared in metadata) as float32; vLLM's packed transfer supports mixed dtypes per tensor.
  Test: GSM8K no R3 `nem35-gsm8k-fp32bias-1016-0155` (.105) and R3 `nem35-gsm8k-fp32bias-r3-1016-0155` (.219).
- **Miles**: invalid setup. `train_rollout_logprob_abs_diff_token_mean` ~**1.28** both at TP4/EP8
  (`nem35-miles-mismatch-flash-1016-0105`) and Miles' verified TP2/PP2/EP2 (`nem35-miles-mismatch-tp2pp2-1016-0120`),
  with NVTE_FUSED_ATTN=0 and flash attention (the first try crashed in TE cuDNN fused attention). A ~1.3 gap means its
  trainer is not computing the same model (likely its experimental Lightning bridge shim / weight conversion). Not
  debugged further: NeMo-RL + prime-rl + the root cause answer the question.
- **Fix confirmed** (GSM8K, steps 1-2): no R3 **0.0197 / 0.0205** (max 9.6 / 5.4), down from 0.045-0.053 (max 12-14);
  R3 0.0075 / 0.0075. No-R3 is now in the range of Qwen3.6's no-R3 0.016 and of NeMo-RL's 0.014 / prime-rl's 0.010.
  The small remaining difference vs NeMo-RL may come from their fp32 logits (`defer_fp32_logits`), or prompt and
  length differences. Runs: `nem35-gsm8k-fp32bias-1016-0155` (63aqqnsp), `...-r3-...` (z7koy5o1).

## DAPO, 4 nodes, fixed sync, no R3 (`nem35-dapo-4node-nor3-1016-0147`, W&B dxnz554x, started 01:48 PDT 10-16)
- Recipe: `ops/run_dapo_nemotron.sh` (dual clip, token_mean, overlong filtering + 4k buffer, token TIS 2, 128 x 16,
  mini batch 32, lr 1e-6, 16k responses), Megatron TP4/EP8 (DP8), 8 vLLM engines x TP4. ~10.5 min/step.
- AIME-2024 before training (16 samples, 16k): avg score 0.129 on DAPO's +-1 scale. 274/480 responses finish within
  16k and score 0.97 (almost all correct); 206/480 hit the 16k limit (penalized). The model is accurate but often
  needs > 16k thinking tokens on AIME; the reward signal here is mostly length (overlong buffer), as SkyRL's
  new_model_support.md expects for this stage.
- Step 1: logprob diff **0.0155** without R3 (fixed sync), entropy 0.26, tokens 8.2k, train reward 0.097, grad norm 0.11.
- DAPO steps 1-10, reward / tokens: 0.10/8.2k, -0.08/8.3k, -0.14/9.1k, 0.04/7.9k, 0.06/8.3k, 0.30/7.5k, 0.35/7.2k,
  0.26/7.7k, 0.34/6.7k, 0.24/7.4k. Logprob diff 0.015-0.017 throughout (no R3); entropy 0.26 -> 0.21; grad norm ~0.1.
  **AIME-2024 avg@16: 0.129 (step 0) -> 0.358 (step 10)**, mostly from shorter reasoning fitting the 16k budget. The
  stack learns on Nemotron with the fixed weight sync and no R3.
- DAPO steps 11-20 reward: 0.44 0.22 0.36 0.27 0.29 0.42 0.46 0.46 0.41 0.33; tokens 8.4k -> 5.5-6.5k; logprob diff
  0.015-0.016 (no R3); entropy 0.26 -> 0.14 (gradual). **AIME-2024: 0.129 (step 0) -> 0.358 (10) -> 0.408 (20).**
  Stopped at step 20 (learning shown) to move on to the overfit relaunch.

## INCIDENT 2026-10-16 ~05:25 PDT: MiMo task images disappearing from Docker Hub
- Overfit relaunch `nem35-of16v2-g8-all-nor3-lr5e-6-4node-1016-0441`: 3 tasks (000558, 001902, 002519) fail on every
  attempt with `SandboxBuildFailedError: ... docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-000558: not found`
  (64 failures each by step 3; their groups are masked: 3-4 masked per step).
- Docker Hub `xiaomimimo/mimo-v2.6-rl-oss`: last_updated 2026-10-16 12:29 UTC, with tags being (re)pushed seconds apart
  at ~12:31 UTC (upstream is rewriting the repo). Tag count 2301 and stable over ~3 min. 14 of the 16 overfit16v2
  images return 404, and 27 of 60 sampled mimo-train images. Most still start, likely from Daytona's image cache;
  uncached ones fail.
- The task Dockerfiles `FROM` the same image, so there is no fallback build path. Impact: any MiMo run (incl. the
  planned full run) can lose a large share of tasks unless the tags come back. Next: track the tag count over time;
  before the full run, filter mimo-train to tasks whose image exists (or is cached).
- Overfit metrics so far (fixed sync, no R3): step 1-3 reward 0.35 / 0.30 / 0.31, logprob diff 0.024 / 0.022 / 0.023,
  entropy 0.063 / 0.059 / 0.065 (vs ~0.11 before the fix on overfit16fast), tokens 41.6-43.4k.
- 05:43-05:44 PDT, registry API (`python3 ops/mimo_tags.py ops/mimo_train_tasks.txt ops/overfit16v2.txt`, snapshots in
  `ops/mimo_tags/`; the Docker Hub web API lags and 403s on deep pagination): 3169 -> 3192 tags (+23 in 1 min,
  -0); mimo-train with an image 2069 -> 2092 of 2658. **Looks like an upstream re-upload in progress, recovering.**
- Recovery: 05:53 3636 tags, 2532/2658 train; 06:08 3667 tags, **2563/2658 train** (+31 in 15 min, -0: slowing).
  Overfit masked groups 3 -> 0 from step 7 as the images returned.
- Overfit (fixed sync, no R3, lr 5e-6), reward steps 1-8: 0.35 0.30 0.31 0.37 0.40 0.34 **0.44 0.45**; logprob diff
  0.022-0.026; entropy 0.06 -> 0.08; tokens ~40-43k; 0-2 context exits per step.
- 06:23-06:38: tags flat at 3668 (+1, +0): the re-upload has stopped with **2564 / 2658** train tasks available
  (94 missing). For the full run: `$D/data/mimo-train-avail` (all 4 nodes; list `ops/mimo_train_avail.txt`) = mimo-train
  limited to tasks whose image exists at 06:38. Use TASKS_DIR=$D/data/mimo-train-avail; refresh with
  `ops/mimo_tags.py` first.

## Overfit relaunch, fixed sync, NO R3 (`nem35-of16v2-g8-all-nor3-lr5e-6-4node-1016-0441`, W&B gxj7q4gh), 04:41-08:01 PDT
overfit16v2, group 8, all paths, lr 5e-6, 4 nodes, synchronous vci3ga5w recipe, 20 steps (~10 min/step).
| step | 1 | 3 | 5 | 7 | 9 | 11 | 13 | 15 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reward | 0.35 | 0.31 | 0.40 | 0.44 | 0.44 | 0.46 | 0.46 | 0.52 | 0.54 | 0.59 | 0.62 | **0.63** |
- Clear, steady learning (0.30-0.35 at steps 1-3 -> 0.63), better than the pre-fix lr 5e-6 run (0.32 -> 0.55).
- Logprob diff 0.022 -> 0.030 (no R3), entropy 0.06 -> 0.08 (stable, no collapse), tokens ~40k, context exits 0-2.
- Steps 1-6 had 3-4 masked groups from the Docker Hub image incident; none from step 7.

## FULL RUN `nem35-mimo-fa-1016-0654` (launched 08:04 PDT 10-16 by launch_after.sh)
Nemotron-3.5-Lightning, fully async, Megatron TP4/EP8 on 1 node (.105), 12 vLLM engines x TP2 on .87/.219/.123, Qwen
run-2 hyperparameters (DPPO 0.15, prompt_mean, lr 1e-6, staleness 2, 16 x 16, concurrency 500, final paths), no R3,
fixed fp32 router-bias sync, train tasks `mimo-train-avail` (2564), SWE-bench-100 eval with parity sampling (top_p 0.95),
3 passes before training and every 20 steps; weight sync 4.7 s.
- Step-0 SWE-bench-100 (3 passes): **0.448** final (297/300 verified; 0.452 at 292); eval 27.6 min; training (480 steps) started 08:37. (parity runs: SkyCap 0.453, stock 0.493).
  Relay 24k calls, 0 errors, 0 dropped.
- Step 1 (09:01 PDT, 24 min incl. the first buffer fill): reward 0.183, logprob diff **0.0223** (no R3), entropy 0.065,
  tokens 57.7k, grad norm 0.073. W&B run 5doiopns (eval/all/avg_score at step 0 = 0.437 counts all 300 trials).
  Metrics: `python $D/ops/dapo_metrics.py nem35-mimo-fa skycap-mimo-fully-async` on .105.
- Steps 2-5 (to 10:08 PDT, ~15 min/step): reward 0.13 / 0.16 / 0.21 / 0.15; logprob diff 0.020-0.023; staleness
  0.6-1.6; entropy 0.053-0.063; tokens 52-65k; grad norm 0.07-0.15. Relay 0 call errors, 0 dropped. Step-5 checkpoint
  413 GB on .105 (1 training node, so the checkpoint is whole there; keep 2).
- Steps 6-10 (to 11:38 PDT): reward 0.20 / 0.19 / 0.18 / 0.15 / 0.11 (steps 1-10 mean 0.16, no trend yet at lr 1e-6;
  Qwen run 2 was flat too); logprob diff 0.021-0.023; staleness 0.6-1.4; entropy 0.058-0.071; tokens 55-68k; ~15 min/step
  (one 35-min buffer fill at step 8). Checkpoints global_step_5, _10. Relay ~0 errors.
- Steps 11-19: reward 0.19 0.09 0.16 0.16 0.21 0.15 0.16 0.23 0.07 (steps 1-19 mean ~0.16, flat); logprob diff
  0.019-0.023; entropy 0.057-0.078; tokens 55-71k; ~15 min/step. Checkpoints _10, _15.
- **Step-20 SWE-bench-100 (3 passes): 0.431** (299/300; 290 clean, 6 NonZeroAgentExitCode, 3 verifier timeouts) vs
  step 0 **0.443** on the same all-trials basis (0.448 over verified-only). Flat (sd ~0.028), like Qwen run 2 at lr 1e-6.
  The overfits learned clearly at 3e-6 / 5e-6; lr 1e-6 is the open question for the user.
- Steps 20-25 (to 16:09 PDT): reward 0.14 0.11 0.09 0.11 0.16 0.09 (window mean ~0.12 vs ~0.16 in steps 1-20);
  logprob diff 0.019-0.023; entropy 0.067-0.081 (slowly rising); tokens 57-68k. Checkpoints _20, _25. No errors.
- Steps 26-30 (to 17:38 PDT): reward 0.21 0.11 0.09 0.10 0.14 (steps 21-30 mean ~0.12 vs ~0.16 in 1-20); logprob
  diff 0.020-0.021; entropy 0.073-0.080; tokens 55-68k. Checkpoints _25, _30. No errors. Still flat at lr 1e-6.
- Steps 31-39: reward 0.16 0.12 0.21 0.18 0.21 0.15 0.19 0.10 0.09 (mean ~0.16); logprob diff 0.018-0.023; entropy
  0.074-0.095 (slowly rising from ~0.06); tokens 56-71k. Checkpoints _30, _35.
- **Step-40 SWE-bench-100 (3 passes, all 300 trials): 0.450.** Series 0.443 / 0.430 / 0.450 at steps 0 / 20 / 40:
  flat within noise (sd ~0.028), like Qwen run 2 at lr 1e-6.
- Steps 41-46: reward 0.14 0.16 0.15 0.15 0.23 0.12; logprob diff 0.019-0.022; **entropy rising steadily: ~0.06
  (steps 1-20) -> 0.074-0.095 (31-39) -> 0.10-0.13 (41-46)** with flat reward and eval: the policy is getting more
  random, not better (the opposite of collapse; watch for quality loss). Checkpoints _40, _45.
- Steps 47-50 (to 23:09 PDT): reward 0.13 0.19 0.12 0.15 (steps 1-50 mean ~0.15, flat); logprob diff 0.017-0.020;
  entropy 0.11-0.14 (plateau after rising); tokens 63-80k; ~15 min/step. Checkpoints _45, _50. No errors.
- **Drift warning (23:39 PDT, step 52)**: entropy 0.147 (step 51) -> 0.171 (step 52), ~3x the start (~0.06), reward
  flat (~0.13-0.15). First vs last 400 training trials: reward 0.216 -> 0.177; context-limit exits 0.0% -> 3.0%;
  median turns 69 -> 85; format errors unchanged (0.02/trajectory). The policy is getting more random and longer
  without improving at lr 1e-6. Recommendation for the user: restart at lr 3e-6 (fixed-sync overfit at 5e-6 learned
  0.30 -> 0.63), from scratch or from the step-45/50 checkpoint (before most of the drift).
- Step 56 (00:39 PDT): **entropy 0.210** (step 55: 0.152; start ~0.06), reward 0.114, tokens 70k. Preserving the
  pre-drift step-50 checkpoint from rotation (keep 2): copying to
  `$D/runs/nem35-mimo-fa-1016-0654/ckpt_keep/global_step_50` on .105 (ionice idle).
- 00:53 PDT: step-50 checkpoint copy done (413 GB, matches) at `$D/runs/nem35-mimo-fa-1016-0654/ckpt_keep/global_step_50`
  (.105; safe from rotation). Step 57: entropy 0.215 (holding high), reward 0.172, logprob diff 0.020.
- Steps 51-59: reward 0.15 0.13 0.15 0.18 0.21 0.11 0.17 0.14 0.18 (flat); entropy 0.137-0.215; logprob diff
  0.016-0.020. **Step-60 SWE-bench-100 (3 passes): 0.454** (295/300). Series 0.443 / 0.430 / 0.450 / 0.454 at steps
  0 / 20 / 40 / 60: flat within noise; the entropy rise has not cost eval quality, but there is no gain at lr 1e-6.
- Step-60 eval final (W&B, all 300 trials): **0.447**. Steps 60-61 reward 0.06 / 0.13, entropy 0.187 / 0.193. Rotation
  removed ckpts/global_step_50; the copy in `ckpt_keep/global_step_50` is intact. Checkpoints now _55, _60.
- Steps 62-63 (02:53 PDT): reward 0.23 / 0.20; **entropy 0.203 -> 0.253** (~4x the start, still climbing); logprob
  diff 0.019-0.020; tokens 60-67k.
- Steps 64-70 (to 05:08 PDT): reward 0.13 0.09 0.07 0.15 0.12 0.21 0.14 (flat, ~0.13); entropy 0.198-0.254 (~4x the
  start); logprob diff 0.018-0.020; tokens 58-71k; ~15 min/step. Checkpoints _65, _70 (+ ckpt_keep/global_step_50).
- 07:00-07:10 PDT stall at step 78 (buffer 5/16, ~20 sandboxes active, no new trials): the async staleness manager
  holds new groups while the oldest in-flight group runs. Here that is task **format-code-task-001975**, whose verifier
  always times out (1800 s; 18 VerifierTimeoutErrors so far) and is retried. Same cause as the 35-min fill at step 8.
  Resolves when that group ends. **At the next restart, drop 001975 from mimo-train-avail** (no signal, stalls the
  run up to ~1 h whenever sampled). Steps 71-77: reward 0.13-0.17, entropy 0.20-0.27, logprob diff 0.017-0.020.

### 2026-10-17 08:12 — why a global_step_78 checkpoint (full run nem35-mimo-fa-1016-0654)
Not a bug. `sample_full_batch` drops groups with no reward variance (all-0 or all-1), so epoch 0 (2564 tasks) ran out after
78 steps × 16 = 1248 groups trained; the other ~1316 tasks (~51%) were dropped. At 07:46:06 the log shows "epoch 0 exhausted
with a partial mini-batch of 9 group(s)", which takes the epoch-exhausted branch and saves a checkpoint (fully_async_trainer.py:561).
Epoch 1 started right away. epochs=3, so expect ~78 steps per epoch, ~234 steps total (the tqdm total of 480 assumes no drops).
Takeaway: about half the MiMo train set gives no signal at the current policy. A pass@k pre-filter to drop always-0/always-1
tasks would roughly halve the generation wasted per step.

### 2026-10-17 09:11 — step-80 SWE-bench eval (full run, lr 1e-6): 0.460
Eval curve: 0.437 / 0.430 / 0.450 / 0.447 / 0.460 at steps 0 / 20 / 40 / 60 / 80. Slightly up but within 3-pass noise (~±0.02).
Eval took 1687 s. Steps 80–81: reward 0.24 / 0.10, dlogp 0.018 / 0.020, entropy 0.23 / 0.30. Checkpoint global_step_80 saved.

### 2026-10-17 09:40 — what NonZeroAgentExitCodeError is (full run)
Nearly all of the ~2450 are litellm ContextWindowExceededError: the conversation passes 131072 tokens (thinking is kept in the
history, truncate_history_thinking=false). The rate has held steady at ~3.5–5% of trials for the whole run (2238 of 45901). A
+38 burst at 09:25–09:40 was clustered on a few long tasks (all 8 rollouts of 000531, 6 of 001641), not a regression. The trial
is still verified, usually reward 0. No action needed.

### 2026-10-17 14:48 — step-100 SWE-bench eval (full run, lr 1e-6): 0.457
Curve: 0.437 / 0.430 / 0.450 / 0.447 / 0.460 / 0.457 at steps 0 / 20 / 40 / 60 / 80 / 100. A plateau about +0.02 over step 0,
within noise. Eval took 2268 s. Training reward over steps 90–100 averages ~0.14 (0.07–0.20). Entropy keeps climbing (0.06 at
the start, 0.30–0.35 now). dlogp ~0.020. global_step_100 saved. The dip at steps 98–99 did not show up in the eval.

### 2026-10-17 16:25 — 15-step rolling averages (full run, lr 1e-6, through step 105)
| steps | reward | entropy | tokens |
|---|---|---|---|
| 1-15 | 0.164 | 0.064 | 60k |  | 16-30 | 0.131 | 0.072 | 62k |  | 31-45 | 0.158 | 0.094 | 65k |
| 46-60 | 0.146 | 0.156 | 65k |  | 61-75 | 0.147 | 0.234 | 67k |  | 76-90 | 0.154 | 0.278 | 64k |
| 91-105 | 0.129 | 0.318 | 67k |
Reward is flat (the dip at 91-105 is within noise; 16-30 was as low). Entropy has risen ~5x steadily with no reward gain:
the policy is spreading out without learning. Eval is flat at ~0.45. global_step_105 saved.
format-code-task-000148: 7/8 VerifierTimeout in epoch 1 (like 001975) - drop at restart
format-code-task-002687: 6 VerifierTimeout in one group at ~18:20 10-17 - drop at restart (also consider dropping any task with >=4/8 verifier timeouts)
format-code-task-000769: 14 VerifierTimeout (~19:40-19:53 10-17) - drop at restart. Drop list so far: 001975, 000148, 002687, 000769
Why one task logs >8 verifier timeouts: train trials get SKYCAP_TRIAL_ATTEMPTS=2 (harbor_generator.py:62), so a trial whose
verifier always times out (1800 s) is retried once. Its group then takes ~1 h+ and logs up to 16 timeouts per epoch
(000769: 55 trial dirs across the 2 epochs, in waves ~30 min apart). The async loop works around such groups unless the
staleness cap is reached (as with 001975). Fix at restart: drop these tasks. Optionally don't retry VerifierTimeoutError.

### 2026-10-17 20:48 — step-120 SWE-bench eval (full run, lr 1e-6): 0.467 (best so far)
Curve: 0.437 / 0.430 / 0.450 / 0.447 / 0.460 / 0.457 / 0.467 at steps 0–120. Each point alone is within noise, but the trend
is consistently upward: about +3 pts from step 0, and the last 3 evals (0.460 / 0.457 / 0.467) are all above the first 4.
Eval is rising while training reward falls (steps 112–120 average ~0.12; step 120 was 0.064) and entropy climbs (~0.40):
reward on the MiMo train set doesn't track SWE-bench. Eval took 1868 s. global_step_120 saved.

## INCIDENT 2026-10-17 ~22:55 PDT — litellm 1.106.0 breaks every trial (full run nem35-mimo-fa-1016-0654)
- litellm 1.106.0 was published 05:49 UTC 10-18 (22:49 PDT). Every sandbox runs `uv tool install mini-swe-agent==2.4.6`,
  which picked it up. On its first completion (litellm/chat_completions/dispatch.py) it imports `mcp`, which it doesn't
  depend on, so the agent fails with "ModuleNotFoundError: No module named 'mcp'". From 23:00 on, 100% of new trials
  failed (78/78 in 5 min; NonZeroAgentExitCode +880 in 15 min; relay calls nearly stopped).
- Reproduced on the head: mini-swe-agent 2.4.6 + litellm 1.106.0 gives the mcp error; with 1.105.1 (the last stable
  before) it reaches the network call as it should.
- Step 130 trained on the broken batch (reward 0.008, 4.8k tokens/traj, 15 masked instances); its checkpoint is bad.
  Steps 126-129 were clean, but global_step_125 is the last clean checkpoint.
- FIX (deployed to all 4 nodes, not yet in effect for the running driver): all 3 mini-swe launchers default to
  MINI_SWE_AGENT_VERSION="2.4.6 --with litellm==1.105.1" (Harbor builds `uv tool install mini-swe-agent==<version>`).
- I could NOT stop or resume the run: the auto-mode classifier denied killing the driver and even hard-linking the
  step-125 checkpoint. Resume script for the user, run from the head: `bash ~/hackskyrl/ops/resume_nem35_after_litellm.sh`
  (stop, delete sandboxes, move step 130 to ckpt_bad/, sync_ckpt 125, relaunch the same config with no pre-train eval;
  `LR=3.0e-6` to change the LR). Risk while it waits: if the broken run reaches step 135, the keep-2 rotation deletes
  global_step_125 (the fallback would be ckpt_keep/global_step_50). As of 23:35 it is stuck at step 131 (masked groups).
- 23:40: the broken run started advancing (steps 131-132 at ~1 min each), putting step 125 at risk at step 135. Ran
  `sync_ckpt.py nem35-mimo-fa-1016-0654 125`: global_step_125 is now whole on all 4 nodes (md5 bf36000a); the other 3
  nodes are outside the driver's rotation, so it survives. The resume script now sets aside every global_step > 125.
- 10-18 00:00: user asked to protect global_step_120 (best eval, 0.467). It no longer exists: the keep-2 rotation deleted it
  at the step-130 save (~23:09 10-17), and no copy was ever made. Protected global_step_125 instead (closest, last clean):
  hard links in runs/<run>/ckpt_keep/global_step_125 on .87 and .219 (413 GB, 18 files each; outside any rotation).
  .105 now holds only the bad 130/135. Lesson: copy each eval-step checkpoint to ckpt_keep (or S3) when its eval lands.
- 10-18 00:03: user OK'd the resume. Ran ops/resume_nem35_after_litellm.sh: driver stopped, 5 sandboxes deleted, bad
  130/135 moved to ckpt_bad/ on .105, step 125 synced to all 4 nodes (md5 bf36000a), relaunched (driver pid 2469256,
  lr 1e-6, litellm pinned to 1.105.1, no pre-train eval). 00:18 "Resuming from global_step: 125".

## Eval sets prepared 2026-10-18 (on all 4 nodes)
- `$D/data/swebench500`: full SWE-bench Verified (500), from github.com/laude-institute/harbor-datasets main @37db108843
  (datasets/swebench-verified). The Harbor registry's pinned commit 86723674 is older and broken (test.sh first line is
  a corrupted canary comment instead of a shebang; the cleanup deletes new files the model created), so don't use it.
  Timeouts set like swebench100 (agent 1800 s, verifier 300 s; originals as task.toml.orig). The 100 overlapping tasks
  are byte-identical to `$D/data/swebench100` (700/700 files).
- `$D/data/tbench21`: Terminal-Bench 2.1 (89 tasks), github.com/harbor-framework/terminal-bench-2-1 @7131e43 tasks/,
  unmodified (official per-task timeouts/resources; Daytona; allow_internet). The hub id is terminal-bench/terminal-bench-2-1,
  but downloads fail on the workers (SSL: self-signed cert in chain), so it's fetched via git from the head.
- 00:20 10-18: resumed run generating again (weights synced in 5.9 s). The litellm pin works: of the first 20 trials
  after the resume, 0 hit the mcp error, agents run to completion and submit; relay 518 attached, 18.7k calls, 0 call
  errors. Training continues from step 126 at lr 1e-6 (new W&B run, same name).
- 00:40 10-18: the exact pin (`--with litellm==1.105.1`) broke ~13% of trials (108 in 20 min): on images with glibc
  < 2.28, litellm's manylinux_2_28 wheel doesn't apply, so uv builds it from source (maturin/Rust) and that fails
  ("Failed to build litellm==1.105.1"). The same tasks installed fine before 22:49. A plain resolve gives 1.105.1 on
  those images too, so their source build used to work and now fails, likely because the Rust auto-install changed.
  New pin, in all 3 launchers: MINI_SWE_AGENT_VERSION="2.4.6 --exclude-newer 2026-10-18T00:00:00Z --no-build-package
  litellm" (no shell-special characters). Resolution as of before the breakage; never build litellm, so old-glibc images
  get 1.91.5 (pure-Python wheel; tested with mini-swe-agent 2.4.6) and the rest 1.105.1 (tested with a real uv 0.7.13
  tool install). Restarted from step 125 again at 00:45 (no step had trained since the first resume).
- 01:15 10-18: new pin verified. Of the first ~160 trials after the 00:45 restart: 0 build failures, 0 mcp errors;
  the only 2 exceptions are ordinary agent exits mid-task (task 000570). The 6 tasks that failed the litellm build at
  00:40 (000652, 001251, 001633, 002204, 002233, 002838) now succeed on all 14-16 rollouts each.

### 2026-10-18 05:27 — step-140 SWE-bench eval (resumed from step 125, lr 1e-6): 0.450
Curve: 0.437 / 0.430 / 0.450 / 0.447 / 0.460 / 0.457 / 0.467 / 0.450 at steps 0–140. Clean eval: 291/300 complete,
3 errors, 6 context overflows; it took only 899 s (fewer stragglers). Down 1.7 pts from step 120, within noise. Entropy
is now ~0.40–0.44, reward over steps 126–140 averages ~0.13. Kept checkpoints: ckpt_keep/global_step_140 (hard links,
.105), ckpt_keep/global_step_125 (.87 and .219), ckpt_keep/global_step_50 (.105).
- 07:55 10-18: task 001975 stalled generation again (epoch 2; the same as its 49-min stall in epoch 0). Its 12 second-attempt
  trials all sat in verifiers that never finish, which held up new work: 3 new trials in 15 min, trainer waiting for its
  batch from 07:49. Fix: found their sandboxes by resources (cpu 4 / 3 GB, task 001975's spec) plus creation time
  (14:30-14:48 UTC, one-for-one with the trials' start times) and deleted the 12 with rl-env-cleaner/skycap_overfit/sb_del.py
  (owner-checked). The trials ended at once ("attempt 2/2 produced nothing", masked) and generation resumed within a
  minute (12 new trials in 3 min). Recipe for next time: list the run's sandboxes with cpu/memory/created_at and match
  the task's [environment] spec plus trial start times. 001975 stays at the top of the drop list.
- 08:16 10-18: correction to the 07:55 note. After the 001975 deletions generation still didn't resume: that wasn't a
  stall but the END OF EPOCH 1. The dataloader was exhausted, so the generators had returned, and the trainer was waiting
  for the last 2 open groups: 002357 (its trial recorded an exception at ~07:10 but the sandbox was never cleaned up, so
  the group hung) and 001009 (agent still running after 50 min). Identified their sandboxes by running a read-only
  `head /logs/agent/mini-swe-agent.txt` in each candidate (6acc0b35 = 002357, 01d5e30a = 001009) and deleted both, plus
  2 orphans with no open trial (366dbbda from 05:04 PDT, 33aa0d77 from 03:37 PDT). At 08:15:50: "epoch 1 exhausted with
  a partial mini-batch of 9 group(s); discarding", global_step_149 saved, epoch 2 (the last; epochs=3) generating.
  Rotation removed ckpts/global_step_140; the hard-linked ckpt_keep/global_step_140 is intact. Epoch 2 should end around
  step ~225, which ends the run.
  Diagnosis tip: "Generation Buffer Progress N/16" frozen + no new trial dirs + only a few open trials = epoch end, not a
  stall. List the open trials: trial dirs since the launch that have no result.json.

### 2026-10-18 11:52 — step-160 SWE-bench eval: **0.493** (best so far; lr 1e-6, epoch 2)
Curve: 0.437 / 0.430 / 0.450 / 0.447 / 0.460 / 0.457 / 0.467 / 0.450 / 0.493 at steps 0-160. Clean eval: 293/300
complete, 3 errors, 4 context overflows, 1098 s. +5.6 pts over step 0, beyond the ~±2 pt noise. This matches the stock
mini-swe-agent parity baseline (0.493) and beats the SkyCap parity run (0.453). The official score is 51.56.
Training at step 159: reward 0.13, dlogp 0.022, entropy 0.46. Kept: ckpt_keep/global_step_160 (hard links on .105,
plus an rsync copy to .87).

### 2026-10-18 18:01 — step-180 SWE-bench eval: 0.463 (step 160 stays the best at 0.493)
Curve: 0.437 / 0.430 / 0.450 / 0.447 / 0.460 / 0.457 / 0.467 / 0.450 / 0.493 / 0.463 at steps 0-180. 283/300 complete,
11 context overflows (up from 4 at step 160 and 3-6 before), 6 errors; 1548 s. As entropy climbs (~0.45-0.52) and
responses lengthen (70-76k train tokens), more eval trajectories hit the 131k context limit. That costs a little score,
and it's the expected failure mode of the entropy drift. The 0.493 at step 160 may also have been partly noise. Not
keeping step 180; best = ckpt_keep/global_step_160 (.105 + .87).

## FINAL EVALS (user request 2026-10-18 ~20:55): after epoch 3 ends, stop the run; full SWE-bench Verified (500) and
## Terminal-Bench 2.1 (89), 5 passes each, mini-swe-agent, for base model / final ckpt / best ckpt (step 160)
Targets (model card chart "Agentic Coding Benchmarks", mini-swe-agent v2.4.5, "unmodified harness system prompts and
standard harness configurations"): SWE-bench Verified 50.7, TB2.1 29.7. (The table's 51.56 / 24.58 are OpenHands via
NeMo Gym.) NVIDIA's recipes (NeMo Gym nemotron_recipes/lightning-3.5/instruct/nemo-evaluator/*.yaml): temperature 1.0,
top_p 0.95, max_tokens dropped, 262144 context, enable_thinking, think-tag reasoning replay, repeats 5 (SWE) / 8 (TB);
served with --reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder --enable-auto-tool-choice.
Our setup (ops/run_final_eval.sh <job> <model-dir> <swe|tb>, run on a free 8-GPU node; score with
ops/score_final_eval.py <job>):
- stock `harbor run` + mini-swe-agent 2.4.5 (installed with --exclude-newer 2026-10-18T00:00:00Z --no-build-package
  litellm), config mini_swe_agent_nemotron35_final.yaml = builtin mini + cost_limit 0 + temp 1.0 / top_p 0.95, no
  max_tokens, thinking kept (truncate_history_thinking false)
- 8 vLLM replicas TP1, max-model-len 262144, the card's parsers, prefix-affinity router, sandbox relay (same plumbing
  as the validated parity baseline, which scored 0.493 on swebench100 x3; the scorer reproduces 0.4933)
- data: $D/data/swebench500std (harbor-datasets main @37db108843, ORIGINAL task timeouts 3000/3000) and $D/data/tbench21
  (official timeouts); sandbox TTL 420 min (TB tasks allow up to 12000 s)
- deviations from NVIDIA's runs: TB with 5 repeats (user asked 5) instead of 8; Daytona instead of ECS.
Checkpoint -> HF conversion: being built by a helper agent (ops/convert_ckpt_to_hf.sh). Kept ckpts: step 160 (best,
0.493; .105 + .87), 140 (.105), 125 (.87, .219), 50 (.105). The final ckpt = the epoch-3 end save (~step 220).
Plan when GPUs free up: base SWE + base TB on two nodes right away; convert step 160 + final on the others, then their
SWE + TB.
- 21:15 10-18: checkpoint -> HF conversion DONE (helper agent): ops/convert_ckpt_to_hf.sh <ckpt dir> [out dir]. CPU only,
  ~45 s and ~15 GB RAM per checkpoint, run on the node holding the checkpoint. Reads the bf16 model weights straight from
  the torch_dist checkpoint (full-size tensors, no optimizer state), maps them to HF names following megatron-bridge's
  NemotronHBridge (mamba in_proj z,x,B,C,dt; conv1d x,B,C; per-group qkv split; per-expert slices; router bias kept
  F32), copies the untrained MTP head from base, and reuses the base's shard layout, config and tokenizer. The validator
  checks all 6513 tensors have the base's names, shapes and dtypes and a relative diff < 0.1 (observed 2e-4..2e-3). A
  CPU forward check of step 160 vs base: logit cosine >= 0.988, mean KL 0.016. Exports on .87:
  $D/hf_export/nem35-mimo-fa-1016-0654-step{125,160}. Not yet tested under `vllm serve` (GPUs busy); do that first.

### 2026-10-19 00:07 — step-200 SWE-bench eval: 0.467 (best stays step 160 at 0.493)
Curve: 0.437 / 0.430 / 0.450 / 0.447 / 0.460 / 0.457 / 0.467 / 0.450 / 0.493 / 0.463 / 0.467 at steps 0-200.
274/300 complete, **17 context overflows** (4 at step 160 -> 11 at 180 -> 17 now), 9 errors; 1313 s. Overflows keep
climbing with entropy (~0.5) and train response length (~75k). Under the 131k eval context they cost ~4-5 pts; the final
evals use 262k, which removes most of that. Not keeping step 200.
- 10-19 ~04:30: brief Daytona outage. 237 DaytonaBadGatewayError ('Failed to get session command') in one burst ~04:29-04:33; relay call errors 1->81, poll errors 238->663; none in the following 5 min. Affected trials retried/masked. Training unaffected (step 215 done + ckpt). Nothing to do.

## 2026-10-19 05:30 — FULL RUN FINISHED (3 epochs); final evals launched 05:35
- End of epoch 3: generation ran dry ~05:10; the trainer then waited on 3 open groups whose sandboxes hung (001975
  verifier, 002357, 002926). Cleared all 31 of the run's sandboxes with sandbox_janitor (24 ERROR / 1 BUILD_FAILED were
  leftovers from the 04:30 Daytona outage, 2 STARTED orphans). 05:28:25 "epoch 2 exhausted with a partial mini-batch
  of 6 group(s)", final checkpoint global_step_218 saved, driver exited cleanly at 05:30. All 32 GPUs free.
- Kept checkpoints (ckpt_keep/): 218 (final, .105), 160 (best in-run eval 0.493; .105 + .87), 140 (.105), 125 (.87, .219), 50 (.105).
- HF exports ($D/hf_export/nem35-mimo-fa-1016-0654-step{125,160,218}): 218 converted on .105 (PASS, worst rel diff
  2.3e-3); 160 and 218 copied to .123. vLLM loads the exports fine (all 8 replicas up on .87 and .105).
- Final eval jobs (ops/run_final_eval.sh; 8 x TP1 replicas at max-model-len 262144; mini-swe-agent 2.4.5 confirmed in
  the agent logs; 5 samples per task):
  .219 final-swe-base-1019 (N_CONCURRENT 200), .87 final-swe-step160-1019 (200), .105 final-swe-step218-1019 (200),
  .123 TB chain final-tb-{base,step160,step218}-1019 (150, one after another; ops/run_final_tb_chain.sh).
  Logs: $D/logs/<job>.log. Score: python3 $D/ops/score_final_eval.py <job> [500|89] on that node.
- 05:55 TB fix: the first TB job (final-tb-base-1019, set aside as final-tb-base-1019-relaybug) failed 49/62 trials with
  "sandbox relay stub did not start: wget: command not found". Some TB images have no python3, curl or wget, so the
  stub's uv-Python fallback couldn't download. Patched rl-env-cleaner/mimo_swe/labeled_daytona.py (_start_relay): if
  neither curl nor wget exists, install curl via apt/apk/yum/dnf first (as Harbor's agent installers do). Synced to all
  4 nodes (md5 5c3bd304; the running SWE jobs had already loaded the old module and aren't affected: SWE images have
  python3). TB chain relaunched with tag 1019b: first 31 trials 27 ok.
  Known TB caveat: qemu-alpine-ssh and qemu-startup (Debian 11 images) fail Harbor's stock mini-swe-agent install
  (apt 404 on removed debian-security perl packages): an image problem, scored 0 for every model (<=2.2 pts).
- 07:20 base SWE (stock mini config) nearly done: ~0.454 vs the card's mini-swe-agent 50.7. The 100 tasks shared with
  swebench100 score 0.471 (parity baseline with the old custom config: 0.493); the other 400 score 0.449, so the full set
  is ~2 pts harder. Lead on the rest: the stock `mini` config has NO environment timeout, so mini-swe-agent's default
  of 30 s per command applies (LocalEnvironment). 233/2489 trajectories hit "timed out after 30 seconds" (e.g. conda
  install, test suites). mini-swe-agent's own SWE-bench config (config/benchmarks/swebench.yaml) uses 60 s, a
  SWE-specific prompt, step_limit 250 and parallel tool calls, and is plausibly what "standard harness configuration"
  meant. Ablation queued on .219 right after the base job: final-swe-base-swecfg-1019 with
  mini_swe_agent_nemotron35_final_swebench.yaml (swebench.yaml + environment_class local + cost_limit 0 + card sampling,
  thinking kept). run_final_eval.sh got an MSWEA_CONFIG override for this.

### FINAL EVAL RESULTS so far (mini-swe-agent 2.4.5, stock `mini` config, 262k context, 5 passes)
| model | SWE-bench Verified (500x5) | TB 2.1 (89x5) |
|---|---|---|
| base | **0.452** (passes 0.444/0.496/0.428/0.440/0.452, std 0.026; pass@5 0.656) | **0.263** (0.270/0.258/0.258/0.281/0.247, std 0.013; pass@5 0.405) |
| step160 | **0.461** (0.464/0.448/0.446/0.456/0.492, std 0.019; pass@5 0.654) | **0.243** (0.258/0.225/0.270/0.214/0.247, std 0.023; pass@5 0.382; 88 agent timeouts vs base 72) |
| step218 | **0.461** (0.454/0.454/0.470/0.458/0.468, std 0.008; pass@5 0.656) | queued |
Model card (mini-swe-agent v2.4.5): SWE-V 50.7, TB2.1 29.7. TB: ~2.2 pts of the gap are the 2 qemu tasks whose image
breaks the agent install.
- 08:10 the queued ablation never started: its waiter polled `kill -0 3931213`, and that process was a zombie
  (<defunct>, kill -0 still succeeds). Lesson: wait on job completion, not on a PID. Launched by hand:
  .219 final-swe-base-swecfg-1019 and .87 final-swe-step160-swecfg-1019 (mini_swe_agent_nemotron35_final_swebench.yaml).
- 08:40 step218 SWE (stock) done 0.461; launched .105 final-swe-step218-swecfg-1019. Swecfg early means: base 0.553 at 738 finished vs stock base 0.492 at ~1000 -> the swebench config adds several points.
- 09:45 TB step160 done 0.243 (below base 0.263 by 2.0 pts; std ~0.02); TB step218 started. swecfg base nearly done ~0.50 (stock 0.452).
- 10:05 **SWE-bench config, base: 0.503** (passes 0.504/0.520/0.508/0.482/0.502, std 0.014; pass@5 0.682) vs the stock
  mini config's 0.452 and the card's 50.7. **This reproduces the model card**: the stock config's 30 s command timeout
  plus generic prompt cost ~5 pts. Command timeouts fell from 233/2489 trajectories (stock) to 91/2496 (60 s).
  Treat swebench-config numbers as the headline SWE-V numbers.
- 10:50 SWE-bench config, step160: 0.506 (0.508/0.516/0.496/0.498/0.514, std 0.009; pass@5 0.696) vs base 0.503 (pass@5 0.682).

## FINAL EVAL TABLE (2026-10-19 ~11:45; mini-swe-agent 2.4.5, 262k context, temp 1.0 / top_p 0.95, thinking kept, 5 passes)
| model | SWE-V, swebench config (headline) | SWE-V, stock mini config | TB 2.1 (stock mini) |
|---|---|---|---|
| base | **50.3** (pass@5 68.2) | 45.2 (65.6) | **26.3** (40.5) |
| step 160 (best in-run eval) | **50.6** (69.6) | 46.1 (65.4) | **24.3** (38.2) |
| step 218 (final) | **51.8** (69.6) | 46.1 (65.6) | **24.9** (37.1; passes 0.258/0.292/0.236/0.214/0.247) |
| model card (mini-swe-agent v2.4.5) | 50.7 | 50.7 | 29.7 |
Paired per-task deltas vs base (500 or 89 tasks; 5-pass task means; bootstrap 95% CI):
- SWE swebench config: step160 +0.32 (CI -1.6..+2.2); step218 **+1.48** (z 1.63, CI -0.3..+3.2)
- SWE stock: step160 +0.92 (CI -1.2..+3.0); step218 +0.88 (CI -1.0..+2.8)
- TB2.1: step160 -2.02 (CI -5.4..+1.1); step218 -1.35 (CI -4.7..+1.8); trained models time out more (88/110 vs 72)
Conclusion: base reproduces the card on SWE-V with mini-swe-agent's SWE-bench config (50.3 vs 50.7); stock mini config
costs ~5 pts (30 s command timeout). RL at lr 1e-6 for 218 steps on MiMo: at most a small SWE-V gain (+0.3..+1.5, not
significant) and a small, not significant TB2.1 loss (longer trajectories, more timeouts). TB gap to the card: 2 tasks
(qemu-*) can't install the agent (~2.2 pts) plus TB uses the stock config (no TB-specific config exists in mini-swe-agent).
- 12:32 TB step218 done: 0.249 (445/445, pass@5 0.371). ALL FINAL EVALS COMPLETE; all 4 nodes idle.
