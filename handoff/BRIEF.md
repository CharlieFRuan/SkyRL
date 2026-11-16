# Brief: train Nemotron-3.5-Lightning on CalibForge (Dune, no R3); throughput work comes after

Written 2026-11-15 for an agent starting on a new, larger GPU machine. Two pieces of work:

- **Part 1 (do this now). Nemotron-3.5-Lightning.** Measure the base model on Terminal-Bench 2.1 with our own
  eval settings through SkyCap, run a 15-step overfit on CalibForge, then one full epoch on CalibForge v2 with
  the recipe of our Qwen3.5-9B batch-32 run, without R3. The user cares more about this model than about
  Qwen3.5-9B: it is what will probably be trained from here on.
- **Part 2 (later, not now). Throughput of the fully async loop.** Make the number of trajectories actually
  running visible in W&B, find out with measurements what limits it, then raise it. **Do not start it and do
  not change the async loop's behaviour before the Nemotron run**: the user wants that run as the un-optimized
  baseline that any optimization is later compared against. Part 2 is kept here so the plan is not lost.

Sandboxes come from **Dune** for everything here (see section 1 for the rules on that key).

Everything stated as fact below was measured or read from code on the source machine; where something is my
inference or was not checked, it says so. File paths are relative to the SkyRL checkout unless they start with
`ops/`, which is `handoff/ops/` on this branch (section 2).

---

## 1. Ground rules (the user's standing instructions)

1. **The Dune key is shared with other people.** Label every sandbox `owner=<us>` and `run=<run name>` (the
   launchers do). Delete a sandbox only if it carries both our owner label and one of our run labels; count by
   label before any cleanup; an unlabelled sandbox is not ours. Never delete by state, age or "everything the
   key can see".
2. **Never print secrets.** Key files are sourced, not read. Nothing key-shaped goes into logs, commits, W&B or
   Hugging Face.
3. **Kill by PID.** No `pkill -f` / `pgrep -f` patterns. If the machine runs a managed Ray cluster, never
   `ray stop` it. The driver ignores SIGINT when launched in the background: stop a run with `kill -TERM <driver
   python pid>`, and SIGKILL only if it lingers after the GPUs are free.
4. **No relay stress test** until the user says so (it shares the provider's request limits with live runs).
5. **Commit or push only when asked**, as `Charlie Ruan <53290280+CharlieFRuan@users.noreply.github.com>`.
6. **Terminal-Bench traces are never public** (benchmark canary). Private Hugging Face repos only, with
   key-shaped strings redacted.
7. **No RL compute wasted on infra problems.** After every launch, check at +10, +20 and +30 minutes that trials
   finish with sane rewards (a relaunch once burned 38 minutes on a refused key because it was checked once).
   Upload every 25th checkpoint off the node as soon as it is written.
8. **Write findings down** as you go, with clock-read timestamps: a dated `PLAN.md` log and a `FINDINGS.md` table
   per project folder (`ops/calibforge_rl_1110/FINDINGS.md` shows the format).
9. Report every eval number with its noise (section 8) and, when reporting evals, remind the user to look at
   max_tokens truncation in SkyCap (capped turns; `ops/calibforge_rl_1110/capped_turns.py`, `eval_caps.py`).


## 2. Where things are

| What | Where |
|---|---|
| This brief | `github.com/CharlieFRuan/SkyRL` (public fork), branch `charlie/calibforge-rl-1115`, `handoff/BRIEF.md` |
| SkyRL code with the SkyCap / Harbor integration used here | the same branch: upstream SkyRL `f854d6d9` + five `[harbor_skycap]` commits. Everything the runs use is in the tree: the sandbox relay (`examples/train_integrations/harbor_skycap/sandbox_relay/`), the in-sandbox Terminus agent, the SkyCap package (`skycap/`), the fully async entrypoint and launcher. Harbor is pinned by commit in `pyproject.toml`. Do not start from a plain upstream checkout: several of these files exist only here |
| Ops scripts and findings of the Qwen runs | `handoff/ops/calibforge_rl_1110/` (`FINDINGS.md`, `run_rl.sh`, `setup_data.sh`, monitoring and eval scripts). The scripts assume the layout `$D/SkyRL` (this checkout) and `$D/ops` (copy `handoff/ops` there); node addresses are replaced by `<node>` |
| Earlier Nemotron notes | `handoff/ops/RUN_NOTES.md`, `handoff/ops/upstream_mismatch/REPORT.md`, `handoff/ops/evals_1104/{PLAN,RESULTS,SETTINGS_RESEARCH}.md` |
| W&B | entity `sky-posttraining-uc-berkeley` at `api.wandb.ai`, project `skycap-calibforge-fully-async` |

Integration files (all under `examples/train_integrations/harbor_skycap/`): `harbor_generator.py` (trial loop,
retries, outage gate, stall rule), `terminus_installed.py` + `terminus_runner.py` (Terminus 2 running inside the
sandbox), `sandbox_relay/` (model calls from the sandbox to the cluster: `exec` transport for Dune),
`outage_guard.py`, `loop_offload.py`, `timing.py`, `compose.py`, `run_fully_async_terminus.sh` (the launcher),
`entrypoints/main_harbor_skycap_fully_async.py`.

Data (rebuilt by `ops/calibforge_rl_1110/setup_data.sh`; `$D` is the node-local work directory):

- `calibforge_v2_train48`: CalibForge v2, 2,788 tasks, every sandbox 4 vCPU / 8 GiB / 20 GiB, each task's own
  prebuilt image `aweaiteam/calibforge:<task>`. Source: split `calibforge_clean_v2` of the Hugging Face dataset
  `Fzz1/tb-full-data` (`data/tasks-calibforge_clean_v2.tar`, 471 MB; `ops/rdl.py` downloads it).
- `tbench21_4cpu`: Terminal-Bench 2.1 (commit `7131e43`), 89 tasks at 4 vCPU / 8 GiB / 20 GiB, plus two patches:
  `fix_qemu_apt.sh` (two Debian-bullseye tasks whose tests could not install curl) and `fix_no_ctrf.py`.

### Keys the launchers expect

Two files in `$D`, mode 600, sourced by `run_rl.sh` and never printed:

| File | Variables | Used for |
|---|---|---|
| `$D/dune_keys.env` | `DAYTONA_API_KEY`, `DAYTONA_API_URL` | every sandbox (Dune speaks Daytona's API at its own URL). Required |
| `$D/keys.env` | `WANDB_API_KEY` | logging to the W&B entity above. Required |
| | `HF_API_KEY` (exported as `HF_TOKEN`) | the Nemotron weights and tokenizer, the CalibForge task tarball (the token's account needs access to that dataset), private uploads of traces. Required |
| | `GH_TOKEN` | only if the agent is to push; this branch and Harbor are public |

Not needed: any other Daytona key (the two Daytona orgs of the Qwen runs are not used here). Still to decide on
the new machine: **where every 25th checkpoint is uploaded**. `ckpt_backup_watch.sh` / `upload_ckpt_s3.py` used
the source cluster's own storage role and bucket variable, which do not exist elsewhere; give the agent a bucket
with credentials, or a private Hugging Face repo, before the full run starts.

## 3. The reference run to replicate

`qwen35-cf-v2-insb-3600-cap16k-t1-b32-lr2e-6-1113` (W&B `hwzf6cq7`), Qwen3.5-9B on Dune. Launcher:
`ops/calibforge_rl_1110/run_rl.sh` → `run_fully_async_terminus.sh`.

| Setting | Value |
|---|---|
| Algorithm | GRPO advantages, `policy_loss_type=dppo` with delta 0.15 / 0.15, `loss_reduction=prompt_mean`, no std normalisation, no KL, zero-variance filter on |
| Batch | 32 tasks x 16 trajectories (`train_batch_size = policy_mini_batch_size = 32`, `n_samples_per_prompt = 16`) |
| Optimizer | LR 2e-6, betas [0.9, 0.98], weight decay 0.01, grad clip 1.0, full fine-tuning |
| Async | `fully_async`, `max_staleness_steps=3`, 128 generation workers (= 32 x 4), `sample_full_batch=true` |
| Generation limits | `rate_limit.max_concurrency=1600` trials, 3 trial starts per second |
| Sampling | train and eval: temperature 1.0, top_p 1.0, top_k off |
| Context | engines serve 262,144; training trajectories held to 131,072; one reply capped at 16,384 tokens (`skycap.max_tokens_per_turn`) |
| Agent | Terminus 2 **inside the sandbox** (`terminus_installed:TerminusInstalled`), thinking kept, **no summarization**, 250 turns max, agent timeout 3,600 s, trial backstop 6,600 s |
| What trains as reward 0 | agent timeouts and out-of-context endings (`train_on_timeouts=true`); errors are masked |
| Eval | TB2.1 x 3 before training and every 25 steps, same sampling as training |
| Layout | 32 GPUs: trainer on 1 node (Megatron TP 2), 12 vLLM engines x TP 2 on 3 nodes, NCCL weight sync |

Results: step 0 **0.2247** (pass@3 0.337), step 25 **0.2772** (0.382). A provider outage ruined it after step 35.
The batch-16 / LR 1e-6 run before it went 0.2135 → 0.2434 → 0.2697 → 0.2959 → 0.3258 → 0.3146 at steps
0 / 25 / 50 / 75 / 100 / 125. On a different provider the same batch-32 recipe gave 0.2809 at step 25 against
base evals of 0.2547 and 0.2172.


---

# Part 1. Nemotron-3.5-Lightning on CalibForge, without R3

Model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (hybrid Mamba2 / attention / MoE, 52 layers, 128
routed experts with 6 active, ~3B active parameters, sigmoid router with an fp32 expert bias, 262k context used
here). Served name `nemotron-3.5-lightning`.

**There is no Terminus-2 training recipe for Nemotron yet.** Every earlier Nemotron RL run here used
mini-swe-agent on SWE-style tasks in an older fork; `run_fully_async_terminus.sh` is written for Qwen3.5. 1.2
lists what has to change.

## 1.1 Hardware layout

32 GPUs (4 x 8, H200 or B200): **1 node trains, 3 nodes serve.**

- Trainer: Megatron **TP 4, EP 8** on the 8 GPUs (TP 1 runs out of memory), CPU optimizer offload, full
  recompute. This is the layout of the 218-step October run.
- Inference: 24 GPUs as **12 engines x TP 2** (what the October run used; weight sync over NCCL took ~4.7 s) or
  6 engines x TP 4. Start with TP 2: it gives twice the engines for routing and the model fits; measure tokens
  per second at the target concurrency before choosing TP 4.
- `trainer.placement.colocate_all=false`, `policy_num_nodes=1`.

## 1.2 Port the launcher

Make a Nemotron variant of `run_fully_async_terminus.sh` (env-switched, as `ops/run_fully_async.sh` did with
`MODEL_FAMILY`), with these differences from the Qwen settings:

- remove `language_model_only=true` (trainer, ref and engine) and `engine_init_kwargs.gdn_prefill_backend`;
- `megatron_config.tensor_model_parallel_size=4`, `expert_model_parallel_size=8`;
- `megatron_config.moe_router_score_function=sigmoid`, `moe_router_enable_expert_bias=true`,
  `trainer.mtp.enabled=false`;
- `skycap.tokenizer=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (**the Hugging Face id, not a local
  path**: with a path the renderer falls back to the default one, which refuses `thinking_retention=all`),
  `skycap.renderer=nemotron-3.5`, `skycap.chat_template_kwargs.enable_thinking=true`,
  `skycap.chat_template_kwargs.truncate_history_thinking=false`, `skycap.thinking_retention=all`;
- served model name, and Terminus's `model_info` for it.

**R3 off** means both `trainer.policy.megatron_config.moe_enable_routing_replay=false` and
`generator.inference_engine.enable_return_routed_experts=false` (both default false; the fully async launcher
never set them).

**Check the router-bias fix before any training.** Weight sync used to round the fp32 router bias to bf16,
which made rollout and trainer disagree; that is what R3 was papering over. The fix is in mainline (PR #2543 and
#2618: fp32 and integer tensors keep their dtype on the wire, `weight_sync/sources.py`). Confirm the checkout has
it, then confirm on a few steps that the mean rollout-vs-trainer logprob gap is ~0.02 (0.017-0.030 was measured
without R3 after the fix; 0.045-0.07 means the fix is missing). `SKYRL_DEBUG_EXPERT_BIAS=1` helps.

Sampling: train and evaluate at temperature 1.0, top_p 1.0, top_k off, as the Qwen reference run did (the model
card recommends top_p 0.95; the user asked for temperature 1 and the same settings as our runs).

## 1.3 Base-model eval on Terminal-Bench 2.1 (our settings; explain the gap to the published number)

**User's decision (11-15): we do not have to reproduce the published number.** Ours may come out higher or
slightly lower; what is owed is our number, measured the way we will measure checkpoints, and an explanation of
the difference.

Published: the model card's Terminal-Bench 2.1 = **24.58** ("measured by NVIDIA under a consistent harness (NeMo
Gym / NeMo Evaluator SDK)"; SWE-bench Verified 51.56 in the same table). The card also has an image, "Agentic
Coding Benchmarks", with harness-level numbers that I could not read as text; our October notes recorded 29.7
for mini-swe-agent v2.4.5 from that chart. Look at the image and note which number belongs to which harness.

**How to run it** (the user's instructions: temperature 1, no summarization, no thinking stripping, through
SkyCap with the SkyRL entrypoint):

- the SkyRL training entrypoint with `eval_before_train=true` on the base model (this is how every Qwen step-0
  eval was made, and it is exactly the path later checkpoints go through), or the eval-only entrypoint
  `entrypoints/main_harbor_skycap_eval.py` with the same settings;
- sampling temperature 1.0, top_p 1.0, top_k off (the training sampling; the card recommends top_p 0.95, see the
  table below);
- **no summarization** (`enable_summarize: false`), **thinking kept in the history**
  (`truncate_history_thinking=false`, `thinking_retention=all`);
- Terminus 2 inside the sandbox through the relay, one reply capped at 16,384 tokens, 262,144 context, agent
  timeout 3,600 s, sandboxes 4 vCPU / 8 GiB / 20 GiB on Dune, TB2.1 `tbench21_4cpu` (commit `7131e43` with the
  two task patches);
- 89 tasks x 3 is one eval. **Run the base eval at least twice** (or with 6-9 repeats): it is the baseline of
  the whole RL curve and one 3-repeat eval is only good to about ± 0.03 (section 8).

Report: average and pass@3 with the spread between runs, endings (agent timeouts, out of context, runner exits,
verifier failures), capped turns (`eval_caps.py`), and whether the model loops at this sampling (Qwen3.5-9B did
at lower temperature, not at 1.0).

**Explaining the difference from 24.58.** What differs between NVIDIA's recipe as we recorded it
(`ops/evals_1104/SETTINGS_RESEARCH.md`) and ours:

| | NVIDIA's recipe | Ours |
|---|---|---|
| Harness | harbor 0.3.0, Terminus 2 with five patches, agent outside the sandbox | Harbor at the pinned commit, Terminus 2 **inside** the sandbox, model calls through SkyCap and the relay |
| Sampling | temperature 1.0, top_p 0.95 | 1.0 / 1.0 |
| Summarization | on (proactive threshold 8,000) | off: a trial that fills its context ends there |
| Reply length | no `max_tokens` | 16,384 per reply |
| Thinking | on, replayed in think tags | on, kept in history |
| Time limit | 7,200 s (`timeout_strategy: max`), unlimited turns | 3,600 s, 250 turns |
| Sandbox | 4 vCPU / 8 GiB / 100 GiB | 4 vCPU / 8 GiB / 20 GiB |
| Task set | TB2.1 commit `c5ee500c` | commit `7131e43` + two patched tasks (worth up to ~2 points) |
| Repeats | 8 | 3 per eval |

Reference points we already have for this model: **0.301 through SkyCap and 0.297 through stock Harbor** on
11-04 / 11-06 with settings close to NVIDIA's (summarization on, 7,200 s, Terminus on the cluster with three of
the five patches, 3 repeats; `ops/evals_1104/RESULTS.md`), both above 24.58, and 26.3 with mini-swe-agent in
October. So expect to be in the high 20s; if the new number differs from 0.30, the first suspects are
summarization off and the shorter time limit (out-of-context and timeout endings will show it). A second arm
that changes one thing (top_p 0.95, or summarization on) is cheap if the user wants the gap attributed.

On Dune, keep Terminal-Bench starts gentle: earlier notes record trouble above ~150 concurrent TB sandbox
starts and recommend <= 60 concurrent with 0.5-1 starts per second for standalone evals.

## 1.4 Overfit run, ~15 steps

Purpose: prove the Nemotron + Terminus + in-sandbox pipeline learns at all, and bracket the learning rate.

- Tasks: 32 CalibForge tasks the base model solves sometimes (pass rate roughly 0.2-0.8). Pick them from a
  one-pass generation scan over a few hundred tasks with Nemotron (the eval-only entrypoint), not from Qwen's
  history. Exclude tasks whose verifier or setup is flaky.
- Same recipe as the full run (1.5) except the task set repeats every step: batch 32 x 16.
- Watch per step: raw reward and pass@16, policy entropy, clip ratio, logprob gap, trajectory length, timeouts
  and out-of-context counts, grad norm.

**Learning rate.** Evidence so far:

| Model, setting | LR | What happened |
|---|---|---|
| Nemotron, overfit 16 tasks x 8, no R3, fixed sync | 5e-6 | reward 0.35 → 0.63 in 20 steps, stable, entropy 0.06-0.08 |
| Nemotron, overfit with R3 (before the fix) | 1e-6 / 3e-6 / 5e-6 | flat / learned / learned fastest, with entropy falling and rollouts growing 22k → 51k tokens |
| Nemotron, full MiMo run, batch 16 x 16 | 1e-6 | reward flat for 218 steps, entropy drifting up; the note then recommended 3e-6 |
| Qwen3.6, another cluster | 5e-6 | collapsed (entropy 0.04, 122k-token rollouts) |
| Qwen3.5-9B, CalibForge, batch 16 | 1e-6 | TB2.1 +0.11 in 100 steps |
| Qwen3.5-9B, CalibForge, batch 32 | 2e-6 | +0.05 by step 25; clip ratio and logprob gap rose 2-3x, then levelled |

On the user's question, whether an LR that overfits well should be used for the full run: **not directly.** An
overfit run sees the same 32 tasks every step, so the gradient points the same way step after step and a high
LR looks good; it also cannot show the slow failures (entropy drift, rollouts growing until they hit the
context) that the 218-step run showed. What the overfit can do is bound the LR from above: the largest LR that
is still stable there (no entropy collapse, no explosion of clip ratio or rollout length) is an upper bound, and
an LR that does not learn even there is too low. My suggestion: overfit at 2e-6 and 5e-6 (and 1e-5 if GPUs
allow a third arm), then run the full epoch at the largest stable one divided by ~2, which I expect to land at
2e-6 to 3e-6; confirm with the step-25 eval and the stability metrics in the first 25 steps. This is a
judgement, not a measured rule; put the choice to the user with the overfit curves.

## 1.5 Full run, one epoch

The reference recipe (section 3) with the Nemotron differences of 1.2 and the layout of 1.1:

- CalibForge v2 `calibforge_v2_train48` on Dune (exec relay), 2,788 tasks → 87 steps at batch 32; `EPOCHS=1`.
- Batch 32 x 16, DPPO delta 0.15, `prompt_mean`, zero-variance filter, LR from 1.4, betas [0.9, 0.98], weight
  decay 0.01, grad clip 1.0, full fine-tuning.
- `max_staleness_steps=3`, 128 workers, 3 trial starts per second, `max_concurrency` 1,600: **the async loop
  exactly as it is**, because this run is the baseline for Part 2. Record its throughput from what is already
  logged: `timing/step`, `timing/policy_train`, `timing/wait_for_generation_buffer`,
  `async/gen_buffer_qsize_at_wait_start`, `async/num_groups_dropped`, `generate/group_completion_time_*`,
  `generate/trajectory_time_completion_mean`, and trials in flight from `<run>/loop_lag.jsonl`
  (`handoff/ops/calibforge_rl_1110/wandb_speed.py <W&B run id>` prints the W&B ones).
- Train sampling per 1.2; context 262,144 served, 131,072 trained (check what fits the trainer at TP 4 / EP 8);
  16,384-token reply cap; no summarization; thinking kept; agent timeout 3,600 s.
- TB2.1 x 3 before training and every 25 steps at the settings of 1.3; more repeats if the budget allows,
  given the noise.
- Checkpoints every 5 steps locally, every 25th uploaded off the node at once (`ckpt_backup_watch.sh`). A
  checkpoint used for a resume or an eval must exist at the same path on every node: the trainer can be placed
  on any of them.
- Guards on by default in `run_rl.sh`: `SKYCAP_OUTAGE_GUARD=1` (a provider outage pauses trial starts instead
  of training on empty batches), `SKYCAP_STALL_TIMEOUT_SEC=900` (an agent timeout caused by one stuck model
  call is an error, not a reward-0 rollout), `SKYCAP_VERIFIER_RETRIES=3`, tar offload, timing.

Stop and ask the user if: the step-25 eval is below the step-0 eval by more than the noise, entropy collapses or
rollout length runs away, or more than a few percent of trajectories per step are errors.


---

# Part 2 (later; not before the Nemotron run). Throughput of the fully async loop

**Status: deferred by the user on 11-15.** "If we want to optimize the fully async stuff, we should have an
un-optimized baseline first." The Nemotron run of Part 1 is that baseline. Start here only when the user says
so; until then the only allowed step is reading the baseline numbers listed in 1.5.

## 2.1 What limits trials in flight today: reading of the code

Notation: B = groups per batch (32), S = `max_staleness_steps` (3), W = generation workers (128), G = group size
(16). All of this is `skyrl/train/fully_async_trainer.py` unless noted.

- **Unit of everything is the group** (one task x 16 trials). W asyncio workers each run one group at a time
  (`_run_generate_for_a_group_loop`, ~l. 952-1028): take a prompt, `acquire_submission_slot()`, `await
  generator.generate()` for the whole group, put it in the buffer.
- **Staleness window** (`_compute_capacity_unlocked`, ~l. 182-188): a worker may start a group only while
  `running + accepted-but-not-yet-trained <= (S + current_step) * B`, i.e.
  **running + buffered + in the batch being trained <= (S+1)·B = 128 groups**. A finished group keeps its place
  in the window while it waits in the buffer and while it is trained. The constructor enforces
  `B <= W <= B·(S+1)`, so W cannot be raised without raising S or B.
- **Capacity comes back once per step, in one burst**: `notify_capacity_change` (~l. 700) runs after collect,
  convert, training, weight sync, eval, checkpoint and the W&B log. It frees B groups = 512 trials at once.
- **Zero-variance groups** (dropped by the filter) free their place only when the consumer drains them from the
  buffer at the start of the next step (`on_rollout_filtered`, ~l. 895), not when they finish.
- **A group returns only when all 16 trials have ended** (`harbor_generator.py` `run_trials`, an
  `asyncio.TaskGroup`). Finished trials of a group that is still running free nothing.
- **Rate limiter and concurrency cap** (`skyrl/train/utils/rate_limiter.py`, applied per trial attempt at
  `harbor_generator.py` ~l. 436) count trials: a token bucket at `trajectories_per_second` first, then a
  semaphore of `max_concurrency`. The 512 trials released at each step need 512 / 3 = 170 s just to get tokens.
- **Groups with no live trajectory are kept** (`trainer_utils.py` zero-variance filter keeps groups with <= 1 live
  trajectory): they take a batch slot and train nothing.
- **One process, one event loop** runs the trainer coroutine, the generator and every Harbor trial (Ray task
  `skyrl_entrypoint`). Model calls of in-sandbox agents do not pass through it (the relay managers live in the
  SkyCap server actors), but sandbox start, the runner's long-poll, uploads, downloads and the verifier do.
- **No overlap across epochs**: generation drains to zero at each epoch boundary.

## 2.2 What we measured (Qwen3.5-9B, batch 32, 28 clean steps of W&B run `c0gc5h63`; the Dune run `hwzf6cq7` agrees)

| Quantity | Value |
|---|---|
| Step time | mean 1,030 s (min 677, max 1,769) |
| of which policy update (`timing/policy_train`) | mean 805 s |
| of which waiting for the buffer (`timing/wait_for_generation_buffer`) | mean 201 s, median 121 s, min 0.1 s, max 647 s |
| Groups already in the buffer when a step starts (`async/gen_buffer_qsize_at_wait_start`) | mean 42 |
| Groups consumed per step | 32 kept + 19 dropped by the zero-variance filter (keep rate 0.52-0.74) |
| Trial completion time | mean 963 s (agent part 673 s, p90 978 s, 0.7% hit the 3,600 s limit) |
| **Group** completion time | mean 1,610 s, p90 2,970 s, max ~4,800 s |
| Trials in flight (from `loop_lag.jsonl`, not in W&B) | mean ~580 (Dune ~640), 900 at the cap for a few minutes after each step, 110-280 during an update |
| Staleness of trained groups | mean 1.9 steps, max 4-12 |
| Event-loop lag | max 2-8 s per hour since the tar offload (below) |

**My reading, to be confirmed by 2.3 / 2.4, not yet proven:** the loop is limited from both sides and the two
interfere.

- The trainer alone would allow a step every ~825 s. Generation alone, with all 128 window places running,
  would deliver the ~51 groups a step needs every ~640 s (128 / 1,610 s per group).
- Together they take 1,030 s, because finished groups pile up in the buffer during the 805 s update (42 of the
  128 places by the time the update ends), which stops new groups from starting; the trainer then waits ~200 s
  for the missing ones.
- Trials in flight ≈ running groups x 16 x (trial time / group time) ≈ 60 x 16 x 0.6 ≈ 580: **a third or more of
  the trial slots of running groups are already finished and waiting for their group's slowest trial.** The
  concurrency cap (900-1,600) binds only in the minutes after each burst.
- 37% of generated groups are dropped as zero-variance: over a third of generation is spent on groups that are
  never trained.

The user is not convinced it is only the staleness window ("there's a world where there are just bugs"), and is
right to want proof: an earlier version of this loop was throttled by a real bug. Harbor tarred and gzipped
agent logs synchronously on the event loop, freezing every trial for up to 376 s and holding trials in flight
near 700 of 1,024. That one is fixed from our side (`loop_offload.py`: `SKYCAP_OFFLOAD_TAR=1`,
`SKYCAP_SKIP_LOG_REUPLOAD=1`, both set by `run_rl.sh`), but nothing guarantees it was the only one.

## 2.3 Step 1: make it visible in W&B

Goal: per training step, W&B shows how many trials and groups were in each state and what was blocking the
workers. The user suggested a thread in the rate limiter that logs periodically; a sampler is the right idea, but
do not call `tracker.log` from it: W&B rows are keyed by `step=self.global_step` and committed once per step
(~l. 677), so a second writer either loses data or races the commit. Follow the existing GPU monitor
(`skyrl/train/utils/ray_gpu_monitor.py`): sample in the background, aggregate, merge into the step's
`log_payload` (~l. 671-675).

1. **Sampler task** created next to the generator tasks in `train()` (~l. 522-537), cancelled with them. Every
   2-5 s it records, and per step it logs mean / min / max / p50 and the time-weighted share:
   - from the staleness manager (`_stat`): groups running, accepted (finished, not yet trained), buffer size,
     both terms of `_compute_capacity_unlocked` (staleness capacity, worker capacity);
   - from the generator (add counters and a `stats()` method to `HarborSkycapGenerator`): trials waiting at the
     outage gate, waiting for a rate token, waiting for the concurrency semaphore, inside `trial.run()` split by
     phase if cheap (environment start, agent, verifier), finishing, and **finished while their group is still
     running**;
   - from `AsyncRateLimiter`: waiters on the token bucket and on the semaphore, starts per second;
   - its own sleep overshoot as event-loop lag (p50 / p99 / max), so lag is in W&B and not only in
     `loop_lag.jsonl`;
   - the trainer phase at the time of each sample, so every metric can be split into "during the update" and
     "while waiting for the buffer".
   Suggested names: `async/inflight_trials_{mean,min,max}`, `async/running_groups_mean`,
   `async/buffered_groups_mean`, `async/done_trials_in_running_groups_mean`, `async/loop_lag_p99`.
2. **Blocked-time counters**, in worker-seconds per step, which add up to W x step time:
   - blocked on the staleness window (`acquire_submission_slot`, staleness capacity <= 0);
   - blocked on the worker limit (worker capacity <= 0);
   - generating;
   - idle because the dataloader is exhausted (epoch end).
   And in trial-seconds: waiting for a rate token, waiting for the semaphore, waiting at the outage gate.
3. **Wall time of the synchronous pieces on the loop**, per step: `compose()` per group
   (`harbor_generator.py` ~l. 294-303), SkyCap `finish` JSON decoding (`skycap/client.py` ~l. 276, several MB per
   long trajectory), `_should_keep_group`, advantage computation, `tracker.log`.
4. Also publish the live values through the existing Prometheus gauges (`skyrl/train/utils/metrics.py`) if a
   Grafana is at hand.

Test the instrumentation without GPUs with the simulated trainer (`fully_async_trainer_sim.py`,
`trainer.fully_async.simulate_training`) before a real run. Check its overhead: the sampler must not itself add
loop lag.

## 2.4 Step 2: prove what limits it

One short real run (the Part 1 recipe, ~8 steps after the first fill is enough) with the new metrics answers
each hypothesis. Decide in advance what each would look like:

| Hypothesis | It is the limit if | It is not if |
|---|---|---|
| Staleness window | workers spend most blocked time on staleness capacity; running + buffered + training sits at 128 groups; in flight drops during updates and recovers after each burst | staleness capacity is usually > 0 while in flight is low |
| Group tail | `done_trials_in_running_groups` is a large share of the slots of running groups (expected ~40%) | it is small |
| Trainer too slow | buffer wait ≈ 0 and the buffer is full at step start | the trainer waits for the buffer most steps |
| Rate limiter | trials queue for tokens for long stretches (expected only ~170 s after each burst) | token wait ≈ 0 |
| Concurrency cap | semaphore waiters > 0 for a large share of the step | in flight rarely reaches the cap |
| Event-loop lag / a bug on the loop | lag p99 in seconds, or in-flight dips that line up with lag spikes or with one synchronous piece from 2.3 item 3 | lag stays under ~1 s |
| Slow sandbox operations | environment start, upload, download or verifier phases grow with concurrency | phase times are flat as concurrency rises |
| Provider | outage-gate waits, creation failures, rate-limit errors in the relay stats | none |

Two cheap controlled experiments separate "window" from "bug" directly:

1. **Simulated trainer** with a fixed fake update time and real generation: set the fake update to 0 s, then to
   800 s. If the window explains the numbers, in flight with a 0 s update should rise to roughly
   W x G x (trial time / group time) and the staleness-blocked time should vanish; if in flight stays near 580,
   something else is holding it.
2. **Generation-only pass** (the eval-only entrypoint `ops/timeout_repro_1110/run_infer.sh`, no window at all)
   over ~1,000 tasks at the same cap: our two full passes of this kind (2,600 tasks, one trial each) held 900
   in flight until the tasks ran out, which already suggests the trial machinery itself can sustain the cap
   (loop lag was not recorded for them). Repeat it at the
   concurrency you intend to use (e.g. 1,600 and 2,400) to find where sandbox operations or the loop start to
   degrade.

Keep `py-spy` dumps of the driver every 15 s during these runs (`ops/calibforge_rl_1110/pyspy_watch.sh`); a
stack at the moment of a lag spike is what found the tar bug.

## 2.5 Step 3: raise it

Apply in this order, one change at a time, each judged on the 2.3 metrics and on wall-clock per trained group.
Stop when the trainer is the only limit, then speed up the trainer.

1. **Free window places as early as is correct.**
   - Decide a group's fate when it finishes, not when the consumer drains it: run the zero-variance check at
     completion, and release the place of a dropped group immediately (they are 37% of groups).
   - Release the batch's capacity when the batch is collected (or when training starts) rather than after
     training, sync, eval, checkpoint and logging. This changes what "staleness" counts; say so in the metric.
   - Release in smaller pieces rather than one burst of B groups, or raise the token rate so a burst does not
     queue for minutes.
2. **Stop waiting for the slowest trial of a group** (the group-granularity question). Options, least invasive
   first:
   - return a group once k of G trials have ended and a deadline has passed, masking the rest (the masked-trial
     path already exists for errors); check what it does to the reward estimate, since slow trials are mostly
     failures and timeouts, so dropping them biases the group mean upward unless they are counted as reward 0;
   - a shorter effective tail: 0.7% of trials run to the 3,600 s limit but they set the group time for ~11% of
     groups;
   - accounting and acceptance per trajectory instead of per group, with groups reassembled at training time.
     This is the larger change the user has in mind; `docs/content/docs/tutorials/fully_async.mdx` states the
     group is the smallest unit today and nothing in the trainer supports a partial group. Propose a design to
     the user before building it, with the measured gain from the cheaper options next to it.
3. **Do not train on or wait for dead groups**: drop (better, regenerate) a group with no live trajectory, and
   consider detecting an all-pass or all-fail group early.
4. **Widen the window** (`max_staleness_steps` 3 → 4-6, with W raised to match) only with the off-policy metrics
   in view: `policy/loss_metrics/clip_ratio` (was 3e-6 to 1e-5), the rollout-vs-trainer logprob gap (was ~0.01)
   and `async/staleness_*`. More staleness is the one change here that can cost learning quality.
5. **Faster policy update** (805 s for 512 trajectories of ~65k tokens on 8 GPUs at TP 2). With more GPUs: more
   trainer nodes (`POLICY_NUM_NODES`), and check recompute settings and packing. This moves the other limit, so
   re-measure after each generation-side change.
6. **Then** raise `max_concurrency` and the token rate toward what the sandbox provider and the relay sustain.
   Known provider facts: on Dune the exec relay costs ~13-16 provider requests per sandbox-minute; no rate limit
   was seen up to ~890 sandboxes; its limit above that is unknown (and the stress test is not allowed yet).
7. Smaller items: generation drains at epoch boundaries; one retried eval trial can hold 32 GPUs idle for an
   hour while 32 GPUs wait; evals block training.

Acceptance for Part 2: (a) the W&B panel exists and its numbers add up (worker-seconds sum to W x step time);
(b) a written answer to "what limits it", with the table above filled in from data; (c) a measured before/after
for each change kept, in trained groups per hour and in-flight trials, with clip ratio and logprob gap unchanged
or explained; (d) nothing merged that changes what is trained on without the user's agreement.


---

## 6. Pitfalls already paid for

- **Provider outages.** A two-hour Dune outage trained ~15 empty steps before the outage guard existed; later the
  key was refused (HTTP 403) for a day. The guard covers 5xx, connection errors, rate limits and 401 / 403.
- **Stuck model calls trained as failures.** On a different provider's preview relay, lost poll responses left
  114 of 301 "agent timeouts" waiting ~48 minutes on one call. Dune's exec transport replays, but keep the stall
  rule on and look at `relay_stats.py` (`rate_limited`, `recovered`, `duplicates`, error types) every hour.
- **Eval rot and false zeros.** Graders that download tools at test time give zeros during network trouble;
  about 20 training tasks never write the result file the retry check looked for (`fix_no_ctrf.py`); two TB2.1
  tasks could never pass without `fix_qemu_apt.sh`.
- **Looping at low temperature.** Qwen3.5-9B at 0.6 / 0.95 / top_k 20 loops inside its thinking and fills the
  context in 40% of TB2.1 trials; at 1.0 it does not. Check Nemotron for the same before trusting any eval at a
  lower temperature, and count capped turns in every eval.
- **In-sandbox agent install needs the network** per trial (uv, Python, wheels); a ten-minute blip failed 213
  trials once. Images with neither curl nor wget cannot start the relay stub without the fallback.
- **Checkpoint rotation deleted the best checkpoint** in the October run. Keep every eval-step checkpoint.
- **A library release broke every trial mid-run** (litellm, October). Pin the agent's dependencies.
- **Megatron checkpoints are not Hugging Face weights.** `ops/convert_ckpt_to_hf.sh` converts Nemotron
  checkpoints (CPU, ~45 s, needs the base model directory); checkpoint evals can also go through the training
  entrypoint with a resume path and `eval_before_train`.
- **Counting steps in the log**: the number of `Finished: 'step'` lines is the global step, or the step + 1 when
  an eval ran before training.

## 7. What to hand back

1. Part 1: the base-model eval (at least two runs, with the spread) and the explanation of its distance from
   24.58; the overfit curves for each LR with a recommendation; the full-run curve with evals at 0 / 25 / 50 / 75
   and where the checkpoints are; the baseline throughput numbers of 1.5.
2. Part 2, when the user starts it: the W&B panel, the filled-in limits table, before/after numbers for each
   change against the Part 1 baseline, the diff.
3. A dated `PLAN.md` and a `FINDINGS.md` table for the new work.

## 8. Eval noise (read before comparing any two evals)

Two TB2.1 x 3 evals of the same base Qwen model on the same provider gave **0.2547 and 0.2172**. It was not
infrastructure: the lower run had fewer timeouts, and the tasks that flipped ended with the agent finishing and
the tests failing. 31 of the 89 tasks are solved only sometimes, so with 3 draws per task the standard deviation
of one eval is about 0.017 and of a difference of two about 0.024. A gain under ~0.05 from single 3-repeat evals
is not established. Use more repeats (9 per task brings the difference to ~0.014), compare per task, and pool
neighbouring checkpoints when judging a trend.
