# Step-wise Training Picks on `penfever/working`

This branch (`charlieruan/penfever-stepwise-picks`) starts from
[`penfever/working`](https://github.com/penfever/SkyRL/tree/penfever/working)
and ports the step-wise training changes tracked in
[NovaSky-AI/SkyRL#1278](https://github.com/NovaSky-AI/SkyRL/issues/1278)
so Harbor + Terminal-Bench training runs correctly in step-wise mode
on the penfever codebase.

`penfever/working` diverged from NovaSky-AI/SkyRL `main` at an old
skyrl-tx benchmarks commit (~393 commits behind main) and still uses the
pre-rename `skyrl-train/skyrl_train/` package layout. The modern step-wise
PRs are all on the post-rename `skyrl/` tree, so these picks are
**reimplementations of each PR's intent on the old layout**, not literal
`git cherry-pick`s.

## Quick map: upstream PR → file(s) in this branch

| Upstream PR | Title | Status | Files in this branch |
|---|---|---|---|
| [#1281](https://github.com/NovaSky-AI/SkyRL/pull/1281) | step-wise GeneratorOutput validation | ✅ ported | `skyrl-train/skyrl_train/utils/trainer_utils.py`, `skyrl-train/skyrl_train/trainer.py` |
| [#1285](https://github.com/NovaSky-AI/SkyRL/pull/1285) | unified left-pad + right-aligned response tensors | ✅ ported + validated | `skyrl-train/skyrl_train/dataset/preprocess.py`, `skyrl-train/skyrl_train/trainer.py` |
| [#1507](https://github.com/NovaSky-AI/SkyRL/pull/1507) | step-wise advantage broadcast with per-step response mask | ✅ ported | `skyrl-train/skyrl_train/trainer.py::compute_advantages_and_returns` |
| [#1529](https://github.com/NovaSky-AI/SkyRL/pull/1529) | prompt-based mini-batching for step-wise | ≈ partial (by user agreement) | `skyrl-train/skyrl_train/utils/utils.py` (batch-size guard), `skyrl-train/skyrl_train/trainer.py::_remove_tail_data` |
| [#1536](https://github.com/NovaSky-AI/SkyRL/pull/1536) | plumb step-wise through fully-async trainer | ✅ ported + validated | `skyrl-train/skyrl_train/fully_async_trainer.py`, `skyrl-train/skyrl_train/generators/utils.py::concatenate_generator_outputs` |
| [#1538](https://github.com/NovaSky-AI/SkyRL/pull/1538) | prefix-aware merging for step-wise | ✅ ported + validated | `skyrl-train/skyrl_train/generators/utils.py`, `skyrl-train/skyrl_train/trainer.py::postprocess_generator_output`, `skyrl-train/skyrl_train/config/ppo_base_config.yaml` |
| [#1542](https://github.com/NovaSky-AI/SkyRL/pull/1542) | Harbor → step-wise training | ✅ ported (core) | `skyrl-train/examples/terminal_bench/terminal_bench_generator.py`, `skyrl-train/examples/terminal_bench/entrypoints/main_tbench.py` |

Plus:
- **`setup_otagent.md` §3 env-var patch** — propagate `DAYTONA_API_KEY`,
  `DAYTONA_API_URL`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `HF_TOKEN`,
  `HF_HOME`, `HF_HUB_CACHE` to Ray workers in
  `skyrl-train/skyrl_train/utils/utils.py::prepare_runtime_environment`.

---

## Per-PR detail

### PR #1281 — step-wise `GeneratorOutput` validation

**Bug:** `validate_generator_output()` was skipped entirely when
`step_wise_trajectories=True`, so malformed `is_last_step` or
non-contiguous `trajectory_ids` would silently corrupt the cumsum-based
advantage broadcast.

**Port:**
- Added `step_wise: bool = False` kwarg to `validate_generator_output`
  in `trainer_utils.py`. When on, skips the `num_prompts ==
  num_responses` assert (step-wise emits one entry per turn, not per
  prompt) and calls a new `_validate_step_wise_fields` helper that
  checks:
  - `is_last_step` and `trajectory_ids` present and length-matched.
  - `is_last_step[-1]` is True.
  - `trajectory_ids` are contiguous (no reappearance after a gap).
  - `is_last_step[i]` is True iff trajectory id changes at `i+1` (or
    `i` is last).
- `trainer.py::generate` now always calls `validate_generator_output`
  with `step_wise=self.cfg.trainer.step_wise_training` (no more skip).

### PR #1285 — unified left-pad + right-aligned response tensors

**Bug:** The old two-segment `[PAD prompt][response PAD]` layout padded
each row to `max_input_len + max_output_len`. In step-wise training,
prompts grow and responses shrink turn-over-turn, so this inflates
sequences to nearly `2 × max_seq_len`. **Critical for scale.**

**Port:** `convert_prompts_responses_to_batch_tensors` now produces:

```
| [PAD] [PAD] prompt prompt prompt respon respon |
| [PAD] prompt prompt prompt respon respon respon |
| prompt prompt prompt respon respon respon respon |
                        |<---- max_response_len ---->|
```

- Padded sequence length is `max(prompt_i + response_i)` (tight bound).
- Response-level tensors (`action_mask`, `rewards`, `loss_masks`,
  `logprobs`) are **right-aligned** within `(batch, max_response_len)`
  so they match `log_probs[:, -num_actions-1:-1]` slicing in the model
  forward.
- New optional `max_seq_len` kwarg emits a warning (no truncation) when
  the tight bound exceeds it.
- Trainer logs `generate/batch_num_seq` and
  `generate/batch_padded_seq_len` so padding efficiency is trackable.

**Validated:** colocated step 1 ran end-to-end at 987s,
`policy_loss=9.9e-6`, `grad_norm=0.29`, batch shape metrics logged.

### PR #1507 — advantage broadcast with per-step response mask

**Bug:** In step-wise mode, `advantages = last_step_advantages[traj_ids]`
copied the last step's already-masked advantage tensor to every earlier
step, so the advantage landed at the *last* step's non-padding positions
instead of each step's own. ~(N-1)/N of step-samples trained with
zeroed advantages when turns had different response lengths.

**Port** (`trainer.py::compute_advantages_and_returns` step-wise branch):
```python
# Use all-ones mask so GRPO broadcasts the scalar to every position…
last_step_advantages, last_step_returns = ppo_utils.compute_advantages_and_returns(
    token_level_rewards=token_level_rewards[is_last_step],
    response_mask=torch.ones_like(last_step_response_mask, dtype=torch.float),
    …
)
# …then re-apply each step's own response_mask after broadcast.
response_mask_float = response_mask.to(last_step_advantages.dtype)
advantages = last_step_advantages[traj_ids] * response_mask_float
returns   = last_step_returns[traj_ids]   * response_mask_float
```

### PR #1529 — prompt-based mini-batching (partial)

User explicitly accepted: *"If otherwise requires a lot of code
changes, I am fine with doing multiple gradient updates for mini_batch_size
of prompts."* So this port is the **small-scale-friendly subset**:

- Relaxed `validate_batch_sizes` in `utils/utils.py`: the sharding
  constraint is on the number of *sequences*, not prompts, so replaced
  `train_batch_size >= lcm_dp_size` with `train_batch_size *
  n_samples_per_prompt >= lcm_dp_size`. Otherwise 4-prompt runs on 8 DP
  GPUs fail validation.
- Fixed `trainer.py::_remove_tail_data` to truncate on the GCD-based
  prompt shard (`dp_size / gcd(dp_size, n_samples_per_prompt)`). The old
  `(len // dp_size) * dp_size` formula zeroed the batch when
  `train_batch_size < dp_size` — 4 prompts on 8 DP → empty batch.
  Small-batch runs now keep the whole batch (pad_batch handles residual).

Full prompt-based mini-batch boundary threading through
`MeshDispatch.stage_chunks` / `WorkerDispatch.stage_data` was not
ported; training just does multiple optimizer updates per step, which
the user accepted.

### PR #1536 — plumb step-wise through fully-async trainer

**Bug:** `FullyAsyncRayPPOTrainer.convert_generation_group_mini_batch_to_training_input`
assumed a constant `group_size = len(cur_generation_group_mini_batch[0].generator_output["response_ids"])`,
broken when step-wise makes each group have a variable number of
per-turn entries. Also `concatenate_generator_outputs` couldn't signal
step-wise validation.

**Port:**
- `fully_async_trainer.py`: compute `group_size` per-group inside the
  loop. Log `effective_batch=total_kept_samples` (sum across groups,
  not `kept_groups * group_size`). Pass
  `step_wise=cfg.trainer.step_wise_training` to `concatenate_generator_outputs`.
- `generators/utils.py::concatenate_generator_outputs`: new
  `step_wise: bool = False` kwarg forwarded to
  `validate_generator_output`.

**Validated:** fully-async step 1 ran end-to-end at 492.50s
(wait_for_generation_buffer=464s — buffer waiting, not compute;
convert=0.08s, fwd_logprobs=8.5s, policy_train=11.32s, sync=8.47s).
`effective_batch=35 samples (4/4 groups)` — **variable per-group sizes
visible in the log**, exactly what #1536 fixes. avg_final_rewards=0.3125
(5/16 solved). GPU partition worked: GPUs 0-3 = FSDP policy+ref (63GB),
GPUs 4-7 = vLLM inference (20GB).

### PR #1538 — prefix-aware merging

**Motivation:** Step-wise O(T²) training cost. When consecutive turns in
the same trajectory share a token-exact prefix, collapse them into a
single sequence with obs-delta tokens loss-masked out → cost becomes
O(T).

**Port:**
- Added `_is_prefix`, `_slice_generator_output`,
  `_merge_single_trajectory`, and `merge_stepwise_output` to
  `generators/utils.py`. Greedy merge: if `acc_prompt + acc_response`
  is a prefix of the next turn's prompt, extend the accumulator with
  the obs-delta (tokens in `prompt[i]` past the prefix, loss_mask=0 /
  reward=0 / logprob=0) and then the new turn's response +
  loss_mask + logprobs. When prefix check fails, flush and start a
  new group. Per-turn fields (`stop_reason`, `is_last_step`,
  `trajectory_id`) take the last turn's value.
- Keyed `TrajectoryID` comparisons by `to_string()` since penfever's
  dataclass is not frozen (not hashable).
- New config flag `generator.merge_stepwise_output` (default false) in
  `ppo_base_config.yaml`.
- `trainer.py::postprocess_generator_output` runs merge before
  metrics, logs `generate/num_seq_{before,after}_merge`, rebuilds
  `uids` from the merged `trajectory_ids` (instance_id groups the GRPO
  baseline). **Signature now returns `Tuple[GeneratorOutput,
  List[str]]`**; both call sites (sync + fully-async) updated to
  unpack.

**Validated:** both colocated (46→46) and fully-async (35→35) runs
executed the merge path without error. Zero merges triggered on the
terminus-2 agent because its chat-history re-formatting breaks the
token-exact prefix property — **function works; merge isn't beneficial
for this particular agent**. For other agents that don't re-tokenize
mid-conversation, merges will kick in.

### PR #1542 — Harbor → step-wise training (core)

**Motivation:** The existing `TerminalBenchGenerator` collapsed every
trajectory's multi-turn conversation into a single `(prompt, response)`
pair. For step-wise training we need one `GeneratorOutput` row per
agent turn with `is_last_step` + `trajectory_ids`.

**Port** (`examples/terminal_bench/terminal_bench_generator.py`):
- `TerminalBenchAgentOutput` now carries `rollout_details` (Harbor's
  raw per-turn data: `prompt_token_ids`, `completion_token_ids`,
  `logprobs`).
- New module-level `build_step_wise_generator_output_from_trial_outputs`
  helper flattens each trajectory's `rollout_details` into one
  GeneratorOutput row per turn. Failed trajectories collapse to a
  single zeroed placeholder step (so the batch shape stays consistent
  and the trainer doesn't NaN).
- `TerminalBenchGenerator.__init__` takes a `step_wise_training: bool`
  flag. At the end of `generate()`, if on, we call the helper to build
  the step-wise output (while preserving the trajectory-level
  `rollout_metrics`), and propagate `exclude_from_baseline` per-turn
  via `tid.to_string()` keys.
- `main_tbench.py` passes `cfg.trainer.step_wise_training` to the
  generator.

**Not ported from PR #1542:** the `use_expandable_segments` allocator
toggle; covered at runtime by
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the launch
scripts.

---

## Launch scripts

Two smoke-test scripts are included:

### `skyrl-train/run_tbench_stepwise_test.sh` — colocated (8 GPUs)
- `colocate_all=true`, all 8 GPUs host both FSDP + vLLM.
- `train_batch_size=4`, `n_samples_per_prompt=4` (smoke-test size).
- `trainer.step_wise_training=true`, `generator.merge_stepwise_output=true`.
- Harbor env: Modal; `collect_rollout_details=true` for per-turn data;
  `max_episodes=16`; `enable_summarize=false`.

**Result on step 1:** 987.20s total, `avg_final_rewards=0.1875`
(3/16 solved), `policy_loss=9.9e-6`, `grad_norm=0.29`,
`policy_update_steps=3.0`. Prefix-aware merging: 46 → 46 sequences
(no merges for terminus-2 as discussed).

### `skyrl-train/run_tbench_stepwise_async_test.sh` — fully-async (4+4 GPUs)
- `colocate_all=false` → selects `FullyAsyncRayPPOTrainer`.
- 4 GPUs for FSDP policy+ref, 4 GPUs for vLLM inference engines.
- `trainer.fully_async.max_staleness_steps=1`, `num_parallel_generation_workers=8`.
- `generator.batched=false` (async assertion).

**Result on step 1:** 492.50s total. `wait_for_generation_buffer=464s`
(buffer filling, not compute), `convert_to_training_input=0.08s`
(effective_batch=35 samples across 4 variable-size groups —
**proves per-group size fix works**), `fwd_logprobs=8.5s`,
`policy_train=11.32s`, `sync_weights=8.47s`.
`avg_final_rewards=0.3125` (5/16 solved).

Both runs use:
- Data: `$HOME/data/harbor/CodeContests` (`open-thoughts/CodeContests`).
- Model: `Qwen/Qwen3-8B`.
- Caches/trials/ckpts on `/mnt/local_storage` (keeps `~/default` ≤ 10 GB).
- Env vars sourced from `~/.bashrc` and propagated to Ray workers via
  the §3 patch.

---

## Commit history (8 commits)

```
ab706c4a [test] Add fully-async step-wise launch script
750aeed4 [stepwise][port] PR #1536 plumb step-wise through fully-async trainer
61d4ec20 [stepwise][port] PR #1285 unified left-pad + right-aligned response tensors
49ea147a [stepwise][port] PR #1538 prefix-aware merging for step-wise training
ef1256a8 [stepwise][port] Fix unhashable TrajectoryID key in step-wise output builder
d70c618d [stepwise][port] Fix _remove_tail_data to keep small batches on large dp_size
10c9009d [stepwise][port] TB generator emits per-turn entries when step_wise_training=true
370cdbae [stepwise][port] Ray env-var propagation + #1281 validation + #1507 advantage broadcast + small-batch guard
```

## Wandb runs used for validation

| Run | Scenario | Key metrics |
|---|---|---|
| `m67n5547` | colocated, merging off (initial step-wise port) | step=761.9s, avg_final_rewards=0.1875 |
| `vhd2tjnf` | colocated, with PR #1285 layout + PR #1538 merge | step=987s, policy_loss=9.9e-6, 46→46 merge |
| `zjht2t0a` | fully-async (PR #1536 validation) | step=492s, effective_batch=35 in 4 groups, avg_final_rewards=0.3125 |

Project: <https://wandb.ai/sky-posttraining-uc-berkeley/terminal_bench>

## What's deliberately NOT ported

- **PR #1529 full prompt-based mini-batching** — threading
  `prompt_mini_batch_boundaries` through the dispatch stack. The user
  accepted multiple gradient updates per step as the small-scale
  approximation. Worth doing at scale.
- **PR #1542 `use_expandable_segments` toggle** — covered by
  `PYTORCH_CUDA_ALLOC_CONF` env var in the launch scripts.
- Any documentation updates from the original PRs
  (`docs/content/docs/...`).
- Unit tests. The validation here is end-to-end training runs.
