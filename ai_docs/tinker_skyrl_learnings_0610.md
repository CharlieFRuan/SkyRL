# Running SkyRL Tinker on a Single Node — Learnings (2026-06-10)

Notes from getting the SkyRL Tinker server (Megatron backend) running on a single Together
H200 node, plus how we verified data parallelism (DP > 1).

## TL;DR — the two things that bite you

1. **Disk fills the 93G root (`/`) and the run dies with `No space left on device`.** Cause is
   *not* `--isolated`. When you launch with `uv run`, Ray's `RAY_ENABLE_UV_RUN_RUNTIME_ENV`
   (default `True`) auto-ships your whole project dir — **including the 13G `.venv`** — as the Ray
   worker `working_dir`, copied fresh into `/tmp/ray/session_*/runtime_resources/` *every* launch.
   The same hook re-runs `uv` on workers, which rebuilds `transformer-engine` from source and fails
   on `fatal error: cudnn.h: No such file or directory`.
2. **Ray's temp dir defaults to `/tmp` (on the 93G root).** Stale sessions accumulate there.

## Node disk layout (why it fills)

| Mount | Device | Size | Notes |
|-------|--------|------|-------|
| `/` | `/dev/vda2` | **93G** | small — fills easily, breaks runs |
| `/home`, `/scratch` | `/dev/md0` | ~7TB | the big array, multi-TB free |
| `/dev/shm` | tmpfs | ~880G | RAM-backed |

Keep bulky things (Ray sessions, tmp, logs, checkpoints) on `/home` / `/scratch`, never `/tmp`.
The usual root-fillers: `/tmp/ray` and `/tmp/skyrl-logs` (default `trainer.log_path`).

## The fix

Set these env vars before launching the API. They propagate to the engine subprocess (which the
API hardcodes to relaunch via `uv run`) and into its `ray.init`:

```bash
# Single-node: workers reuse the already-built local .venv.
# -> no 13G working_dir copy, no transformer-engine rebuild. ray_tmp drops to <1M/session.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
# Keep Ray session dirs off the 93G root, on the 7TB /home array.
export TMPDIR=/home/charlie_key/ray_tmp
export RAY_TMPDIR=/home/charlie_key/ray_tmp
```

> Note: the Tinker API (`skyrl/tinker/api.py`) always relaunches the engine via `uv run`, so you
> can't avoid the uv hook by starting the API with `.venv/bin/python` — disable it via the env var.

## How to run (single node, Megatron, DP=2 example)

```bash
cd /home/charlie_key/SkyRL
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export TMPDIR=/home/charlie_key/ray_tmp RAY_TMPDIR=/home/charlie_key/ray_tmp

uv run --extra tinker --extra megatron -m skyrl.tinker.api \
  --base-model "Qwen/Qwen3-4B-Instruct-2507" \
  --backend megatron \
  --backend-config '{"trainer.placement.policy_num_gpus_per_node": 2,
                     "generator.inference_engine.num_engines": 2,
                     "trainer.policy.megatron_config.tensor_model_parallel_size": 1,
                     "trainer.policy.megatron_config.pipeline_model_parallel_size": 1,
                     "trainer.log_path": "/home/charlie_key/skyrl-logs"}'
```

(reusable launcher: `/home/SkyRL/examples/tinker/qwen3-4b-test/launch_tinker_dp2.sh`)

**Config notes**
- `DP = (policy_num_gpus_per_node * policy_num_nodes) / (TP * PP * CP)`. Here `2 / (1*1*1) = 2`.
- Defaults are TP=PP=CP=1, `policy_num_gpus_per_node=1` → **DP=1**. You must bump GPUs for DP>1.
- `colocate_all=True` (default) asserts `policy_gpus == num_engines * tp * pp * dp`, so set
  `num_engines` = your GPU count (TP=1) or the build fails the assertion.
- To scale: e.g. 8 GPUs, TP=1 → `policy_num_gpus_per_node=8`, `num_engines=8` → DP=8.

## Verifying DP > 1 is actually used

With DP=2 (TP=PP=1), the server logged two distinct ranks:

```
Mesh Ranks: [MeshRank(dp=0, ... world_size=2, dp_size=2), MeshRank(dp=1, ... world_size=2, dp_size=2)]
```

Empirically, during each `forward_backward` **GPU 0 and GPU 1 spiked together** (712 MiB → ~125 GB,
both with util at the same time). The engine look-ahead-batches all pending `forward_backward`
requests before the next barrier into one batch, then `MeshDispatch` chunks it evenly across
`dp_size` ranks (32 examples → 16/16). So requests are **spread across all DP ranks**, not piled on
one. A 6-step `forward_backward`+`optim_step` smoke test showed loss decreasing 30.9 → 15.2,
~7.2s/step at steady state.

## Gotchas

- First step is slow (~50s) from model onload + kernel compile; later steps are fast.
- Under `colocate_all=True` the policy offloads to CPU between steps (GPUs sit at ~700 MiB idle)
  and onloads during the pass — that's expected, not a hang.
- The supervised `forward_backward`/`optim_step` path is what's verified here. `sl_loop` eval that
  *samples* additionally wakes the colocated vLLM engines (separate path).
