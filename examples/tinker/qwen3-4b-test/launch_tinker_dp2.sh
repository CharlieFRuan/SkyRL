#!/bin/bash
cd /home/charlie_key/SkyRL
# Keep Ray's temp/session dirs off the 93G root disk (-> 6.3T /home volume).
export TMPDIR=/home/charlie_key/ray_tmp
export RAY_TMPDIR=/home/charlie_key/ray_tmp
# Single-node: do NOT ship the working dir (incl. the 13G .venv) to Ray workers
# and do NOT re-run `uv run` on workers (which rebuilds transformer-engine from
# source and fails on cudnn.h). Workers reuse the already-built local .venv.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
exec uv run --extra tinker --extra megatron -m skyrl.tinker.api \
  --base-model "Qwen/Qwen3-4B-Instruct-2507" \
  --backend megatron \
  --backend-config '{"trainer.placement.policy_num_gpus_per_node": 2, "generator.inference_engine.num_engines": 2, "trainer.policy.megatron_config.tensor_model_parallel_size": 1, "trainer.policy.megatron_config.pipeline_model_parallel_size": 1, "trainer.log_path": "/home/charlie_key/skyrl-logs"}'
