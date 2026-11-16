#!/usr/bin/env bash
# Runs ON a worker: SkyRL-upstream (branch skycap-evals-1104, at $D/SkyRL) megatron + harbor + skycap venv, by adding
# the harbor / skycap extras to $D/venvs/upstream-megatron in place (their wheels are in the node's uv cache from the
# evals venv; egress is throttled). --frozen: exactly the lock.
D=/mnt/local_storage/charlie
export UV_CACHE_DIR=/home/ray/.cache/uv UV_HTTP_TIMEOUT=900 UV_HTTP_RETRIES=8 UV_PROJECT_ENVIRONMENT=$D/venvs/upstream-megatron
cd $D/SkyRL
timeout 5400 uv sync --frozen --extra megatron --extra harbor --extra skycap && echo VENV_train_OK || echo VENV_train_FAIL
