#!/usr/bin/env bash
# The parity baseline: the same mini-swe-agent run as run_swebench_eval_mini_swe.sh, but with stock `harbor run` against
# a plain vLLM endpoint (no skycap). Same tasks, agent version, config file, timeouts, sandbox class and sandbox env.
#
#   BASE_URL=https://relay.example.com/base/v1 JOB_NAME=baseline-r1 bash examples/train_integrations/harbor_skycap/run_swebench_baseline_mini_swe.sh
#
# The endpoint must serve SERVED_MODEL_NAME with the eval's engine settings (vLLM 0.30.0, max_model_len 73728, qwen3
# reasoning parser, qwen3_coder tool parser). Knobs: BASE_URL, API_KEY_FILE, TASKS_DIR, JOBS_DIR, JOB_NAME, N_CONCURRENT.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-35b-a3b}"
TASKS_DIR="${TASKS_DIR:-/home/charlieruan/mimo/swebench100/raw}"
JOBS_DIR="${JOBS_DIR:-/home/charlieruan/mimo/jobs-swe100}"
JOB_NAME="${JOB_NAME:?set JOB_NAME}"
BASE_URL="${BASE_URL:?set BASE_URL to the OpenAI-compatible base, ending in /v1}"
API_KEY_FILE="${API_KEY_FILE:-/home/charlieruan/mimo/vllm_api_key}"
N_CONCURRENT="${N_CONCURRENT:-100}"
LABELED_DAYTONA_DIR="${LABELED_DAYTONA_DIR:-/home/charlieruan/mimo/rl-env-cleaner/mimo_swe}"
KEYS_FILE="${KEYS_FILE:-$HOME/charlie_keys.env}"
KEY="$(cat "$API_KEY_FILE")"
set -a
# shellcheck disable=SC1090
source "$KEYS_FILE"
set +a
export PYTHONPATH="$LABELED_DAYTONA_DIR${PYTHONPATH:+:$PYTHONPATH}" MIMO_RUN_ID="${MIMO_RUN_ID:-$JOB_NAME}" MIMO_TTL_MINUTES=180
# Agent 1800 s and verifier 300 s out of the tasks' 3000 s budgets (Marin's limits; the skycap eval sets them directly).
exec harbor run -y -p "$TASKS_DIR" -o "$JOBS_DIR" --job-name "$JOB_NAME" -e daytona -n "$N_CONCURRENT" \
  --environment-import-path labeled_daytona:LabeledDaytonaEnv \
  -a mini-swe-agent -m "openai/$SERVED_MODEL_NAME" \
  --ak version=2.4.6 --ak config_file="$HERE/mini_swe_agent_qwen36.yaml" \
  --ae OPENAI_API_KEY="$KEY" --ae OPENAI_API_BASE="$BASE_URL" --ae OPENAI_BASE_URL="$BASE_URL" \
  --ae MSWEA_API_KEY="$KEY" --ae MSWEA_COST_TRACKING=ignore_errors --ae LITELLM_LOCAL_MODEL_COST_MAP=True \
  --agent-timeout-multiplier 0.6 --verifier-timeout-multiplier 0.1
