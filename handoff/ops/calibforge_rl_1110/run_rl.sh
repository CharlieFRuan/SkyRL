#!/usr/bin/env bash
# CalibForge v2 RL, Qwen3.5-9B, fully async, 2026-11-10 (SkyRL-upstream run_fully_async_terminus.sh).
# As the 11-07 run (DPPO 0.15, prompt_mean, lr 1e-6, 16 x 16, zero-variance filter, cache salt,
# sticky_least_loaded, thinking kept, no summarization, timeouts and out-of-context trained as reward 0) except:
#   - every agent timeout 3600 s (train and eval);
#   - training context 131072, eval context 262144 (the engines serve 262144; skycap.train_max_model_len holds
#     training trajectories to 131072);
#   - Terminus-2 runs INSIDE each sandbox (terminus_installed.py) and calls the model through the sandbox relay,
#     with the Ctrl-C guard, one-call exec for the runner's long-poll and verifier retries;
#   - sandboxes on Dune Cloud (8 vCPU / 16 GB per sandbox now): every training and TB2.1 eval sandbox 4 vCPU / 8 GB /
#     20 GB;
#   - no summarization, train or eval: a trial that fills its context ends there (the friend's setup);
#   - one reply is capped at 16,384 tokens, train and eval (skycap.max_tokens_per_turn), as the friend's run;
#   - eval (TB2.1 x 3) before training and every 25 steps, sampled like training since the 11-11 resume;
#     checkpoints every 25 steps, all kept;
#   - staleness 3 and 1024 trials in flight (the staleness window: (3 + 1) x 16 x 16).
# Run ON a GPU worker; the driver joins the Anyscale Ray cluster (never `ray stop` it).
#
#   cd $D && setsid nohup bash ops/calibforge_rl_1110/run_rl.sh <run-name> [overrides...] &> logs/<run-name>.log < /dev/null &
set -euo pipefail
D=/mnt/local_storage/charlie
# Run from a snapshot: bash reads a script as it goes, so editing (or rsyncing) this file would break a running job.
if [[ -z "${RL_SNAPSHOT:-}" ]]; then
  mkdir -p $D/runs/$1; cp "$0" $D/runs/$1/launcher.sh
  cp $D/SkyRL/examples/train_integrations/harbor_skycap/run_fully_async_terminus.sh $D/runs/$1/run_fully_async_terminus.sh
  RL_SNAPSHOT=1 exec bash $D/runs/$1/launcher.sh "$@"
fi
export RUN_NAME=${1:?run name}
export RUN_DIR=$D/runs/$RUN_NAME
export RAY_ADDRESS=auto
export MODEL=${MODEL:-$D/hf/Qwen3.5-9B}
# PROVIDER=dune (default): each task's own prebuilt image, exec relay. PROVIDER=daytona: the shared Daytona org of
# 11-14 (no image builds, 5 snapshots): tasks converted to "base snapshot + setup script" (make_daytona_tasks.py,
# harbor_skycap/daytona_setup.py), relay over signed preview URLs; TB2.1 cannot run there, so set EVAL_INTERVAL=0.
export PROVIDER=${PROVIDER:-dune}
if [[ "$PROVIDER" == daytona ]]; then
  export TASKS_DIR=${TASKS_DIR:-$D/data/calibforge_v2_daytona}
  ENV_CLASS=examples.train_integrations.harbor_skycap.daytona_setup:SetupRelayDaytonaEnvironment
  RELAY_TRANSPORT=preview
elif [[ "$PROVIDER" == daytona_eval ]]; then
  # The second Daytona org (11-14, $D/daytona_eval_keys.env): image builds allowed, 4 vCPU / 8 GiB / 20 GiB fine, so
  # tasks run from their own images as on Dune; relay over signed preview URLs. Used for TB2.1 evals.
  ENV_CLASS=examples.train_integrations.harbor_skycap.sandbox_relay.environment:RelayDaytonaEnvironment
  RELAY_TRANSPORT=preview
else
  ENV_CLASS=examples.train_integrations.harbor_skycap.sandbox_relay.exec_environment:ExecRelayDaytonaEnvironment
  RELAY_TRANSPORT=exec
fi
export TASKS_DIR=${TASKS_DIR:-$D/data/calibforge_v2_train48}
export EVAL_TASKS_DIR=${EVAL_TASKS_DIR:-$D/data/tbench21_4cpu}
export LABELS_OWNER=charlieruan
export XDG_CACHE_HOME=$D/.cache UV_CACHE_DIR=/home/ray/.cache/uv
export VENV=${VENV:-$D/venvs/upstream-megatron}
# torch 2.13 needs NCCL 2.29; the image's /usr/lib libnccl (2.28.9) shadows the wheel's. EFA libs for multi-node.
export LD_LIBRARY_PATH=$D/nccl/pkg/nvidia/nccl/lib:/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export SKYRL_LD_LIBRARY_PATH_EXPORT=1
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 NCCL_DEBUG=WARN
# Anyscale's raylet env points W&B at its hosted server; ours is wandb.ai.
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai} WANDB_ENTITY=${WANDB_ENTITY:-sky-posttraining-uc-berkeley}
export PROJECT=${PROJECT:-skycap-calibforge-fully-async}
export RAY_enable_ray_event=0 RAY_task_events_report_interval_ms=0
# Sandboxes on Dune (Daytona's API at DAYTONA_API_URL; forwarded to the Ray workers by skyrl/train/utils/utils.py).
umask 077
if [[ "$PROVIDER" == daytona ]]; then
  { grep -v '^export DAYTONA_API' $D/keys.env; cat $D/daytona_keys.env; } > "$RUN_DIR/keys.env"
elif [[ "$PROVIDER" == daytona_eval ]]; then
  { grep -v '^export DAYTONA_API' $D/keys.env; cat $D/daytona_eval_keys.env; } > "$RUN_DIR/keys.env"
else
  cat $D/keys.env $D/dune_keys.env > "$RUN_DIR/keys.env"
fi
umask 022
export KEYS_FILE="$RUN_DIR/keys.env"
set -a; . "$KEYS_FILE"; set +a
export HF_TOKEN=${HF_TOKEN:-$HF_API_KEY}
ulimit -n 524288 || true

# The recipe's knobs (run_fully_async_terminus.sh).
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16} GROUP_SIZE=${GROUP_SIZE:-16} MAX_STALENESS_STEPS=${MAX_STALENESS_STEPS:-3}
export MAX_CONCURRENCY=${MAX_CONCURRENCY:-1024} TRAJECTORIES_PER_SECOND=${TRAJECTORIES_PER_SECOND:-2}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}          # what the trainer takes; the engines' limit is set below
export TRAIN_PATHS=final SUMMARIZE=false TRAIN_ON_TIMEOUTS=true
export EVAL_INTERVAL=${EVAL_INTERVAL:-25} EVAL_BEFORE_TRAIN=${EVAL_BEFORE_TRAIN:-true} EVAL_SAMPLES=${EVAL_SAMPLES:-3}
export CKPT_INTERVAL=${CKPT_INTERVAL:-5} MAX_CKPTS=${MAX_CKPTS:-40}
export EPOCHS=${EPOCHS:-3} MAX_TURNS=${MAX_TURNS:-250}
EVAL_MODEL_LEN=${EVAL_MODEL_LEN:-262144}
AGENT_TIMEOUT=${AGENT_TIMEOUT:-3600}
# Eval-only agent kwargs. Summarization stays off in eval too (user, 11-10: follow the friend's setup).
EVAL_AGENT_KWARGS="${EVAL_AGENT_KWARGS:-{enable_summarize: false}}"
# One reply's token cap, train and eval (user, 11-10: the friend's run caps a turn at 16k). Terminus is told the same
# number (model_info.max_output_tokens), which it quotes when it asks the model to re-issue a truncated reply.
MAX_TOKENS_PER_TURN=${MAX_TOKENS_PER_TURN:-16384}
# Eval sampling = the training sampling (1.0 / 1.0, top_k and min_p off), as the reference recipe does (user, 11-11: "the
# training sampling parameter for eval"). The repo launcher's own default is Qwen3.5's 0.6 / 0.95 / top_k 20, under
# which the model loops more and more as training goes on (35% -> 45% of eval trials in 75 steps). Spelled out as a
# mapping (a `null` would reach the entrypoint as a string).
EVAL_SAMPLING_TRAINLIKE='{temperature: 1.0, top_p: 1.0, top_k: -1, min_p: 0.0}'
export EVAL_SAMPLING="${EVAL_SAMPLING:-$EVAL_SAMPLING_TRAINLIKE}"
# RESUME_PATH=<run>/ckpts/global_step_N resumes from another run's checkpoint (weights, optimizer, which tasks were
# consumed); this run then writes its own checkpoints. With EVAL_BEFORE_TRAIN the resumed step is evaluated first.
RESUME_ARGS=()
if [[ -n "${RESUME_PATH:-}" ]]; then
  export RESUME_MODE=from_path
  RESUME_ARGS=(trainer.resume_path="$RESUME_PATH")
fi

# The integration's opt-ins: one-call exec (the in-sandbox runner's long-poll), the Ctrl-C guard, a verifier whose
# tests never ran is retried, per-trial timing events and the event loop's lag.
export SKYCAP_FAST_EXEC=1 SKYCAP_TERMINAL_GUARD=1
# Harbor's tar / gzip of uploaded and downloaded directories in a thread pool instead of on the event loop (it froze
# every trial for up to 376 s), and no re-upload of the in-sandbox agent's own logs (loop_offload.py).
export SKYCAP_OFFLOAD_TAR=1 SKYCAP_SKIP_LOG_REUPLOAD=1
export SKYCAP_VERIFIER_RETRIES=${SKYCAP_VERIFIER_RETRIES:-3}
export SKYCAP_TIMING=1 SKYCAP_TIMING_DIR="$RUN_DIR/timing"
# Backstop per trial attempt: the agent's 3600 s + sandbox build + install + the verifier.
export SKYCAP_TRIAL_TIMEOUT_SEC=${SKYCAP_TRIAL_TIMEOUT_SEC:-6600}
# A provider outage pauses trial starts instead of failing every trial and training on empty batches
# (outage_guard.py; added after the 2 h Dune 503 outage of 11-14). SKYCAP_OUTAGE_GUARD=0 turns it off.
export SKYCAP_OUTAGE_GUARD=${SKYCAP_OUTAGE_GUARD:-1}
# An agent timeout with one model call stuck longer than this is an infrastructure failure, not a reward-0
# rollout (11-14: a lost relay message left 108 trials waiting ~48 min on one call). 0 or empty turns it off.
export SKYCAP_STALL_TIMEOUT_SEC=${SKYCAP_STALL_TIMEOUT_SEC-900}
# Snapshot org only (sandboxes start from ready snapshots in seconds): give up on one creation after 120 s so that
# Harbor's retry creates another sandbox (11-15: one broken Daytona runner held ~3% of creates for ~900 s each).
# Needs harbor_skycap/daytona.py of 11-15 on the workers. Not for orgs where a creation builds / pulls an image.
if [[ "$PROVIDER" == daytona ]]; then export SKYCAP_CREATE_TIMEOUT_SEC=${SKYCAP_CREATE_TIMEOUT_SEC-120}; fi
mkdir -p "$RUN_DIR/timing"

export REPO=$D/SkyRL
cd $D/SkyRL
exec bash $RUN_DIR/run_fully_async_terminus.sh \
  generator.inference_engine.engine_init_kwargs.max_model_len="$EVAL_MODEL_LEN" \
  skycap.train_max_model_len="$MAX_MODEL_LEN" \
  "skycap.eval_agent_kwargs=$EVAL_AGENT_KWARGS" \
  harbor_trial_config.agent.kwargs.model_info.max_input_tokens="$EVAL_MODEL_LEN" \
  harbor_trial_config.agent.kwargs.model_info.max_output_tokens="$MAX_TOKENS_PER_TURN" \
  skycap.max_tokens_per_turn="$MAX_TOKENS_PER_TURN" \
  harbor_trial_config.agent.override_timeout_sec="$AGENT_TIMEOUT" \
  harbor_trial_config.agent.import_path=examples.train_integrations.harbor_skycap.terminus_installed:TerminusInstalled \
  harbor_trial_config.environment.import_path="$ENV_CLASS" \
  harbor_trial_config.environment.kwargs.connection_pool_maxsize=4000 \
  skycap.sandbox_relay.enabled=true \
  skycap.sandbox_relay.transport="$RELAY_TRANSPORT" \
  "${RESUME_ARGS[@]}" \
  "${@:2}"
