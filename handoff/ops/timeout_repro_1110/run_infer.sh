#!/usr/bin/env bash
# Inference-only repro of the CalibForge RL rollouts (2026-11-10): the training run's generator path (12 vLLM engines
# TP2, sticky_least_loaded router, skycap servers, Terminus 2 / TerminusThink, each task's own timeouts) through
# SkyRL's eval entrypoint (main_harbor_skycap_eval), with per-trial timing events (SKYCAP_TIMING, timing.py).
# Run ON a worker; the driver joins the Anyscale Ray cluster (never `ray stop` it).
#
#   cd $D && setsid nohup bash ops/timeout_repro_1110/run_infer.sh <run-name> [overrides...] &> logs/<run-name>.log < /dev/null &
#
# Env: TASKS_DIR (default $D/data/calibforge_v2_dune), SANDBOX=dune|daytona (default dune), MAX_CONCURRENCY (500),
#      TRAJECTORIES_PER_SECOND (2), SAMPLES (1), NUM_ENGINES (12), TP_SIZE (2),
#      SKYCAP_FAST_EXEC=1 (fast_exec.py), SKYCAP_WAIT_WATCH=measure|return (wait_watch.py),
#      AGENT=cluster (default: Terminus on the cluster) | installed (terminus_installed.py: the loop runs in the
#      sandbox and calls the model through the sandbox relay; implies SKYCAP_FAST_EXEC=1).
set -euo pipefail
D=/mnt/local_storage/charlie
# Run from a snapshot: bash reads a script as it goes, so editing (or rsyncing) this file would break a running job.
if [[ -z "${INFER_SNAPSHOT:-}" ]]; then
  mkdir -p $D/runs/$1; cp "$0" $D/runs/$1/launcher.sh
  INFER_SNAPSHOT=1 exec bash $D/runs/$1/launcher.sh "$@"
fi
RUN_NAME=${1:?run name}; shift
RUN_DIR=$D/runs/$RUN_NAME
REPO=$D/SkyRL
VENV=${VENV:-$D/venvs/upstream-megatron}
MODEL=${MODEL:-$D/hf/Qwen3.5-9B}
TASKS_DIR=${TASKS_DIR:-$D/data/calibforge_v2_dune}
SANDBOX=${SANDBOX:-dune}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
NUM_ENGINES=${NUM_ENGINES:-12}
TP_SIZE=${TP_SIZE:-2}
mkdir -p "$RUN_DIR/trials" "$RUN_DIR/timing"
export RAY_ADDRESS=auto
export XDG_CACHE_HOME=$D/.cache UV_CACHE_DIR=/home/ray/.cache/uv
# torch 2.13 needs NCCL 2.29; the image's /usr/lib libnccl (2.28.9) shadows the wheel's.
export LD_LIBRARY_PATH=$D/nccl/pkg/nvidia/nccl/lib:/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export SKYRL_LD_LIBRARY_PATH_EXPORT=1
export SKYRL_PY_EXECUTABLE="$VENV/bin/python"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export RAY_enable_ray_event=0 RAY_task_events_report_interval_ms=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export SKYRL_RAY_PG_TIMEOUT_IN_S=1800
umask 077
if [[ "$SANDBOX" == dune ]]; then
  cat $D/keys.env $D/dune_keys.env > "$RUN_DIR/keys.env"
elif [[ -n "${SANDBOX_KEYS:-}" ]]; then
  # A Daytona organization other than the one in keys.env (e.g. $D/daytona_keys.env, the shared org of 11-14).
  { grep -v '^export DAYTONA_API' $D/keys.env; cat "$SANDBOX_KEYS"; } > "$RUN_DIR/keys.env"
else
  grep -v '^export DAYTONA_API_URL' $D/keys.env > "$RUN_DIR/keys.env"
fi
umask 022
set -a; . "$RUN_DIR/keys.env"; set +a
export HF_TOKEN=${HF_TOKEN:-$HF_API_KEY}
[[ "$SANDBOX" == dune || -n "${SANDBOX_KEYS:-}" ]] || unset DAYTONA_API_URL
ulimit -n 524288 || true
# One attempt per trial, as `harbor run` makes; a backstop for a wedged sandbox (all but 4 tasks allow the agent <= 3600 s; those 4 are cut here).
export SKYCAP_TRIAL_ATTEMPTS=${SKYCAP_TRIAL_ATTEMPTS:-1}
export SKYCAP_TRIAL_TIMEOUT_SEC=${SKYCAP_TRIAL_TIMEOUT_SEC:-5400}
export SKYCAP_VERIFIER_RETRIES=${SKYCAP_VERIFIER_RETRIES:-0}
# Per-trial timing events (<trial>/timing.jsonl), the event loop's lag, and each skycap call's server-side split.
export SKYCAP_TIMING=1 SKYCAP_TIMING_DIR="$RUN_DIR/timing"
AGENT=${AGENT:-cluster}
RELAY_TRANSPORT=preview; [[ "$SANDBOX" == dune ]] && RELAY_TRANSPORT=exec
if [[ "$AGENT" == installed ]]; then
  export SKYCAP_FAST_EXEC=${SKYCAP_FAST_EXEC:-1}
  AGENT_PATH=examples.train_integrations.harbor_skycap.terminus_installed:TerminusInstalled
  ENV_PATH=examples.train_integrations.harbor_skycap.sandbox_relay.environment:RelayDaytonaEnvironment
  [[ "$RELAY_TRANSPORT" == exec ]] && ENV_PATH=examples.train_integrations.harbor_skycap.sandbox_relay.exec_environment:ExecRelayDaytonaEnvironment
  # ENV_CLASS: another environment class (e.g. harbor_skycap.daytona_setup:SetupRelayDaytonaEnvironment).
  ENV_PATH=${ENV_CLASS:-$ENV_PATH}
  # Each trial holds one long-poll open for its runner: more connections than the Daytona client's default pool (250).
  RELAY_ARGS=(
    skycap.sandbox_relay.enabled=true
    skycap.sandbox_relay.transport=$RELAY_TRANSPORT
    harbor_trial_config.environment.kwargs.connection_pool_maxsize=4000
  )
else
  AGENT_PATH=examples.train_integrations.harbor_skycap.terminus_think:TerminusThink
  ENV_PATH=examples.train_integrations.harbor_skycap.sandbox_relay.environment:RelayDaytonaEnvironment
  RELAY_ARGS=()
fi
cd "$REPO"
echo "==> $RUN_NAME: agent $AGENT, wait watch ${SKYCAP_WAIT_WATCH:-off},"
echo "==> $RUN_NAME: $(find -L "$TASKS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l) tasks x ${SAMPLES:-1}, sandbox $SANDBOX, concurrency ${MAX_CONCURRENCY:-500}, fast exec ${SKYCAP_FAST_EXEC:-0}, run dir $RUN_DIR"
"$VENV/bin/python" \
  -m examples.train_integrations.harbor_skycap.entrypoints.main_harbor_skycap_eval \
  data.train_data="['$TASKS_DIR']" \
  data.val_data="['$TASKS_DIR']" \
  trainer.policy.model.path="$MODEL" \
  generator.inference_engine.served_model_name=qwen3.5-9b \
  trainer.project_name=skycap-calibforge-timeout-repro \
  trainer.run_name="$RUN_NAME" \
  trainer.logger=console \
  trainer.export_path="$RUN_DIR/exports" \
  trainer.ckpt_path="$RUN_DIR/ckpts" \
  trainer.log_path="$RUN_DIR/logs" \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.ref_num_gpus_per_node=8 \
  trainer.algorithm.use_kl_loss=false \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  trainer.algorithm.max_seq_len="$MAX_MODEL_LEN" \
  generator.eval_n_samples_per_prompt="${SAMPLES:-1}" \
  generator.apply_overlong_filtering=false \
  generator.step_wise_trajectories=true \
  generator.merge_stepwise_output=false \
  generator.batched=false \
  generator.use_cache_salt=true \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.distributed_executor_backend=mp \
  generator.inference_engine.num_engines="$NUM_ENGINES" \
  generator.inference_engine.tensor_parallel_size="$TP_SIZE" \
  generator.inference_engine.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.85}" \
  generator.inference_engine.language_model_only=true \
  generator.inference_engine.engine_init_kwargs.gdn_prefill_backend=triton \
  generator.inference_engine.engine_init_kwargs.max_model_len="$MAX_MODEL_LEN" \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  generator.inference_engine.router_init_kwargs.policy=sticky_least_loaded \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.top_k=-1 \
  generator.sampling_params.min_p=0.0 \
  generator.rate_limit.enabled=true \
  generator.rate_limit.trajectories_per_second="${TRAJECTORIES_PER_SECOND:-2}" \
  generator.rate_limit.max_concurrency="${MAX_CONCURRENCY:-500}" \
  skycap.num_servers="${SKYCAP_SERVERS:-8}" \
  skycap.train_paths=final \
  skycap.record_dir="$RUN_DIR/skycap" \
  skycap.ttl=14400 \
  skycap.wandb.enabled=false \
  skycap.tokenizer=Qwen/Qwen3.5-9B \
  skycap.thinking_retention=all \
  skycap.exposure.type=none \
  harbor_trial_config.trials_dir="$RUN_DIR/trials" \
  harbor_trial_config.agent.name=null \
  harbor_trial_config.agent.import_path="$AGENT_PATH" \
  harbor_trial_config.agent.kwargs.max_turns=250 \
  harbor_trial_config.agent.kwargs.enable_summarize=false \
  harbor_trial_config.agent.kwargs.model_info.max_input_tokens="$MAX_MODEL_LEN" \
  harbor_trial_config.agent.kwargs.model_info.max_output_tokens="$MAX_MODEL_LEN" \
  harbor_trial_config.agent.kwargs.llm_kwargs.timeout=3600 \
  harbor_trial_config.agent.override_timeout_sec=null \
  harbor_trial_config.environment.type=daytona \
  harbor_trial_config.environment.import_path="$ENV_PATH" \
  "${RELAY_ARGS[@]}" \
  harbor_trial_config.environment.kwargs.labels.owner=charlieruan \
  harbor_trial_config.environment.kwargs.labels.run="$RUN_NAME" \
  harbor_trial_config.environment.kwargs.ttl_minutes=240 \
  harbor_trial_config.environment.kwargs.auto_stop_interval_mins=0 \
  harbor_trial_config.environment.override_cpus=null \
  harbor_trial_config.environment.override_memory_mb=null \
  harbor_trial_config.environment.override_storage_mb=null \
  "$@" 2>&1 | tee -a "$RUN_DIR/run.log"
echo "==> done: $RUN_NAME"
