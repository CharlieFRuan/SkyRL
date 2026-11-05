#!/usr/bin/env bash
# Evaluation only, through skycap: SWE-bench Verified with mini-swe-agent, or Terminal-Bench with Terminus-2, on one
# 8-GPU node (vLLM engines started by SkyRL, skycap servers in front of them, Harbor trials in Daytona). No training.
#
#   BENCH=swe MODEL_FAMILY=nemotron-3.5 TASKS_DIR=/path/to/swebench-verified \
#     bash examples/train_integrations/harbor_skycap/run_eval_skycap.sh <experiment> [overrides...]
#
# mini-swe-agent runs inside each trial's sandbox and reaches skycap through the sandbox relay (a stub in the sandbox
# that the cluster polls through Daytona's signed preview URLs; skycap.sandbox_relay), so the cluster needs no inbound
# connectivity. Terminus-2 calls the model from the trial's Ray worker and reaches skycap directly.
#
# Harbor writes each trial's directory under $RUN_DIR/trials as `harbor run` would: score them as a stock Harbor job.
#
# Knobs (environment):
#   BENCH                  swe (mini-swe-agent, each task's own timeouts and resources) or tb (Terminus-2)
#   MODEL_FAMILY           nemotron-3.5 or qwen3.5: model path, served name, the card's sampling, model flags
#   MODEL                  weights to serve (default: the family's under $MODELS_DIR)
#   TASKS_DIR              the Harbor task directories
#   SAMPLES                trials per task (default 3); N_CONCURRENT trials in flight (default 300)
#   NUM_ENGINES, TP_SIZE   vLLM engines (default 8 x TP 1); SKYCAP_SERVERS (default 8)
#   MAX_MODEL_LEN          context (default 262144)
#   AGENT_TIMEOUT          override every task's agent timeout (seconds); default each task's own
#   TB_CPUS, TB_MEMORY_MB  Terminal-Bench sandbox size (default 4 / 8192, NVIDIA's Lightning recipe)
#   TB_MAX_TURNS           Terminus-2 episodes (default: none for Nemotron, as NVIDIA's recipe; 250 for Qwen3.5, as
#                          Artificial Analysis). For Terminal-Bench's agent timeout of max(7200 s, the task's own),
#                          give TASKS_DIR copies of the tasks with that timeout.
#   RUN_DIR                default /tmp/harbor/runs/<experiment>
#   LABELS_OWNER           the sandboxes' owner label (default $USER)
#   RELAY_TRANSPORT        preview (default; Daytona's signed preview URLs) or exec (providers without preview URLs,
#                          e.g. Dune Cloud: the stubs are polled through the exec API)
set -euo pipefail
# REPO: the SkyRL checkout (default: this script's), for running a copy of this script from elsewhere.
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
HERE="$REPO/examples/train_integrations/harbor_skycap"
cd "$REPO"
EXPERIMENT="${1:?experiment name}"
shift
BENCH="${BENCH:?BENCH=swe or BENCH=tb}"
MODEL_FAMILY="${MODEL_FAMILY:?MODEL_FAMILY=nemotron-3.5 or qwen3.5}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
TASKS_DIR="${TASKS_DIR:?TASKS_DIR}"
SAMPLES="${SAMPLES:-3}"
N_CONCURRENT="${N_CONCURRENT:-300}"
NUM_ENGINES="${NUM_ENGINES:-8}"
TP_SIZE="${TP_SIZE:-1}"
SKYCAP_SERVERS="${SKYCAP_SERVERS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
RUN_DIR="${RUN_DIR:-/tmp/harbor/runs/$EXPERIMENT}"
LABELS_OWNER="${LABELS_OWNER:-$USER}"
mkdir -p "$RUN_DIR/trials"

# The model card's sampling (thinking on). skycap samples every call with generator.sampling_params, whatever the
# harness asks for.
case "$MODEL_FAMILY" in
  nemotron-3.5)
    MODEL="${MODEL:-$MODELS_DIR/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}"
    SERVED="nemotron-3.5-lightning"
    MSWEA_CONFIG="${MSWEA_CONFIG:-$HERE/mini_swe_agent_swebench_nemotron35.yaml}"
    SAMPLING=(generator.sampling_params.temperature=1.0 generator.sampling_params.top_p=0.95 generator.sampling_params.top_k=-1)
    DEFAULT_TB_MAX_TURNS=1000000
    # Every earlier turn's thinking stays in the history (NVIDIA's recipe replays it).
    MODEL_ARGS=(skycap.thinking_retention=all skycap.tokenizer=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) ;;
  qwen3.5)
    MODEL="${MODEL:-$MODELS_DIR/Qwen3.5-9B}"
    SERVED="qwen3.5-9b"
    MSWEA_CONFIG="${MSWEA_CONFIG:-$HERE/mini_swe_agent_swebench_qwen35.yaml}"
    # The card's thinking-mode preset for precise coding.
    SAMPLING=(generator.sampling_params.temperature=0.6 generator.sampling_params.top_p=0.95 generator.sampling_params.top_k=20)
    # A vision-language checkpoint served for text; Gated DeltaNet prefill on Triton (vllm-project/vllm#36921).
    DEFAULT_TB_MAX_TURNS=250
    # The history keeps the thinking Qwen3.5's chat template keeps (what a stock vLLM server renders).
    MODEL_ARGS=(
      skycap.thinking_retention=null
      skycap.tokenizer=Qwen/Qwen3.5-9B
      trainer.policy.language_model_only=true
      trainer.ref.language_model_only=true
      generator.inference_engine.language_model_only=true
      generator.inference_engine.engine_init_kwargs.gdn_prefill_backend=triton
    ) ;;
  *) echo "unknown MODEL_FAMILY=$MODEL_FAMILY (nemotron-3.5 or qwen3.5)" >&2; exit 1 ;;
esac

ENV_ARGS=(
  harbor_trial_config.environment.type=daytona
  harbor_trial_config.environment.import_path=examples.train_integrations.harbor_skycap.sandbox_relay.environment:RelayDaytonaEnvironment
  harbor_trial_config.environment.kwargs.labels.owner="$LABELS_OWNER"
  harbor_trial_config.environment.kwargs.labels.run="$EXPERIMENT"
  harbor_trial_config.environment.kwargs.ttl_minutes="${TTL_MINUTES:-480}"
  harbor_trial_config.environment.kwargs.auto_stop_interval_mins=0
  harbor_trial_config.environment.override_storage_mb=null
)
TIMEOUT_ARGS=(harbor_trial_config.agent.override_timeout_sec="${AGENT_TIMEOUT:-null}")
EXEC_ENV_ARGS=()
if [[ "$BENCH" == swe && "${RELAY_TRANSPORT:-preview}" == exec ]]; then
  EXEC_ENV_ARGS=(harbor_trial_config.environment.import_path=examples.train_integrations.harbor_skycap.sandbox_relay.exec_environment:ExecRelayDaytonaEnvironment)
fi
case "$BENCH" in
  swe)
    # Harbor installs `uv tool install mini-swe-agent==<version>`, so the version also carries uv flags: resolve as of
    # before litellm 1.106.0 (imports `mcp` without depending on it), and never build litellm from source (images
    # with glibc < 2.28 get its newest pure-Python wheel).
    MINI_SWE_AGENT_VERSION="${MINI_SWE_AGENT_VERSION:-2.4.5 --exclude-newer 2026-10-18T00:00:00Z --no-build-package litellm}"
    AGENT_ARGS=(
      harbor_trial_config.agent.name=mini-swe-agent
      harbor_trial_config.agent.kwargs.version="$MINI_SWE_AGENT_VERSION"
      harbor_trial_config.agent.kwargs.config_file="$MSWEA_CONFIG"
      # mini-swe-agent sends tool calls: skycap answers with parsed content, reasoning and tool calls.
      skycap.use_raw_content=false
      skycap.exposure.type=none
      skycap.sandbox_relay.enabled=true
      skycap.sandbox_relay.transport="${RELAY_TRANSPORT:-preview}"
      harbor_trial_config.environment.override_cpus=null
      harbor_trial_config.environment.override_memory_mb=null
    ) ;;
  tb)
    AGENT_ARGS=(
      # Terminus-2 that parses the reply after its thinking and replays it whole (terminus_think.py); skycap answers
      # with raw text, the thinking inline (skycap.use_raw_content, its default).
      harbor_trial_config.agent.name=null
      harbor_trial_config.agent.import_path=examples.train_integrations.harbor_skycap.terminus_think:TerminusThink
      harbor_trial_config.agent.kwargs.max_turns="${TB_MAX_TURNS:-$DEFAULT_TB_MAX_TURNS}"
      harbor_trial_config.agent.kwargs.enable_summarize="${TB_SUMMARIZE:-true}"
      harbor_trial_config.agent.kwargs.model_info.max_input_tokens="$MAX_MODEL_LEN"
      harbor_trial_config.agent.kwargs.model_info.max_output_tokens="$MAX_MODEL_LEN"
      harbor_trial_config.agent.kwargs.llm_kwargs.timeout=3600
      skycap.exposure.type=none
      harbor_trial_config.environment.override_cpus="${TB_CPUS:-4}"
      harbor_trial_config.environment.override_memory_mb="${TB_MEMORY_MB:-8192}"
    ) ;;
  *) echo "unknown BENCH=$BENCH (swe or tb)" >&2; exit 1 ;;
esac

# One attempt per trial, as `harbor run` makes: a failed trial scores 0 instead of being rerun. A backstop for a wedged
# sandbox: an attempt past SKYCAP_TRIAL_TIMEOUT_SEC is cancelled (well past build + agent + verifier timeouts).
export SKYCAP_TRIAL_ATTEMPTS="${SKYCAP_TRIAL_ATTEMPTS:-1}"
export SKYCAP_TRIAL_TIMEOUT_SEC="${SKYCAP_TRIAL_TIMEOUT_SEC:-21600}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# PYTHON: the interpreter command (default uv's isolated env with the extras); e.g. a prebuilt venv's python.
read -ra PY <<< "${PYTHON:-uv run --isolated --extra fsdp --extra harbor --extra skycap python}"
echo "==> $EXPERIMENT: $BENCH, $MODEL_FAMILY, $(find "$TASKS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l) tasks x $SAMPLES"
"${PY[@]}" -m examples.train_integrations.harbor_skycap.entrypoints.main_harbor_skycap_eval \
  data.train_data="['$TASKS_DIR']" \
  data.val_data="['$TASKS_DIR']" \
  trainer.policy.model.path="$MODEL" \
  generator.inference_engine.served_model_name="$SERVED" \
  trainer.project_name=harbor-skycap-eval \
  trainer.run_name="$EXPERIMENT" \
  trainer.logger=console \
  trainer.export_path="$RUN_DIR/exports" \
  trainer.ckpt_path="$RUN_DIR/ckpts" \
  trainer.log_path="$RUN_DIR/logs" \
  trainer.algorithm.max_seq_len="$MAX_MODEL_LEN" \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node="$((NUM_ENGINES * TP_SIZE))" \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.ref_num_gpus_per_node="$((NUM_ENGINES * TP_SIZE))" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines="$NUM_ENGINES" \
  generator.inference_engine.tensor_parallel_size="$TP_SIZE" \
  generator.inference_engine.gpu_memory_utilization=0.9 \
  generator.inference_engine.engine_init_kwargs.max_model_len="$MAX_MODEL_LEN" \
  "${SAMPLING[@]}" \
  generator.sampling_params.min_p=0.0 \
  generator.eval_n_samples_per_prompt="$SAMPLES" \
  generator.step_wise_trajectories=true \
  generator.merge_stepwise_output=false \
  generator.batched=false \
  generator.rate_limit.enabled=true \
  generator.rate_limit.trajectories_per_second=5 \
  generator.rate_limit.max_concurrency="$N_CONCURRENT" \
  skycap.num_servers="$SKYCAP_SERVERS" \
  skycap.record_dir="$RUN_DIR/skycap" \
  skycap.wandb.enabled=false \
  "${MODEL_ARGS[@]}" \
  harbor_trial_config.trials_dir="$RUN_DIR/trials" \
  "${AGENT_ARGS[@]}" \
  "${TIMEOUT_ARGS[@]}" \
  "${ENV_ARGS[@]}" \
  "${EXEC_ENV_ARGS[@]}" \
  "$@" 2>&1 | tee -a "$RUN_DIR/run.log"
echo "==> done: $EXPERIMENT (trials in $RUN_DIR/trials)"
