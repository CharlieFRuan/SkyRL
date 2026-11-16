"""Print the Harbor JobConfig (YAML) for a no-SkyCap baseline (ops/evals_1104/baseline_eval.sh), from env vars.

Mirrors run_eval_skycap.sh: mini-swe-agent through the sandbox relay with each task's own timeouts and resources (SWE),
or Terminus-2 calling the router directly in 4 vCPU / 8 GB sandboxes (TB). One attempt per trial, no retries.
"""

import json
import os

import yaml
from harbor.models.job.config import JobConfig

env = os.environ
bench, served = env["BENCH"], env["SERVED"]
sampling = json.loads(env["SAMPLING"])
max_len = int(env["MAX_MODEL_LEN"])
environment = {
    "type": "daytona",
    "import_path": "examples.train_integrations.harbor_skycap.sandbox_relay.environment:RelayDaytonaEnvironment",
    "kwargs": {
        "labels": {"owner": "charlieruan", "run": env["JOB"]},
        "ttl_minutes": int(env.get("TTL_MINUTES", "480")),
        "auto_stop_interval_mins": 0,
    },
    "suppress_override_warnings": True,
}
if env.get("STORAGE_MB"):
    environment["override_storage_mb"] = int(env["STORAGE_MB"])
if bench == "swe":
    base = "http://127.0.0.1:9876/v1"
    environment["kwargs"]["skycap_relay"] = {"manager": env["RELAY"], "target": "http://127.0.0.1:8000", "port": 9876}
    if env.get("RELAY_TRANSPORT") == "exec":
        environment["import_path"] = (
            "examples.train_integrations.harbor_skycap.sandbox_relay.exec_environment:ExecRelayDaytonaEnvironment"
        )
    agent = {
        "name": "mini-swe-agent",
        "model_name": f"openai/{served}",
        "kwargs": {
            "version": env.get(
                "MINI_SWE_AGENT_VERSION", "2.4.5 --exclude-newer 2026-10-18T00:00:00Z --no-build-package litellm"
            ),
            "config_file": env["MSWEA"],
        },
        "env": {
            "OPENAI_API_KEY": "dummy",
            "OPENAI_API_BASE": base,
            "OPENAI_BASE_URL": base,
            "MSWEA_API_KEY": "dummy",
            "MSWEA_COST_TRACKING": "ignore_errors",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        },
    }
else:
    environment["override_cpus"] = int(env.get("TB_CPUS", "4"))
    environment["override_memory_mb"] = int(env.get("TB_MEMORY_MB", "8192"))
    llm_kwargs = {k: v for k, v in sampling.items() if k != "temperature"}
    llm_kwargs.update({"timeout": 3600, "max_retries": 0, "api_key": "dummy"})
    if env.get("CHAT_TEMPLATE_KWARGS"):
        llm_kwargs["extra_body"] = {"chat_template_kwargs": json.loads(env["CHAT_TEMPLATE_KWARGS"])}
    agent = {
        # Terminus-2 that parses the reply after its thinking and replays it whole, as the SkyCap arm.
        "import_path": "examples.train_integrations.harbor_skycap.terminus_think:TerminusThink",
        "model_name": f"hosted_vllm/{served}",
        "kwargs": {
            "api_base": "http://127.0.0.1:8000/v1",
            "max_turns": int(env["TB_MAX_TURNS"]),
            "enable_summarize": env.get("TB_SUMMARIZE", "true") == "true",
            "temperature": sampling["temperature"],
            "model_info": {
                "max_input_tokens": max_len,
                "max_output_tokens": max_len,
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            },
            "llm_kwargs": llm_kwargs,
        },
    }
config = {
    "job_name": env["JOB"],
    "jobs_dir": f"{env['RUN_DIR']}/jobs",
    "n_attempts": int(env["SAMPLES"]),
    "n_concurrent_trials": int(env["N_CONCURRENT"]),
    "quiet": True,
    "retry": {"max_retries": 0},
    "environment": environment,
    "agents": [agent],
    "datasets": [{"path": env["TASKS_DIR"]}],
}
JobConfig.model_validate(config)
print(yaml.safe_dump(config, sort_keys=False))
