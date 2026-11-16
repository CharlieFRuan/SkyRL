"""Terminus-2's agent loop, run INSIDE the task's sandbox (the in-sandbox half of ``terminus_installed.py``).

    /installed-agent/venv/bin/python /installed-agent/pkgs/terminus_runner.py /installed-agent/config.json

The agent is Harbor's own (``terminus_think.TerminusThink``, the package copied from the cluster's install), driving
tmux through a local environment: every ``exec`` is a subprocess here instead of a round trip through the sandbox
provider's API. Model calls go wherever ``kwargs.api_base`` points: the sandbox relay's stub on ``127.0.0.1``.

Writes into ``/logs/agent`` (Harbor downloads it): the agent's own files (``trajectory.json``, the recording),
``runner_result.json`` (the token counts and metadata of Harbor's agent context, and the exception that ended the run,
if any), ``timing.jsonl`` and ``loop_lag.jsonl`` when ``timing`` is on, and ``runner.log``. Its exit code goes to
``/installed-agent/exit_code``: 0, or 3 when the agent raised. SIGTERM (the agent timeout, sent from the cluster)
cancels the agent, which still writes its trajectory and its counts.
"""

import asyncio
import json
import os
import pwd
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LOGS_DIR = Path("/logs/agent")
EXIT_CODE_PATH = Path("/installed-agent/exit_code")
#: Environment the agent's own process was given (its endpoint and keys), kept out of the shell the model drives.
PRIVATE_ENV_PREFIXES = ("OPENAI_", "HOSTED_VLLM_", "MSWEA_", "LITELLM_", "SKYCAP_")


class _TrialPaths:
    trial_dir = LOGS_DIR
    agent_dir = LOGS_DIR


class LocalEnvironment:
    """What Terminus-2 and its tmux session use of a Harbor environment, on this machine."""

    session_id = "local"
    default_user = None
    trial_paths = _TrialPaths()
    skycap_timing_lag_path = LOGS_DIR / "loop_lag.jsonl"

    def __init__(self, workdir=None):
        self._workdir = workdir
        self._env = {k: v for k, v in os.environ.items() if not k.startswith(PRIVATE_ENV_PREFIXES)}
        self._me = self._user_name(os.geteuid())

    @staticmethod
    def _user_name(uid: int) -> str:
        try:
            return pwd.getpwuid(uid).pw_name
        except KeyError:  # a uid without a passwd entry
            return str(uid)

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        from harbor.environments.base import ExecResult

        # A NUL can't be passed in a process argument, and the model can type one.
        command = command.replace("\x00", "")
        argv = ["bash", "-c", command]
        if user is not None and os.geteuid() == 0:
            name = self._user_name(user) if isinstance(user, int) else user
            if name != self._me:
                # As Harbor's Daytona environment switches user (su, keeping the working directory).
                argv = ["su", name, "-s", "/bin/bash", "-c", command]
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd or self._workdir,
            env={**self._env, **(env or {})},
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
        except asyncio.TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
            return ExecResult(stdout=stdout.decode(errors="replace"), stderr=stderr.decode(errors="replace"), return_code=124)
        return ExecResult(
            stdout=stdout.decode(errors="replace"), stderr=stderr.decode(errors="replace"), return_code=process.returncode
        )

    async def upload_file(self, source_path, target_path):
        if Path(source_path).resolve() != Path(target_path).resolve():
            Path(target_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target_path)

    async def download_file(self, source_path, target_path):
        await self.upload_file(source_path, target_path)

    async def is_dir(self, path, user=None):
        return os.path.isdir(path)


async def main(config: dict) -> int:
    from harbor.models.agent.context import AgentContext

    import terminus_think  # noqa: F401 - applies the NeMo Evaluator patches

    environment = LocalEnvironment(config.get("workdir"))
    if config.get("timing"):
        import timing

        timing.apply_agent()
        LocalEnvironment.exec = timing._env_timed("exec", LocalEnvironment.exec, timing._describe_exec)
    if config.get("agent_path_drop") or config.get("agent_path_first"):
        import agent_path

        agent_path.apply(config.get("agent_path_drop") or [], config.get("agent_path_first") or [])
    if config.get("terminal_guard"):
        import terminal_guard

        terminal_guard.apply()
    agent = terminus_think.TerminusThink(logs_dir=LOGS_DIR, model_name=config["model_name"], **config["kwargs"])
    context = AgentContext()
    result: dict = {"started_at": time.time()}

    async def run() -> None:
        await agent.setup(environment)
        result["setup_done_at"] = time.time()
        await agent.run(config["instruction"], environment, context)

    task = asyncio.ensure_future(run())
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    code = 0
    try:
        await task
    except asyncio.CancelledError:
        result["exception_type"] = "Cancelled"
        code = 143
    except BaseException as error:  # noqa: BLE001 - reported to the cluster, which raises it there
        result["exception_type"] = type(error).__name__
        result["exception_message"] = str(error)[:4000]
        result["exception_traceback"] = traceback.format_exc()[-8000:]
        code = 3
    result.update(
        finished_at=time.time(),
        n_input_tokens=context.n_input_tokens,
        n_output_tokens=context.n_output_tokens,
        n_cache_tokens=context.n_cache_tokens,
        cost_usd=context.cost_usd,
        metadata=context.metadata,
    )
    (LOGS_DIR / "runner_result.json").write_text(json.dumps(result, default=str))
    return code


if __name__ == "__main__":
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    exit_code = 1
    try:
        exit_code = asyncio.run(main(json.loads(Path(sys.argv[1]).read_text())))
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
    finally:
        EXIT_CODE_PATH.write_text(str(exit_code))
    sys.exit(exit_code)
