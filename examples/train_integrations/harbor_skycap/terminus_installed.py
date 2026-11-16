"""Terminus-2 as an installed agent: its loop runs inside the task's sandbox, not on the cluster.

``terminus_think.TerminusThink`` runs in the process that runs the trial: every command it types and every screen it
reads is a round trip (four, with Harbor's Daytona ``exec``) through the sandbox provider's API, and every trial's
loop shares that one process's event loop. Here the same agent code (the cluster's own Harbor package, copied in) runs
in the sandbox (``terminus_runner.py``) and drives tmux locally; only its model calls leave the sandbox, through
the sandbox relay (``skycap.sandbox_relay``), as any installed agent's do.

    harbor_trial_config.agent.name=null
    harbor_trial_config.agent.import_path=examples.train_integrations.harbor_skycap.terminus_installed:TerminusInstalled
    harbor_trial_config.agent.kwargs.<Terminus-2's kwargs>   (max_turns, enable_summarize, model_info, llm_kwargs, ...)
    skycap.sandbox_relay.enabled=true   (and its environment class; transport=exec where there are no preview URLs)

Costs of running there: an install per trial (a Python 3.12 environment with LiteLLM, ~200 MB, plus tmux as before),
and the agent's Python sharing the task's CPU and memory. The cluster waits for the runner with a long-poll exec
(``SKYCAP_FAST_EXEC=1`` sends it through the provider's one-call exec; Harbor's own path would poll every second).
"""

import asyncio
import json
import os
import shlex
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import harbor
from harbor.agents.installed.base import BaseInstalledAgent, NonZeroAgentExitCodeError, with_prompt_template
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import ContextLengthExceededError
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

from . import agent_path, terminal_guard, timing

HERE = Path(__file__).resolve().parent
AGENT_DIR = "/installed-agent"
#: What the runner imports besides the Harbor package (the cluster's versions).
REQUIREMENTS = ("litellm=={litellm}", "pydantic=={pydantic}", "tenacity=={tenacity}", "shortuuid", "toml")
UV_VERSION = "0.9.5"
#: The runner's interpreter, as the process list shows it (a link to the environment's python).
RUNNER_NAME = "terminus-agent"
#: Seconds one long-poll for the runner's exit may last.
WAIT_SEC = 45
#: Seconds before asking again after a long-poll that failed.
WAIT_RETRY_SEC = 5
#: Seconds past the agent timeout at which ``run`` gives up by itself (Harbor's timeout normally comes first).
DEADLINE_GRACE_SEC = 30
#: Consecutive failed long-polls that say the sandbox no longer exists (not found / gone / its token rejected)
#: after which ``run`` stops waiting: a deleted sandbox answers that way for ever, and the trial would otherwise
#: sit until the agent timeout. 12 x WAIT_RETRY_SEC is a minute, longer than a provider hiccup of that kind.
GONE_AFTER = 12
#: The long-poll's first words: fast_exec.py sends it through the provider's one-call exec.
WAIT_PREFIX = ": skycap-runner-wait;"
_bundle: Optional[Path] = None


class SandboxGoneError(RuntimeError):
    """The sandbox stopped existing while the runner was being waited for."""


def _sandbox_gone(error: BaseException) -> bool:
    """Whether a failed request says the sandbox itself is gone (HTTP 401 / 404 / 410 from the provider), looking
    through wrappers (a retry library's error, ``raise ... from``)."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        names = {cls.__name__ for cls in type(error).__mro__}
        if names & {"DaytonaNotFoundError", "DaytonaGoneError", "DaytonaAuthenticationError"}:
            return True
        text = str(error).lower()
        if "bearer token is invalid" in text or "sandbox not found" in text or "authentication failed" in text:
            return True
        attempt = getattr(error, "last_attempt", None)  # tenacity.RetryError
        inner = attempt.exception() if attempt is not None and attempt.done() else None
        error = inner or error.__cause__ or error.__context__
    return False


def _installed_version(package: str) -> str:
    from importlib.metadata import version

    return version(package)


def bundle() -> Path:
    """A tarball of what the runner imports from this install: the Harbor package (and its metadata, which its
    ``__init__`` reads), ``terminus_think``, the opt-in modules and the runner. Built once per process."""
    global _bundle
    if _bundle is not None and _bundle.exists():
        return _bundle
    package = Path(harbor.__file__).resolve().parent
    handle, path = tempfile.mkstemp(prefix="skycap-terminus-bundle-", suffix=".tgz")
    os.close(handle)

    def keep(info: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        return None if "__pycache__" in info.name or info.name.endswith(".pyc") else info

    with tarfile.open(path, "w:gz") as tar:
        tar.add(package, arcname="harbor", filter=keep)
        for metadata in package.parent.glob("harbor-*.dist-info"):
            tar.add(metadata, arcname=metadata.name, filter=keep)
        for module in (
            "terminus_think.py",
            "timing.py",
            "wait_watch.py",
            "terminal_guard.py",
            "agent_path.py",
            "terminus_runner.py",
        ):
            tar.add(HERE / module, arcname=module)
    _bundle = Path(path)
    return _bundle


def install_command() -> str:
    """Shell (as root) that unpacks the bundle and makes the runner's Python environment with uv: the image's own
    Python 3.12+ when it has one, else a uv-managed one. Nothing is added to PATH or to shell profiles."""
    requirements = " ".join(
        shlex.quote(
            requirement.format(
                litellm=_installed_version("litellm"),
                pydantic=_installed_version("pydantic"),
                tenacity=_installed_version("tenacity"),
            )
        )
        for requirement in REQUIREMENTS
    )
    installer = f"https://astral.sh/uv/{UV_VERSION}/install.sh"
    return (
        f"cd {AGENT_DIR} && mkdir -p pkgs && tar xzf bundle.tgz -C pkgs && rm bundle.tgz\n"
        "if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then\n"
        "  { (apt-get update && apt-get install -y curl) || apk add --no-cache curl || yum install -y curl"
        " || dnf install -y curl; } >/dev/null 2>&1 || true\n"
        "fi\n"
        # Images with curl but no CA bundle (curl: (77)): the installer's download needs one.
        "if [ ! -s /etc/ssl/certs/ca-certificates.crt ] && command -v apt-get >/dev/null 2>&1; then\n"
        "  (apt-get update && apt-get install -y ca-certificates) >/dev/null 2>&1 || true\n"
        "fi\n"
        f"export UV_UNMANAGED_INSTALL={AGENT_DIR}/bin UV_PYTHON_INSTALL_DIR={AGENT_DIR}/python"
        f" UV_CACHE_DIR={AGENT_DIR}/uv-cache UV_LINK_MODE=copy\n"
        "for attempt in 1 2 3; do\n"
        f"  (command -v curl >/dev/null && curl -LsSf {installer} || wget -qO- {installer}) | sh >/dev/null 2>&1"
        f" && [ -x {AGENT_DIR}/bin/uv ] && break\n"
        "  sleep $((attempt * 5))\n"
        "done\n"
        f"[ -x {AGENT_DIR}/bin/uv ] || {{ echo 'uv install failed'; exit 1; }}\n"
        "for attempt in 1 2 3; do\n"
        f"  {AGENT_DIR}/bin/uv venv -q --allow-existing --python 3.12 {AGENT_DIR}/venv"
        f" && {AGENT_DIR}/bin/uv pip install -q --python {AGENT_DIR}/venv/bin/python {requirements} && break\n"
        "  sleep $((attempt * 5))\n"
        "done\n"
        f"rm -rf {AGENT_DIR}/uv-cache; chmod -R a+rX {AGENT_DIR}\n"
        # The runner runs under a name of its own, so the model's `pkill python3` / `killall python` leaves it alone.
        f"ln -sf python {AGENT_DIR}/venv/bin/{RUNNER_NAME}\n"
        f"PYTHONPATH={AGENT_DIR}/pkgs {AGENT_DIR}/venv/bin/{RUNNER_NAME} -c 'import harbor.agents.terminus_2.terminus_2' && echo RUNNER_READY"
    )


class TerminusInstalled(BaseInstalledAgent):
    #: The runner writes this agent's logs in the sandbox; nothing is added to them on the host afterwards, so Harbor's
    #: upload of the log directory back to the sandbox would only copy it (loop_offload.py, SKYCAP_SKIP_LOG_REUPLOAD).
    LOGS_STAY_IN_SANDBOX = True

    def __init__(self, logs_dir: Path, model_name: Optional[str] = None, version: Optional[str] = None, **kwargs: Any):
        """``kwargs`` are Terminus-2's (``max_turns``, ``enable_summarize``, ``api_base``, ``llm_kwargs``, ...): they
        go to the runner as they are."""
        base = {key: kwargs.pop(key) for key in ("logger", "mcp_servers", "skills_dir", "extra_env") if key in kwargs}
        if base.get("mcp_servers") or base.get("skills_dir"):
            raise ValueError("TerminusInstalled doesn't pass MCP servers or skills to the runner")
        super().__init__(logs_dir=logs_dir, model_name=model_name, version=version or "2.0.0", **base)
        #: The trial's agent timeout, when the generator knows it (``override_timeout_sec``): ``run`` then stops the
        #: runner itself a little after it. Harbor's own timeout is a cancellation, and one that lands while the
        #: event loop is stalled inside a failing provider request has been seen to get lost (a trial ran 90 min).
        timeout = kwargs.pop("agent_timeout_sec", None)
        self._deadline_sec = float(timeout) + DEADLINE_GRACE_SEC if timeout else None
        self._terminus_kwargs = kwargs
        self._record = bool(kwargs.get("record_terminal_session", True))

    @staticmethod
    def name() -> str:
        return "terminus-installed"

    async def install(self, environment: BaseEnvironment) -> None:
        # tmux (and asciinema) as Terminus-2 itself installs them, as root, so the runner needs no root.
        session = TmuxSession(
            session_name="install",
            environment=environment,
            logging_path=EnvironmentPaths.agent_dir / "install.pane",
            local_asciinema_recording_path=None,
            remote_asciinema_recording_path=(EnvironmentPaths.agent_dir / "recording.cast") if self._record else None,
        )
        await session._attempt_tmux_installation()
        await environment.upload_file(bundle(), f"{AGENT_DIR}/bundle.tgz")
        result = await self.exec_as_root(environment, install_command(), timeout_sec=300)
        if "RUNNER_READY" not in (result.stdout or ""):
            raise RuntimeError(f"terminus runner install failed: {self._truncate_output(result.stdout)}")

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        config = {
            "instruction": instruction,
            "model_name": self.model_name,
            "kwargs": self._terminus_kwargs,
            "timing": timing.ENABLED,
            "terminal_guard": terminal_guard.ENABLED,
            "agent_path_drop": agent_path.DROP,
            "agent_path_first": agent_path.FIRST,
        }
        local = self.logs_dir / "runner_config.json"
        local.write_text(json.dumps(config))
        await environment.upload_file(local, f"{AGENT_DIR}/config.json")
        local.unlink()  # it carries the trajectory's key
        # Detached, in its own session, so that the wait below can be many short execs instead of one held open.
        await self.exec_as_agent(
            environment,
            f"rm -f {AGENT_DIR}/exit_code; "
            f"setsid nohup {AGENT_DIR}/venv/bin/{RUNNER_NAME} {AGENT_DIR}/pkgs/terminus_runner.py {AGENT_DIR}/config.json"
            f" > {EnvironmentPaths.agent_dir}/runner.log 2>&1 < /dev/null & echo $! > {AGENT_DIR}/runner.pid",
        )
        wait = (
            f"{WAIT_PREFIX} p=$(cat {AGENT_DIR}/runner.pid); i=0; "
            f"while kill -0 $p 2>/dev/null && [ $i -lt {WAIT_SEC * 4} ]; do sleep 0.25; i=$((i+1)); done; "
            f"if [ -f {AGENT_DIR}/exit_code ]; then echo RUNNER_DONE $(cat {AGENT_DIR}/exit_code); "
            "elif kill -0 $p 2>/dev/null; then echo RUNNER_RUNNING; else echo RUNNER_DONE 137; fi"
        )
        started = time.monotonic()
        gone = 0
        try:
            while True:
                if self._deadline_sec is not None and time.monotonic() - started > self._deadline_sec:
                    raise asyncio.TimeoutError(f"terminus runner still running {self._deadline_sec:.0f} s after start")
                try:
                    result = await environment.exec(command=wait, timeout_sec=WAIT_SEC + 15)
                except Exception as error:  # noqa: BLE001
                    # The agent timeout cancels this task; a cancellation that lands in a failing request can come
                    # out of the provider's client as an ordinary error. Without this check it would be retried
                    # like one and the timeout would never take effect.
                    if asyncio.current_task().cancelling():
                        raise asyncio.CancelledError() from error
                    # A sandbox its task has starved (or a provider hiccup) answers late or not at all. The runner
                    # may well be alive: keep asking until the agent timeout, as the cluster-side loop would.
                    # A sandbox that no longer exists is different: it will never answer.
                    gone = gone + 1 if _sandbox_gone(error) else 0
                    if gone >= GONE_AFTER:
                        raise SandboxGoneError(
                            f"the sandbox is gone ({gone} consecutive answers saying so): {type(error).__name__}: {error}"
                        ) from error
                    self.logger.debug(f"waiting for the terminus runner: {error!r}")
                    await asyncio.sleep(WAIT_RETRY_SEC)
                    continue
                gone = 0
                if asyncio.current_task().cancelling():
                    raise asyncio.CancelledError()
                words = (result.stdout or "").split()
                if words[:1] == ["RUNNER_DONE"]:
                    code = int(words[1]) if len(words) > 1 and words[1].lstrip("-").isdigit() else 1
                    break
        except (asyncio.CancelledError, asyncio.TimeoutError):
            # The agent timeout (Harbor's cancellation, or our own deadline, which Harbor reports the same way): stop
            # the runner (it writes its trajectory and counts on SIGTERM) before the verifier.
            stop = (
                f"p=$(cat {AGENT_DIR}/runner.pid); kill -TERM $p 2>/dev/null; i=0; "
                "while kill -0 $p 2>/dev/null && [ $i -lt 40 ]; do sleep 0.25; i=$((i+1)); done; "
                "kill -KILL $p 2>/dev/null; true"
            )
            try:
                await asyncio.shield(environment.exec(command=stop, timeout_sec=30))
            except BaseException:  # noqa: BLE001 - the timeout is what gets reported
                pass
            raise
        if code == 0:
            return
        outcome = await self._runner_outcome(environment)
        kind, message = outcome.get("exception_type"), outcome.get("exception_message") or ""
        if kind == "ContextLengthExceededError":
            raise ContextLengthExceededError(message)
        log = await environment.exec(command=f"tail -c 2000 {EnvironmentPaths.agent_dir}/runner.log")
        raise NonZeroAgentExitCodeError(f"terminus runner exited {code}: {kind}: {message}\n{log.stdout}")

    async def _runner_outcome(self, environment: BaseEnvironment) -> dict:
        result = await environment.exec(command=f"cat {EnvironmentPaths.agent_dir}/runner_result.json")
        try:
            return json.loads(result.stdout or "{}")
        except ValueError:
            return {}

    def populate_context_post_run(self, context: AgentContext) -> None:
        """The runner's counts, from the agent logs Harbor downloaded."""
        path = self.logs_dir / "runner_result.json"
        if not path.exists():
            return
        try:
            outcome = json.loads(path.read_text())
        except ValueError:
            return
        context.n_input_tokens = outcome.get("n_input_tokens")
        context.n_output_tokens = outcome.get("n_output_tokens")
        context.n_cache_tokens = outcome.get("n_cache_tokens")
        context.cost_usd = outcome.get("cost_usd")
        context.metadata = outcome.get("metadata")
