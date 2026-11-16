"""Fewer provider round trips per Terminus-2 turn on Daytona-API sandboxes (opt-in, ``SKYCAP_FAST_EXEC=1``).

Harbor's Daytona ``exec`` makes four API calls in a row for every command (``create_session``,
``execute_session_command``, ``get_session_command``, ``get_session_command_logs``), and sleeps 1 s between polls of a
command that hasn't exited by the first one. From a cluster ~0.27 s away from the provider's API that is ~1.06 s per
``exec``, and Terminus-2 makes one for its liveness check, one per command it types and one or two to read the screen:
3-5 s of every turn, out of each task's own agent timeout.

With this, tmux's instant control commands (``send-keys``, ``capture-pane``, ``has-session``) go through the
provider's one-call ``process.exec`` instead (~0.27 s). Everything else (installs, the verifier, anything that may run
long or leave processes behind) keeps Harbor's session path. The command line built is Harbor's own
(``_sandbox_exec``: shell, env, timeout, cwd, user).
"""

import os
import shlex

from harbor.environments.base import ExecResult
from harbor.environments.daytona import DaytonaEnvironment

ENABLED = os.environ.get("SKYCAP_FAST_EXEC") == "1"
#: Commands that return at once and start nothing that outlives them, and two bounded long-polls: wait_watch.py's
#: watch (at most the wait it replaces) and terminus_installed.py's wait for its runner.
FAST_PREFIXES = (
    "tmux send-keys",
    "tmux capture-pane",
    "tmux has-session",
    ": skycap-wait-watch;",
    ": skycap-runner-wait;",
)
#: Seconds the one-call exec may take before the provider gives up (these commands take milliseconds).
FAST_TIMEOUT_SEC = 60


def apply() -> None:
    if getattr(DaytonaEnvironment, "_skycap_fast_exec", False):
        return
    original = DaytonaEnvironment._sandbox_exec

    async def _sandbox_exec(self, command, cwd=None, env=None, timeout_sec=None, shell="bash -c", user=None):
        if not command.startswith(FAST_PREFIXES):
            return await original(self, command, cwd=cwd, env=env, timeout_sec=timeout_sec, shell=shell, user=user)
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please build the environment first.")
        wrapped = f"{shell} {shlex.quote(command)}"
        if env:
            wrapped = f"env {' '.join(f'{key}={shlex.quote(value)}' for key, value in env.items())} {wrapped}"
        if timeout_sec:
            wrapped = f"timeout {timeout_sec} {wrapped}"
        if cwd:
            wrapped = f"cd {cwd} && {wrapped}"
        if user is not None:
            user_arg = f"$(getent passwd {user} | cut -d: -f1)" if isinstance(user, int) else shlex.quote(user)
            wrapped = f"su {user_arg} -s /bin/bash -c {shlex.quote(wrapped)}"
        response = await self._sandbox.process.exec(wrapped, timeout=timeout_sec or FAST_TIMEOUT_SEC)
        return ExecResult(stdout=response.result or "", stderr="", return_code=int(response.exit_code))

    DaytonaEnvironment._sandbox_exec = _sandbox_exec
    DaytonaEnvironment._skycap_fast_exec = True


if ENABLED:
    apply()
