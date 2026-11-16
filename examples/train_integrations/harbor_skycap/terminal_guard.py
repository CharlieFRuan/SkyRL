"""Keep Ctrl-C working when the terminal's input buffer is full (opt-in, ``SKYCAP_TERMINAL_GUARD=1``).

Terminus-2 types everything into one tmux pane. When a program is running in the foreground and not reading the
terminal (a script in a loop, a server, ``sleep``), what the model types next queues up in the kernel's terminal
input buffer, which holds 4,095 characters. Once a paste fills it (a file written with a heredoc is often larger),
the kernel drops every further key, **Ctrl-C included**: the model can no longer interrupt the program, the screen
never changes again, and the trial runs out its time limit. About 1% of CalibForge trials end this way.

With this, whenever the model sends ``C-c`` the harness also does, from outside the terminal, what Ctrl-C would have
done: it discards the terminal's pending input (``tcflush``, when the image has python3) and sends SIGINT to the
foreground process group of the interactive shell's terminal. When Ctrl-C worked by itself the shell is already in the
foreground again and nothing is signalled.
"""

import asyncio
import os

from harbor.agents.terminus_2.tmux_session import TmuxSession

try:
    from .wait_watch import leaf_shell_snippet
except ImportError:  # inside the sandbox (terminus_runner.py), where these modules are top-level
    from wait_watch import leaf_shell_snippet

ENABLED = os.environ.get("SKYCAP_TERMINAL_GUARD") == "1"
#: tmux key names that interrupt.
INTERRUPT_KEYS = {"C-c", "^C"}

# $leaf: the interactive shell. Its stat gives its process group and its terminal's foreground process group; when
# they differ a job has the terminal. Pending input is flushed first, so the shell doesn't run a half-typed paste.
_INTERRUPT = r"""
s=$(cat /proc/$leaf/stat 2>/dev/null) || { echo GUARD_NA; exit 0; }
s=${s##*) }; set -- $s; pgrp=$3; fg=$6
if [ "$pgrp" = "$fg" ] || [ -z "$fg" ] || [ "$fg" -le 1 ]; then echo GUARD_IDLE; exit 0; fi
tty=$(readlink /proc/$leaf/fd/0 2>/dev/null)
if command -v python3 >/dev/null 2>&1 && [ -c "$tty" ]; then
  python3 -c 'import os, sys, termios; fd = os.open(sys.argv[1], os.O_RDWR | os.O_NOCTTY); termios.tcflush(fd, termios.TCIFLUSH)' "$tty" 2>/dev/null && echo GUARD_FLUSHED
fi
kill -INT -- "-$fg" 2>/dev/null && echo GUARD_SIGINT
"""


def interrupt_command(session: str) -> str:
    return leaf_shell_snippet(session) + _INTERRUPT


def is_interrupt(keys) -> bool:
    """Whether the model is pressing Ctrl-C. ``C-c`` followed by a newline counts: tmux would type that as the
    literal text "C-c", which is never what the model means (it costs small models several turns to notice)."""
    if isinstance(keys, str):
        keys = [keys]
    return bool(keys) and any(key.strip("\r\n") in INTERRUPT_KEYS for key in keys)


def apply() -> None:
    if getattr(TmuxSession, "_skycap_terminal_guard", False):
        return
    original = TmuxSession.send_keys

    async def send_keys(self, keys, *args, **kwargs):
        if not is_interrupt(keys):
            return await original(self, keys, *args, **kwargs)
        # First the real key (it may be all that is needed), then the same from outside the terminal.
        result = await original(self, "C-c", block=False, min_timeout_sec=0.0)
        try:
            await self.environment.exec(command=interrupt_command(self._session_name), user=self._user, timeout_sec=20)
        except Exception as error:  # noqa: BLE001 - the guard must never fail a trial
            self._logger.debug(f"terminal guard: {error!r}")
        wait = kwargs.get("min_timeout_sec", 0.0) or 0.0
        if wait > 0:
            await asyncio.sleep(wait)
        return result

    TmuxSession.send_keys = send_keys
    TmuxSession._skycap_terminal_guard = True


if ENABLED:
    apply()
