"""Is the command still running while Terminus-2 waits for it? (opt-in, ``SKYCAP_WAIT_WATCH=measure`` or ``return``)

Terminus-2 types a command and then sleeps for the ``duration`` the model asked for (up to 60 s), whether or not the
command has finished. With this, a wait with at least ``MIN_WATCH_SEC`` left is spent in ONE exec that watches the
sandbox's interactive shell from inside the sandbox and returns when the shell is back at its prompt (it is its
terminal's foreground process group again and has no child processes left, so background jobs count as running), or
when the wait is over:

* ``measure``: the wait still lasts as long as the model asked; the moment the shell went idle is recorded (a
  ``watch`` event in the trial's ``timing.jsonl``, with ``SKYCAP_TIMING=1``). Nothing the agent sees changes.
* ``return``: the wait ends when the shell goes idle. The agent sees the same screen sooner. A wait on something the
  shell doesn't own (a daemon, a remote job) ends early too, and the model has to ask again.

Anything unexpected (no ``/proc``, no GNU ``date``, an exec error) falls back to the plain sleep.
"""

import asyncio
import os
import shlex
import time

from harbor.agents.terminus_2.tmux_session import TmuxSession

try:
    from . import timing
except ImportError:  # inside the sandbox (terminus_runner.py), where these modules are top-level
    import timing

MODE = os.environ.get("SKYCAP_WAIT_WATCH", "")
#: Waits with less than this left after the keys were sent are slept, not watched.
MIN_WATCH_SEC = 1.0
#: The watch gives up this long before the wait is over, so that its reply is back in time.
MARGIN_SEC = 0.4
#: The command's first words: fast_exec sends it through the provider's one-call exec.
WATCH_PREFIX = ": skycap-wait-watch;"

# The interactive shell: the pane's shell, or, when Terminus-2 records the session, the shell asciinema runs inside it
# (pane shell -> asciinema -> [sh -c ->] shell; asciinema's own helper processes are skipped).
# Idle: two samples 0.15 s apart with the shell in its terminal's foreground process group and childless.
_LEAF = r"""
S=%(session)s; leaf=$(cat /tmp/.skycap_leaf 2>/dev/null)
if [ -z "$leaf" ] || [ ! -d "/proc/$leaf" ]; then
  leaf=$(tmux display-message -p -t "$S" '#{pane_pid}') || { echo NA; exit 0; }
  cmd() { tr '\0' ' ' < /proc/$1/cmdline 2>/dev/null; }
  kids() { cat /proc/$1/task/*/children 2>/dev/null; }
  for c in $(kids $leaf); do
    case "$(cmd $c)" in *asciinema*)
      for g in $(kids $c); do case "$(cmd $g)" in *asciinema*) ;; *) leaf=$g; break;; esac; done;;
    esac
  done
  case "$(cmd $leaf)" in "sh -c "*) for g in $(kids $leaf); do leaf=$g; break; done;; esac
  echo "$leaf" > /tmp/.skycap_leaf 2>/dev/null
fi
"""
_WATCH = r"""
idle() {
  s=$(cat /proc/$leaf/stat 2>/dev/null) || return 1
  s=${s##*) }; set -- $s
  [ "$3" = "$6" ] && [ -z "$(cat /proc/$leaf/task/$leaf/children 2>/dev/null)" ]
}
start=$(date +%%s%%N); case "$start" in *N|'') echo NA; exit 0;; esac
end=$(( start + %(budget_ns)d )); n=0
while [ "$(date +%%s%%N)" -lt "$end" ]; do
  if idle; then n=$((n+1)); if [ $n -ge 2 ]; then echo "IDLE $(( ($(date +%%s%%N) - start) / 1000000 ))"; exit 0; fi; else n=0; fi
  sleep 0.15
done
echo BUSY
"""


def leaf_shell_snippet(session: str) -> str:
    """Shell that sets ``$leaf`` to the pid of the session's interactive shell (see above)."""
    return _LEAF % {"session": shlex.quote(session)}


def watch_command(session: str, budget_sec: float) -> str:
    return WATCH_PREFIX + leaf_shell_snippet(session) + _WATCH % {"budget_ns": int(budget_sec * 1e9)}


def apply() -> None:
    if getattr(TmuxSession, "_skycap_wait_watch", False):
        return

    async def _send_non_blocking_keys(self, keys, min_timeout_sec):
        start = time.time()
        for command in self._tmux_send_keys(keys):
            result = await self.environment.exec(command=command, user=self._user)
            if result.return_code != 0:
                raise RuntimeError(
                    f"{self.environment.session_id}: failed to send non-blocking keys: "
                    f"command={command!r:.100}, return_code={result.return_code}, "
                    f"stderr={result.stderr!r}, stdout={result.stdout!r}"
                )
        remaining = min_timeout_sec - (time.time() - start)
        if remaining < MIN_WATCH_SEC:
            if remaining > 0:
                await asyncio.sleep(remaining)
            return
        budget = remaining - MARGIN_SEC
        state, idle_after = "ERROR", None
        watch_start = time.time()
        try:
            result = await self.environment.exec(
                command=watch_command(self._session_name, budget), user=self._user, timeout_sec=int(budget) + 10
            )
            words = (result.stdout or "").split()
            state = words[0] if words and words[0] in ("IDLE", "BUSY", "NA") else "ERROR"
            if state == "IDLE":
                idle_after = round(watch_start - start + int(words[1]) / 1000, 3)
        except Exception as error:  # noqa: BLE001 - the watch must never fail a trial
            state = f"ERROR:{type(error).__name__}"
        sink = timing.sink_of(self.environment) if timing.ENABLED else None
        if sink is not None:
            sink.write("watch", start, wait=min_timeout_sec, state=state, idle_after=idle_after, mode=MODE)
        if state == "IDLE" and MODE == "return":
            return
        remaining = min_timeout_sec - (time.time() - start)
        if remaining > 0:
            await asyncio.sleep(remaining)

    TmuxSession._send_non_blocking_keys = _send_non_blocking_keys
    TmuxSession._skycap_wait_watch = True


if MODE in ("measure", "return"):
    apply()
