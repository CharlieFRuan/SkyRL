"""Where a Harbor trial's wall time goes: per-trial timing events (opt-in, ``SKYCAP_TIMING=1``).

Each trial gets ``<trial dir>/timing.jsonl``, a line per event with ``k`` (kind), ``t0`` (epoch start) and ``dt``
(seconds):

* ``create``: the provider's sandbox create (image pull or build included); ``env_start`` / ``env_stop`` around it.
* ``exec``: one ``environment.exec``, with the provider API calls it made (``api``: ``[name, seconds]`` each;
  Harbor's Daytona exec is create_session + execute_session_command + get_session_command polls 1 s apart +
  get_session_command_logs), the command's head, its return code and output size. ``xfer``: an upload or download.
* ``keys`` / ``capture`` / ``alive``: Terminus-2's tmux operations (each is one or more ``exec``); ``keys`` carries the
  wait the model asked for (``wait``).
* ``llm``: one model call as the agent sees it, with the usage the server reported.
* ``tools``: one turn's command execution as a whole (keys, waits and the screen capture).

``<trials dir>/../loop_lag.jsonl`` samples how late this process's event loop wakes up (every agent of a
Terminus-2 run is a task on one loop), and the number of trials whose sandbox is up.

Importing this module with ``SKYCAP_TIMING=1`` patches Harbor's Daytona environment, tmux session, chat and
Terminus-2 in place; without it nothing changes.
"""

import asyncio
import contextvars
import functools
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

ENABLED = os.environ.get("SKYCAP_TIMING") == "1"
#: The provider API calls made inside the exec (or transfer) in flight on this task.
_API: contextvars.ContextVar = contextvars.ContextVar("skycap_timing_api", default=None)
#: The running trial's sink, for events raised where no environment is at hand (the chat).
_SINK: contextvars.ContextVar = contextvars.ContextVar("skycap_timing_sink", default=None)
_sinks: Dict[str, "Sink"] = {}
_lag_task: Optional[asyncio.Task] = None
FLUSH_EVERY = 5


class Sink:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a")
        self._pending = 0

    def write(self, kind: str, t0: float, **fields: Any) -> None:
        event = {"k": kind, "t0": round(t0, 3), "dt": round(time.time() - t0, 4), **fields}
        try:
            self._file.write(json.dumps(event, default=str) + "\n")
            self._pending += 1
            if self._pending >= FLUSH_EVERY:
                self.flush()
        except ValueError:  # closed: an event after the environment stopped
            pass

    def flush(self) -> None:
        self._file.flush()
        self._pending = 0

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:  # noqa: BLE001
            pass


def sink_of(environment: Any) -> Optional[Sink]:
    """The sink of ``environment``'s trial, made on first use."""
    try:
        trial_dir = Path(environment.trial_paths.trial_dir)
    except Exception:  # noqa: BLE001 - an environment without trial paths
        return None
    key = str(trial_dir)
    sink = _sinks.get(key)
    if sink is None:
        sink = _sinks[key] = Sink(trial_dir / "timing.jsonl")
        _start_lag_monitor(getattr(environment, "skycap_timing_lag_path", trial_dir.parent.parent / "loop_lag.jsonl"))
    return sink


def _start_lag_monitor(path: Optional[Path], interval: float = 0.2, report_every: float = 10.0) -> None:
    global _lag_task
    if path is None or (_lag_task is not None and not _lag_task.done()):
        return

    async def monitor() -> None:
        lags, last = [], time.time()
        with open(path, "a") as out:
            while True:
                before = time.perf_counter()
                await asyncio.sleep(interval)
                lags.append(time.perf_counter() - before - interval)
                if time.time() - last >= report_every:
                    lags.sort()
                    out.write(
                        json.dumps(
                            {
                                "t": round(time.time(), 1),
                                "pid": os.getpid(),
                                "trials": len(_sinks),
                                "tasks": len(asyncio.all_tasks()),
                                "lag_mean": round(sum(lags) / len(lags), 4),
                                "lag_p50": round(lags[len(lags) // 2], 4),
                                "lag_p99": round(lags[int(len(lags) * 0.99)], 4),
                                "lag_max": round(lags[-1], 4),
                            }
                        )
                        + "\n"
                    )
                    out.flush()
                    lags, last = [], time.time()

    _lag_task = asyncio.get_running_loop().create_task(monitor())


def _api_timed(name: str, function):
    """``function`` (a provider SDK coroutine method), adding its duration to the exec in flight."""

    @functools.wraps(function)
    async def timed(*args, **kwargs):
        calls = _API.get()
        if calls is None:
            return await function(*args, **kwargs)
        before = time.perf_counter()
        try:
            return await function(*args, **kwargs)
        finally:
            calls.append([name, round(time.perf_counter() - before, 4)])

    return timed


def _env_timed(kind: str, function, describe):
    """``function`` (an environment coroutine method) as one ``kind`` event with the API calls made inside it."""

    @functools.wraps(function)
    async def timed(self, *args, **kwargs):
        sink = sink_of(self)
        if sink is None:
            return await function(self, *args, **kwargs)
        calls: list = []
        token = _API.set(calls)
        t0, fields = time.time(), {}
        try:
            result = await function(self, *args, **kwargs)
            fields = describe(result, *args, **kwargs)
            return result
        except BaseException as error:
            fields = {**describe(None, *args, **kwargs), "error": type(error).__name__}
            raise
        finally:
            _API.reset(token)
            sink.write(kind, t0, api=calls, **fields)

    return timed


def _event(kind: str, function, environment_of, describe=lambda result, *a, **k: {}):
    """``function`` as one ``kind`` event in the sink of ``environment_of(self)``'s trial."""

    @functools.wraps(function)
    async def timed(self, *args, **kwargs):
        environment = environment_of(self, *args, **kwargs)
        sink = sink_of(environment) if environment is not None else _SINK.get()
        if sink is None:
            return await function(self, *args, **kwargs)
        t0, fields = time.time(), {}
        try:
            result = await function(self, *args, **kwargs)
            fields = describe(result, *args, **kwargs)
            return result
        except BaseException as error:
            fields = {"error": type(error).__name__}
            raise
        finally:
            sink.write(kind, t0, **fields)

    return timed


def _describe_exec(result, command=None, *args, **kwargs):
    command = command if command is not None else kwargs.get("command", "")
    fields = {"cmd": str(command)[:100], "to": kwargs.get("timeout_sec")}
    if result is not None:
        fields["rc"] = result.return_code
        fields["out"] = len(result.stdout or "") + len(result.stderr or "")
    return fields


def _describe_transfer(name):
    def describe(result, *args, **kwargs):
        path = args[0] if args else kwargs.get("source_path", kwargs.get("source_dir", ""))
        return {"op": name, "path": str(path)[:100]}

    return describe


def _describe_keys(result, keys=None, block=False, min_timeout_sec=0.0, max_timeout_sec=180.0, **kwargs):
    size = len(keys) if isinstance(keys, str) else sum(len(key) for key in keys or [])
    return {"wait": min_timeout_sec, "block": block, "chars": size}


def _describe_llm(result, *args, **kwargs):
    usage = getattr(result, "usage", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }


def _describe_tools(result, commands=None, *args, **kwargs):
    commands = commands or []
    return {"n": len(commands), "wait": round(sum(command.duration_sec for command in commands), 3)}


def apply() -> None:
    """Time Harbor's Daytona environment (this process's sandboxes) and Terminus-2."""
    from daytona._async.filesystem import AsyncFileSystem
    from daytona._async.process import AsyncProcess
    from harbor.environments.daytona import DaytonaEnvironment

    if getattr(DaytonaEnvironment, "_skycap_timing", False):
        return
    apply_agent()
    for name in (
        "create_session",
        "execute_session_command",
        "get_session_command",
        "get_session_command_logs",
        "exec",
    ):
        setattr(AsyncProcess, name, _api_timed(name, getattr(AsyncProcess, name)))
    for name in ("upload_file", "upload_files", "download_file", "download_files"):
        if hasattr(AsyncFileSystem, name):
            setattr(AsyncFileSystem, name, _api_timed(name, getattr(AsyncFileSystem, name)))

    DaytonaEnvironment.exec = _env_timed("exec", DaytonaEnvironment.exec, _describe_exec)
    for name in ("upload_file", "upload_dir", "download_file", "download_dir"):
        setattr(DaytonaEnvironment, name, _env_timed("xfer", getattr(DaytonaEnvironment, name), _describe_transfer(name)))
    DaytonaEnvironment._create_sandbox = _env_timed(
        "create", DaytonaEnvironment._create_sandbox, lambda result, *a, **k: {}
    )
    DaytonaEnvironment.start = _env_timed("env_start", DaytonaEnvironment.start, lambda result, *a, **k: {})
    original_stop = _env_timed("env_stop", DaytonaEnvironment.stop, lambda result, *a, **k: {})

    @functools.wraps(original_stop)
    async def stop(self, *args, **kwargs):
        try:
            return await original_stop(self, *args, **kwargs)
        finally:
            # The trial's last environment call: an event after it reopens the file.
            sink = _sinks.pop(str(Path(self.trial_paths.trial_dir)), None)
            if sink is not None:
                sink.close()

    DaytonaEnvironment.stop = stop
    DaytonaEnvironment._skycap_timing = True


def apply_agent() -> None:
    """Time Terminus-2 alone: its tmux operations, model calls and turns. Its environment only needs
    ``trial_paths.trial_dir`` (terminus_runner.py uses this inside the sandbox, with its own environment)."""
    from harbor.agents.terminus_2.terminus_2 import Terminus2
    from harbor.agents.terminus_2.tmux_session import TmuxSession
    from harbor.llms.chat import Chat

    if getattr(TmuxSession, "_skycap_timing", False):
        return
    of_session = lambda self, *args, **kwargs: self.environment  # noqa: E731
    TmuxSession.send_keys = _event("keys", TmuxSession.send_keys, of_session, _describe_keys)
    TmuxSession.get_incremental_output = _event(
        "capture",
        TmuxSession.get_incremental_output,
        of_session,
        lambda result, *a, **k: {"out": len(result or "")},
    )
    TmuxSession.is_session_alive = _event("alive", TmuxSession.is_session_alive, of_session)
    TmuxSession.start = _event("tmux_start", TmuxSession.start, of_session)

    Chat.chat = _event("llm", Chat.chat, lambda self, *a, **k: None, _describe_llm)
    Terminus2._execute_commands = _event(
        "tools",
        Terminus2._execute_commands,
        lambda self, *a, **k: getattr(self._session, "environment", None),
        _describe_tools,
    )
    original_run = Terminus2.run

    @functools.wraps(original_run)
    async def run(self, instruction, environment, context):
        sink = sink_of(environment)
        token = _SINK.set(sink)
        t0 = time.time()
        try:
            return await original_run(self, instruction, environment, context)
        finally:
            _SINK.reset(token)
            if sink is not None:
                sink.write("agent_run", t0)
                sink.flush()

    Terminus2.run = run
    TmuxSession._skycap_timing = True


if ENABLED:
    apply()
