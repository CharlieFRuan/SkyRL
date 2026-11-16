"""Harbor's directory transfers off the event loop (opt-in, ``SKYCAP_OFFLOAD_TAR=1``).

Harbor's Daytona environment tars and gzips (level 9) a directory before it uploads it, and untars one it has
downloaded, synchronously inside the coroutine (``tar_transfer.pack_dir_to_file`` / ``extract_dir_from_file``). Every
trial of a run shares one event loop, so while one trial's agent logs are packed nothing else moves: no trial starts,
no result is collected, timeouts keep counting. A trial whose terminal recording is 1 GB stopped a whole eval for
376 s; at 1,024 training trials the loop was stalled most of the time (30-160 s at a stretch), which capped the trials
in flight near 700 and produced setup and download timeouts.

With this, both run in a small thread pool (``SKYCAP_TAR_THREADS``, default 8; zlib and file I/O release the GIL) and
the upload is packed at gzip level 1.

``SKYCAP_SKIP_LOG_REUPLOAD=1`` also skips Harbor's upload of the agent's log directory back to the sandbox after the
run (``Trial._upload_agent_logs``) for an agent that says its logs were written there in the first place
(``LOGS_STAY_IN_SANDBOX = True``, e.g. ``TerminusInstalled``): Harbor has just downloaded that directory from the
sandbox, so the upload only copies it back (and doubles its size on the sandbox's disk before the verifier runs).
"""

import inspect
import os
import tarfile
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import harbor.environments.daytona.environment as _daytona_module
from harbor.environments.daytona import DaytonaEnvironment
from harbor.trial.trial import Trial

ENABLED = os.environ.get("SKYCAP_OFFLOAD_TAR") == "1"
SKIP_LOG_REUPLOAD = os.environ.get("SKYCAP_SKIP_LOG_REUPLOAD") == "1"
THREADS = int(os.environ.get("SKYCAP_TAR_THREADS", "8"))

_PACK = "pack_dir_to_file(source_path, archive_path)"
_EXTRACT = "extract_dir_from_file(archive_path, target_dir)"


def pack_dir_fast(source_dir, archive_path) -> None:
    """``tar_transfer.pack_dir_to_file`` at gzip level 1: the same archive layout (rooted at ``.``, links kept)."""
    source_path = Path(source_dir)
    if not source_path.is_dir():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist")
    with tarfile.open(str(archive_path), mode="w:gz", compresslevel=1) as tar:
        tar.add(source_path, arcname=".")


def _remade(function, anchor: str, replacement: str, name: str):
    src = inspect.getsource(inspect.unwrap(function))
    if src.count(anchor) != 1:
        raise RuntimeError(f"can't patch DaytonaEnvironment.{name}: the anchor matches {src.count(anchor)} times")
    namespace: dict = {}
    exec(textwrap.dedent(src.replace(anchor, replacement, 1)), _daytona_module.__dict__, namespace)
    return namespace[name]


def apply() -> None:
    if getattr(DaytonaEnvironment, "_skycap_offload_tar", False):
        return
    _daytona_module._skycap_tar_pool = ThreadPoolExecutor(max_workers=THREADS, thread_name_prefix="skycap-tar")
    _daytona_module._skycap_pack_dir_fast = pack_dir_fast
    run = "await asyncio.get_running_loop().run_in_executor(_skycap_tar_pool, "
    DaytonaEnvironment._sdk_upload_dir = _remade(
        DaytonaEnvironment._sdk_upload_dir,
        _PACK,
        run + "_skycap_pack_dir_fast, source_path, archive_path)",
        "_sdk_upload_dir",
    )
    DaytonaEnvironment._sdk_download_dir = _remade(
        DaytonaEnvironment._sdk_download_dir,
        _EXTRACT,
        run + "extract_dir_from_file, archive_path, target_dir)",
        "_sdk_download_dir",
    )
    DaytonaEnvironment._skycap_offload_tar = True


def apply_skip_log_reupload() -> None:
    if getattr(Trial, "_skycap_skip_log_reupload", False):
        return
    original = Trial._upload_agent_logs

    async def _upload_agent_logs(self) -> None:
        if getattr(self.agent, "LOGS_STAY_IN_SANDBOX", False):
            return
        await original(self)

    Trial._upload_agent_logs = _upload_agent_logs
    Trial._skycap_skip_log_reupload = True


if ENABLED:
    apply()
if SKIP_LOG_REUPLOAD:
    apply_skip_log_reupload()
