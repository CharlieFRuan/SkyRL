"""Tasks whose image cannot be used as it is: start from a shared base snapshot and replay the task's Dockerfile.

For providers / organizations that allow neither one image per task nor many snapshots (a shared Daytona org:
declarative builds refused, a quota of 40 snapshots, "about 5 per RL source"). Each task then carries, in its
``environment/`` directory (written by ``ops/calibforge_rl_1110/make_daytona_tasks.py``):

    cf_snapshot   the name of the base snapshot to start from (the task's ``FROM`` image at one fixed size)
    setup.sh      the rest of its Dockerfile as a shell script: RUN steps, COPY from the build context, WORKDIR, ENV

``SetupRelayDaytonaEnvironment`` creates the sandbox from that snapshot, uploads ``environment/`` as the build
context, runs ``setup.sh`` as root, and then behaves like the image would: commands run in the Dockerfile's final
``WORKDIR`` and see its ``ENV``. The relay stub starts after the setup, as it does after an image start.

    harbor_trial_config.environment.import_path=examples.train_integrations.harbor_skycap.daytona_setup:SetupRelayDaytonaEnvironment

What differs from a built image: the installs run at every sandbox start (seconds to minutes, and they need the
network), and package versions are whatever the mirrors serve that day.
"""

import time
from typing import Any, Dict, Optional

from .sandbox_relay.environment import RelayDaytonaEnvironment

#: Where the build context goes in the sandbox; removed once the setup has run.
BUILD_DIR = "/var/lib/.cf_build"


class TaskSetupError(RuntimeError):
    """The task's setup script (its Dockerfile steps) failed in the sandbox."""


def parse_setup_output(output: str) -> Dict[str, Any]:
    """``exit``, ``workdir``, ``env`` and ``log`` from what the setup command printed."""
    head, _, log = (output + "\n").partition("\nCF_LOG\n")
    head, _, env_text = (head + "\n").partition("\nCF_ENV\n")
    lines = head.splitlines()
    code = next((line.split("=", 1)[1] for line in lines if line.startswith("CF_SETUP_EXIT=")), "")
    workdir = next((line.split("=", 1)[1] for line in lines if line.startswith("CF_WORKDIR=")), "")
    env: Dict[str, str] = {}
    for line in env_text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key:
            env[key] = value
    return {"exit": int(code) if code.lstrip("-").isdigit() else None, "workdir": workdir or None, "env": env, "log": log}


class SetupRelayDaytonaEnvironment(RelayDaytonaEnvironment):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        marker = self.environment_dir / "cf_snapshot"
        if not marker.exists() or not (self.environment_dir / "setup.sh").exists():
            raise ValueError(f"{self.environment_dir} has no cf_snapshot / setup.sh (make_daytona_tasks.py writes them)")
        self._cf_snapshot = marker.read_text().strip()
        self._cf_workdir: Optional[str] = None
        self.cf_setup_seconds: Optional[float] = None

    async def _resolve_start_sandbox_params(self, daytona: Any, resources: Any, *, force_build: bool) -> Any:
        # Always the task's base snapshot (its size is the snapshot's); no lookup request per trial.
        return self._snapshot_sandbox_params(self._cf_snapshot)

    async def start(self, force_build: bool) -> None:
        # RelayDaytonaEnvironment.start is "sandbox, then stub"; the task's setup goes in between.
        await super(RelayDaytonaEnvironment, self).start(force_build)
        await self._task_setup()
        if self._relay:
            await self._start_relay()

    async def exec(
        self,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_sec: Optional[int] = None,
        user: Any = None,
    ):
        # The Dockerfile's final WORKDIR, as a container started from the image would have it.
        return await super().exec(command, cwd=cwd or self._cf_workdir, env=env, timeout_sec=timeout_sec, user=user)

    async def _task_setup(self) -> None:
        started = time.monotonic()
        await self.upload_dir(self.environment_dir, f"{BUILD_DIR}/ctx")
        timeout = int(getattr(self.task_env_config, "build_timeout_sec", None) or 600)
        command = (
            f"cd / && CF_BUILD={BUILD_DIR} bash {BUILD_DIR}/ctx/setup.sh > {BUILD_DIR}/setup.log 2>&1; rc=$?; "
            f"echo CF_SETUP_EXIT=$rc; echo CF_WORKDIR=$(cat {BUILD_DIR}/workdir 2>/dev/null); "
            f"echo CF_ENV; cat {BUILD_DIR}/env.resolved 2>/dev/null; echo CF_LOG; tail -c 2500 {BUILD_DIR}/setup.log; "
            f"if [ $rc -eq 0 ]; then rm -rf {BUILD_DIR}; fi"
        )
        # As the sandbox's own user (root in our base snapshots), NOT user="root": Harbor switches user with `su`,
        # which resets PATH, and the base image's PATH (e.g. /opt/verifier/bin first) is what the Dockerfile's
        # `pip` / `uv pip install` steps were written against (seen 11-14: "No virtual environment found").
        result = await self.exec(command, cwd="/", timeout_sec=timeout)
        parsed = parse_setup_output(result.stdout or "")
        self.cf_setup_seconds = time.monotonic() - started
        if parsed["exit"] != 0:
            raise TaskSetupError(
                f"task setup failed (exit {parsed['exit']}) after {self.cf_setup_seconds:.0f} s on {self._cf_snapshot}: "
                f"{(parsed['log'] or result.stdout or '')[-1500:]}"
            )
        self._cf_workdir = parsed["workdir"]
        if parsed["env"]:
            self._persistent_env = {**self._persistent_env, **parsed["env"]}
        self.logger.info(
            "task setup on %s took %.1f s (workdir %s, %d env)", self._cf_snapshot, self.cf_setup_seconds, self._cf_workdir, len(parsed["env"])
        )
