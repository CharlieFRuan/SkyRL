"""Daytona sandboxes this run can find and clean up again, without touching anyone else's.

Harbor deletes a trial's sandbox when the trial ends. When the training process dies first, its sandboxes
stay up, counting against an org quota others share, until something deletes them through Daytona's API.
``LabelledDaytonaEnvironment`` tags every sandbox it creates with ``labels`` and gives it a hard
``ttl_minutes``, so this module's ``cleanup`` can delete exactly this run's sandboxes, and Daytona deletes any left over
once their time is up. Use it in place of ``type: daytona``:

    harbor_trial_config.environment.import_path=examples.train_integrations.harbor_skycap.daytona:LabelledDaytonaEnvironment
    harbor_trial_config.environment.kwargs.labels.owner=<you> harbor_trial_config.environment.kwargs.labels.run=<run>
    harbor_trial_config.environment.kwargs.ttl_minutes=180

    uv run --isolated --extra harbor python -m examples.train_integrations.harbor_skycap.daytona list --label owner=<you> --label run=<run>
    uv run --isolated --extra harbor python -m examples.train_integrations.harbor_skycap.daytona cleanup --label owner=<you> --label run=<run>
"""

import argparse
import asyncio
import os
from typing import Any, Dict, Optional

from harbor.environments.daytona import DaytonaEnvironment

#: SKYCAP_CREATE_TIMEOUT_SEC: give up on one sandbox creation after this many seconds, so that Harbor's retry
#: creates another one (unset: Harbor's own limit, the task's ``build_timeout_sec``). For orgs whose sandboxes start
#: from ready snapshots in seconds: on 2026-11-15 one broken Daytona runner took ~3% of our creates and held each
#: for ~900 s before reporting an error, after which the retry (elsewhere) succeeded in seconds; a sandbox in the
#: CREATING state cannot be deleted from outside ("Sandbox state change in progress"). Do not set it where a
#: creation may build or pull an image. The abandoned sandbox ends in the ERROR state and is removed by the label
#: cleanup or its ``ttl_minutes``.
CREATE_TIMEOUT_SEC = float(os.environ.get("SKYCAP_CREATE_TIMEOUT_SEC") or 0) or None


class _ShortCreate:
    """A Daytona client whose ``create`` never waits longer than ``limit`` seconds; everything else is the client."""

    def __init__(self, client: Any, limit: float) -> None:
        self._client, self._limit = client, limit

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    async def create(self, params: Any = None, *, timeout: float = 60, **kwargs: Any) -> Any:
        # The SDK raises DaytonaTimeoutError (a DaytonaError, which Harbor's creation retry retries).
        limit = min(timeout, self._limit) if timeout else self._limit
        return await self._client.create(params=params, timeout=limit, **kwargs)


class LabelledDaytonaEnvironment(DaytonaEnvironment):
    def __init__(
        self, *args: Any, labels: Optional[Dict[str, str]] = None, ttl_minutes: Optional[int] = None, **kwargs: Any
    ) -> None:
        """``labels`` go on every sandbox; ``ttl_minutes`` is how long one may live, whatever happens to us."""
        if not labels:
            raise ValueError("LabelledDaytonaEnvironment needs labels, e.g. {owner: <you>, run: <run name>}")
        super().__init__(*args, **kwargs)
        self._labels = {str(key): str(value) for key, value in labels.items()}
        self._ttl_minutes = ttl_minutes

    async def _create_sandbox(self, params: Any, daytona: Any = None) -> None:
        params.labels = {**(params.labels or {}), **self._labels}
        if self._ttl_minutes is not None:
            params.ttl_minutes = self._ttl_minutes
        if CREATE_TIMEOUT_SEC is not None:
            if daytona is None:
                daytona = await self._client_manager.get_client()
            daytona = _ShortCreate(daytona, CREATE_TIMEOUT_SEC)
        await super()._create_sandbox(params=params, daytona=daytona)


async def _ours(daytona: Any, labels: Dict[str, str]) -> list:
    from daytona import ListSandboxesQuery

    return [sandbox async for sandbox in daytona.list(ListSandboxesQuery(labels=labels))]


async def _main(command: str, labels: Dict[str, str]) -> None:
    from daytona import AsyncDaytona

    async with AsyncDaytona() as daytona:
        sandboxes = await _ours(daytona, labels)
        if command == "list":
            for sandbox in sandboxes:
                print(sandbox.id, sandbox.state, sandbox.labels)
            print(f"{len(sandboxes)} sandboxes labelled {labels}")
            return
        await asyncio.gather(*(daytona.delete(sandbox) for sandbox in sandboxes))
        print(f"deleted {len(sandboxes)} sandboxes labelled {labels}")


def main() -> None:
    parser = argparse.ArgumentParser(description="List or delete the Daytona sandboxes carrying every given label.")
    parser.add_argument("command", choices=["list", "cleanup"])
    parser.add_argument("--label", action="append", required=True, metavar="KEY=VALUE")
    args = parser.parse_args()
    if bad := [item for item in args.label if "=" not in item]:
        parser.error(f"labels are KEY=VALUE, got {bad}")
    labels = dict(item.split("=", 1) for item in args.label)
    if command_is_unscoped(labels):
        parser.error("give at least an owner label and a run label, so cleanup can't reach other runs' sandboxes")
    asyncio.run(_main(args.command, labels))


def command_is_unscoped(labels: Dict[str, str]) -> bool:
    return not {"owner", "run"} <= set(labels)


if __name__ == "__main__":
    main()
