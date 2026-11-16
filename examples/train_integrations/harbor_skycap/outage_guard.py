"""A provider outage pauses trial starts instead of failing every trial (opt-in, ``SKYCAP_OUTAGE_GUARD=1``).

Without it, when the sandbox provider refuses to create sandboxes (an outage, or its capacity taken by another
job on the same key), every trial fails both of its attempts within seconds. The fully async trainer keeps groups
with no live trajectory, so it goes on "training" on empty batches at the trial start rate: on 2026-11-14 a
two-hour HTTP 503 outage produced ~15 empty steps and used up a tenth of the epoch's tasks per hour.

With it:
  * an attempt that failed **before the sandbox came up**, with a provider-side error (503 / 502 / 500 "failed to
    create sandbox" / connection errors / rate limit / 401 and 403: the key refused, as on 11-14 10:14 when Dune
    answered "RBAC: access denied" to everything), is an *outage failure*. While the gate is closed it does not
    use up one of the trial's attempts; while it is open a trial gets a few such retries for free
    (``harbor_generator.OUTAGE_FREE_RETRIES``), so one task whose own sandbox cannot be created still fails;
  * ``OutageGate``: after ``threshold`` outage failures within ``window`` seconds, from at least ``min_tasks``
    different tasks (one broken task is not an outage), the gate closes and trial starts wait. Every ``probe_every`` seconds one waiting trial is let through as a canary. If the canary fails the
    same way the gate stays closed; if it has not failed that way ``canary_grace`` seconds after it started (its
    sandbox came up) the gate opens again;
  * a trial that has waited ``max_wait`` seconds in total gives up waiting and its failures count as usual.

The gate only looks at how attempts end; it makes no request of its own.
"""

import asyncio
import os
import time
from collections import deque
from typing import Any, Callable, Optional

from loguru import logger

ENABLED = os.environ.get("SKYCAP_OUTAGE_GUARD") == "1"

#: Exception types (the provider SDK's names, as Harbor reports them) that mean "the provider, not this trial".
OUTAGE_TYPES = frozenset(
    {
        "DaytonaServiceUnavailableError",
        "DaytonaBadGatewayError",
        "DaytonaInternalServerError",
        "DaytonaConnectionError",
        "DaytonaConnectionTimeoutError",
        "DaytonaRateLimitError",
        # The provider refusing the key (seen right after an outage): no trial can do anything about it.
        "DaytonaForbiddenError",
        "DaytonaAuthenticationError",
    }
)


def is_outage_result(result: Any) -> bool:
    """Whether a Harbor ``TrialResult`` is an attempt the provider failed before the sandbox existed: one of
    ``OUTAGE_TYPES``, and the agent's setup never began."""
    info = getattr(result, "exception_info", None)
    if info is None or getattr(info, "exception_type", None) not in OUTAGE_TYPES:
        return False
    return getattr(result, "agent_setup", None) is None and getattr(result, "agent_execution", None) is None


class OutageGate:
    def __init__(
        self,
        threshold: int = 8,
        min_tasks: int = 3,
        window: float = 60.0,
        probe_every: float = 60.0,
        canary_grace: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self.threshold, self.window, self.probe_every, self.canary_grace = threshold, window, probe_every, canary_grace
        self.min_tasks = min_tasks
        self._clock, self._sleep = clock, sleep
        self._failures: deque = deque()
        self._closed_since: Optional[float] = None
        self._next_probe = 0.0
        self._canary_started: Optional[float] = None
        self.closures = 0  # how many times the gate has closed
        self.waited_seconds = 0.0  # total time trials spent waiting at it

    @property
    def closed(self) -> bool:
        return self._closed_since is not None

    async def enter(self) -> bool:
        """Wait until a trial may start. True when this trial goes through a closed gate as the canary."""
        started = self._clock()
        try:
            while True:
                now = self._clock()
                if self._closed_since is None:
                    return False
                if self._canary_started is not None:
                    if now - self._canary_started >= self.canary_grace:
                        self._open(f"a canary trial got past sandbox creation ({self.canary_grace:.0f} s without an outage failure)")
                        return False
                elif now >= self._next_probe:
                    self._canary_started = now
                    self._next_probe = now + self.probe_every
                    return True
                await self._sleep(5.0)
        finally:
            self.waited_seconds += self._clock() - started

    def report(self, outage: bool, canary: bool = False, task: Any = None) -> None:
        """How an attempt that went through ``enter`` ended (``outage``: ``is_outage_result``; ``task``: what the
        attempt was for, so that one task failing many times over does not look like an outage)."""
        now = self._clock()
        if canary:
            self._canary_started = None
            if not outage and self._closed_since is not None:
                self._open("a canary trial finished without an outage failure")
        if not outage:
            return
        self._failures.append((now, task))
        while self._failures and now - self._failures[0][0] > self.window:
            self._failures.popleft()
        tasks = {t for _, t in self._failures}
        # ``task`` not given: every failure counts as a different task.
        distinct = len(self._failures) if tasks == {None} else len(tasks)
        if self._closed_since is None and len(self._failures) >= self.threshold and distinct >= self.min_tasks:
            self._closed_since = now
            self._next_probe = now + self.probe_every
            self._canary_started = None
            self.closures += 1
            logger.warning(
                f"outage guard: {len(self._failures)} trial starts failed at the provider within {self.window:.0f} s; "
                f"pausing trial starts (one canary every {self.probe_every:.0f} s)"
            )

    def _open(self, why: str) -> None:
        logger.warning(f"outage guard: resuming trial starts after {self._clock() - self._closed_since:.0f} s: {why}")
        self._closed_since = None
        self._canary_started = None
        self._failures.clear()
