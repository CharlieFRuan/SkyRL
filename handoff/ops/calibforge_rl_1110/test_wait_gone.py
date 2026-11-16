"""The in-sandbox agent's wait loop gives up on a sandbox that no longer exists (PLAN.md fix item 10).

Run from a checkout root:  python test_wait_gone.py   (no sandbox, no model: a fake environment)."""
import asyncio, sys, tempfile
from pathlib import Path

sys.path.insert(0, ".")
from daytona.common.errors import DaytonaAuthenticationError, DaytonaConnectionError, DaytonaNotFoundError
from harbor.environments.base import ExecResult

from examples.train_integrations.harbor_skycap import terminus_installed as ti

ti.WAIT_RETRY_SEC = 0.01


class FakeEnvironment:
    def __init__(self, script):
        self.script, self.waits = list(script), 0

    async def upload_file(self, *args, **kwargs):
        return None

    async def exec(self, command, **kwargs):
        if not command.startswith(ti.WAIT_PREFIX):
            return ExecResult(stdout="", stderr="", return_code=0)
        self.waits += 1
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, BaseException):
            raise step
        return ExecResult(stdout=step, stderr="", return_code=0)


async def run(script, timeout=None):
    with tempfile.TemporaryDirectory() as d:
        agent = ti.TerminusInstalled(logs_dir=Path(d), model_name="m", agent_timeout_sec=timeout)
        env = FakeEnvironment(script)
        try:
            await agent.run("do it", env, None)
            return "returned", env.waits
        except BaseException as error:  # noqa: BLE001
            return type(error).__name__, env.waits


async def main():
    gone, hiccup = DaytonaAuthenticationError("unauthorized: authentication failed: Bearer token is invalid"), DaytonaConnectionError("Server disconnected")
    # 1. A deleted sandbox: the same "gone" answer for ever -> SandboxGoneError after GONE_AFTER polls.
    assert await run([gone]) == ("SandboxGoneError", ti.GONE_AFTER), await run([gone])
    assert (await run([DaytonaNotFoundError("sandbox not found")]))[0] == "SandboxGoneError"
    # 2. A wrapped error (raise ... from) is recognised too.
    wrapped = RuntimeError("exec failed"); wrapped.__cause__ = gone
    assert (await run([wrapped]))[0] == "SandboxGoneError"
    # 3. Fewer than GONE_AFTER in a row, broken by another kind of failure, then the runner finishes: no give-up.
    script = [gone] * (ti.GONE_AFTER - 1) + [hiccup] + [gone] * (ti.GONE_AFTER - 1) + ["RUNNER_RUNNING", "RUNNER_DONE 0"]
    assert await run(script) == ("returned", len(script)), await run(script)
    # 4. Provider hiccups alone never trigger it: still waiting when our own short deadline passes.
    ti.DEADLINE_GRACE_SEC = 0
    kind, waits = await run([hiccup], timeout=0.5)
    assert kind == "TimeoutError" and waits > ti.GONE_AFTER, (kind, waits)
    print("OK: gone -> SandboxGoneError after", ti.GONE_AFTER, "polls; hiccups and short runs of 'gone' keep waiting")


asyncio.run(main())
