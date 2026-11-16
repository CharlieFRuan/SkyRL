"""outage_guard.OutageGate and its use in HarborGenerator._trial (no provider, no model: fakes).

Run from a checkout root:  python test_outage_guard.py"""
import asyncio, sys, time, types

sys.path.insert(0, ".")
from examples.train_integrations.harbor_skycap import harbor_generator as hg
from examples.train_integrations.harbor_skycap import outage_guard as og
from examples.train_integrations.harbor_skycap.compose import TrialOutcome
from examples.train_integrations.harbor.trial_metrics import TrialAttempts

hg.OUTAGE_RETRY_SEC = 0.01
fast_sleep = lambda seconds: asyncio.sleep(0.01)


def result(kind=None, agent_setup=None):
    info = types.SimpleNamespace(exception_type=kind) if kind else None
    return types.SimpleNamespace(exception_info=info, agent_setup=agent_setup, agent_execution=None)


def test_is_outage():
    assert og.is_outage_result(result("DaytonaServiceUnavailableError"))
    assert og.is_outage_result(result("DaytonaInternalServerError"))
    assert og.is_outage_result(result("DaytonaForbiddenError")), "403 RBAC: access denied at creation (11-14 10:14)"
    assert not og.is_outage_result(result("DaytonaServiceUnavailableError", agent_setup=object())), "sandbox was up"
    assert not og.is_outage_result(result("AgentTimeoutError")) and not og.is_outage_result(result(None))


async def test_gate():
    gate = og.OutageGate(threshold=3, window=5, probe_every=0.2, canary_grace=0.15, sleep=fast_sleep)
    assert await gate.enter() is False
    gate.report(True); gate.report(True); assert not gate.closed
    gate.report(False); assert not gate.closed
    gate.report(True); assert gate.closed and gate.closures == 1
    t0 = time.monotonic()
    assert await gate.enter() is True and time.monotonic() - t0 >= 0.18, "first canary only after probe_every"
    waiter = asyncio.ensure_future(gate.enter())
    await asyncio.sleep(0.05); assert not waiter.done(), "others wait while a canary is out"
    gate.report(True, canary=True); assert gate.closed, "a failed canary keeps it closed"
    assert await waiter is True, "the waiter becomes the next canary after probe_every"
    others = [asyncio.ensure_future(gate.enter()) for _ in range(5)]
    assert await asyncio.gather(*others) == [False] * 5 and not gate.closed, "no failure within canary_grace opens it"
    gate.report(False, canary=True)  # the canary finishing later changes nothing
    assert not gate.closed and gate.waited_seconds > 0.3
    # One task failing over and over is not an outage; three different tasks are.
    gate = og.OutageGate(threshold=8, min_tasks=3, window=5, sleep=fast_sleep)
    for _ in range(40):
        gate.report(True, task="broken-task")
    assert not gate.closed
    gate.report(True, task="b"); assert not gate.closed
    gate.report(True, task="c"); assert gate.closed


class FakeGenerator:
    """Just what _trial touches."""
    train_paths = None

    def __init__(self, gate, script):
        self._outage_gate, self.script, self.calls, self.attempt_numbers = gate, script, 0, []

    async def _attempt(self, prompt, trajectory_id, cache_salt, step, phase, attempt, attempts, info=None):
        self.calls += 1
        self.attempt_numbers.append(attempt)
        kind = self.script(self.calls)
        await asyncio.sleep(0.005)
        if kind == "outage":
            info["outage"] = True
            return TrialOutcome(trajectory_id=trajectory_id, stop_reason="error")
        if kind == "error":
            info["outage"] = False
            return TrialOutcome(trajectory_id=trajectory_id, stop_reason="error")
        if kind == "raise":
            raise RuntimeError("boom")
        info["outage"] = False
        return TrialOutcome(trajectory_id=trajectory_id, samples=[], reward=1.0, stop_reason="complete")


async def trial(gen, i=0):
    return await hg.HarborSkycapGenerator._trial(gen, "task", f"t{i}", None, 1, "train", TrialAttempts())


async def test_trial():
    # No guard: two outage failures use both attempts, as before.
    gen = FakeGenerator(None, lambda n: "outage")
    assert (await trial(gen)).stop_reason == "error" and gen.calls == hg.MAX_NUM_RETRIES_PER_TRIAL == 2
    # Guard, one trial: outage failures are not attempts; it succeeds when the provider is back, still on attempt 0.
    gate = og.OutageGate(threshold=3, window=5, probe_every=0.1, canary_grace=0.08, sleep=fast_sleep)
    gen = FakeGenerator(gate, lambda n: "outage" if n <= 6 else "ok")
    out = await trial(gen)
    assert out.stop_reason == "complete" and gen.calls == 7 and set(gen.attempt_numbers) == {0}, (gen.calls, gen.attempt_numbers)
    assert gate.closures == 1 and not gate.closed
    # Guard: ordinary failures still use attempts (error, then raise -> masked after 2).
    gen = FakeGenerator(og.OutageGate(sleep=fast_sleep), lambda n: "error" if n == 1 else "raise")
    assert (await trial(gen)).stop_reason == "error" and gen.attempt_numbers == [0, 1]
    # Guard, 200 trials during an outage that ends after 0.6 s: all finish, and the number of starts that hit
    # the dead provider stays near the threshold plus one canary per probe, not 200 x many.
    gate = og.OutageGate(threshold=8, window=5, probe_every=0.1, canary_grace=0.08, sleep=fast_sleep)
    t0 = time.monotonic(); failed = [0]
    def script(n):
        if time.monotonic() - t0 < 0.6:
            failed[0] += 1; return "outage"
        return "ok"
    gen = FakeGenerator(gate, script)
    outs = await asyncio.gather(*(trial(gen, i) for i in range(200)))
    assert all(o.stop_reason == "complete" for o in outs), "every trial trains once the provider is back"
    assert failed[0] < 260, failed[0]   # 200 already under way when it closed + a few canaries; unguarded: 400 and all masked
    print("starts that hit the dead provider:", failed[0], "| closures:", gate.closures, "| waited trial-seconds:", round(gate.waited_seconds, 1))
    # No outage declared (gate open): a trial gets OUTAGE_FREE_RETRIES uncounted retries, then its two attempts.
    gen = FakeGenerator(og.OutageGate(threshold=99, sleep=fast_sleep), lambda n: "outage")
    assert (await trial(gen)).stop_reason == "error" and gen.calls == hg.OUTAGE_FREE_RETRIES + 2, gen.calls
    assert gen.attempt_numbers == [0] * (hg.OUTAGE_FREE_RETRIES + 1) + [1]
    # A group of 16 trials of ONE task whose sandbox cannot be created: no gate closure, all masked, others unaffected.
    gate = og.OutageGate(threshold=8, min_tasks=3, window=5, probe_every=0.1, canary_grace=0.08, sleep=fast_sleep)
    gen = FakeGenerator(gate, lambda n: "outage")
    ids = [types.SimpleNamespace(instance_id="bad", repetition_id=i) for i in range(16)]
    outs = await asyncio.gather(*(hg.HarborSkycapGenerator._trial(gen, "task", t, None, 1, "train", TrialAttempts()) for t in ids))
    assert all(o.stop_reason == "error" for o in outs) and gate.closures == 0 and gen.calls == 16 * (hg.OUTAGE_FREE_RETRIES + 2)
    # The wait is bounded: past the deadline an outage failure counts as an attempt.
    hg.OUTAGE_MAX_WAIT_SEC, saved = 0.0, hg.OUTAGE_MAX_WAIT_SEC
    gen = FakeGenerator(og.OutageGate(threshold=99, sleep=fast_sleep), lambda n: "outage")
    assert (await trial(gen)).stop_reason == "error" and gen.calls == 2
    hg.OUTAGE_MAX_WAIT_SEC = saved


test_is_outage()
asyncio.run(test_gate())
asyncio.run(test_trial())
print("OK")
