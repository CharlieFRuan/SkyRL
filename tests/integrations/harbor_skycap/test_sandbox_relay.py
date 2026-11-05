"""The sandbox relay end to end on one machine: an agent calls the stub, the manager polls it and answers
through a stand-in for skycap, and the stats split the agent's wait into skycap's time and the relay's."""

import asyncio
import hashlib
import json
import threading
import time
import urllib.request

import aiohttp
import pytest
from aiohttp import web

from examples.train_integrations.harbor_skycap.sandbox_relay import stub
from examples.train_integrations.harbor_skycap.sandbox_relay.manager import SandboxRelayManager

SECRET = "s3cret"
DELAY = 0.3  # skycap's time per call


@pytest.fixture
def sandbox():
    """A stub on a free port, as it runs in a sandbox."""
    relay = stub.Relay()
    server = stub.ThreadingServer(("127.0.0.1", 0), stub.make_handler(relay, SECRET.encode(), 30.0))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


async def _skycap() -> tuple:
    async def answer(request: web.Request) -> web.Response:
        await asyncio.sleep(DELAY)
        raw = await request.read()
        return web.json_response(
            {"path": request.path, "body": json.loads(raw), "sha": hashlib.sha256(raw).hexdigest(), "len": len(raw)}
        )

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", answer)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


def _agent_call(base: str, n: int, history: str = "") -> dict:
    request = urllib.request.Request(
        f"{base}/t/abc/v1/chat/completions",
        data=json.dumps({"messages": history, "n": n}).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(request, timeout=30).read())


@pytest.mark.asyncio
async def test_calls_go_through_and_the_stats_split_the_agents_wait(sandbox) -> None:
    runner, target = await _skycap()
    manager = SandboxRelayManager(host="127.0.0.1", advertise_host="127.0.0.1", poll_timeout=2.0)
    url = await asyncio.to_thread(manager.start)
    try:
        async with aiohttp.ClientSession() as session:
            body = {"key": "sb:9876", "preview_url": sandbox, "secret": SECRET, "target": target}
            async with session.post(f"{url}/attach", json=body) as response:
                assert response.status == 200
            replies = [await asyncio.to_thread(_agent_call, sandbox, n) for n in range(3)]
            assert [r["body"]["n"] for r in replies] == [0, 1, 2]
            assert all(r["path"] == "/t/abc/v1/chat/completions" for r in replies)
            # The stub reports a call's agent-side time on the poll after it was answered.
            deadline = time.monotonic() + 10
            while True:
                async with session.get(f"{url}/stats") as response:
                    stats = await response.json()
                if stats["mean_agent_seconds"] is not None or time.monotonic() > deadline:
                    break
                await asyncio.sleep(0.2)
    finally:
        manager.stop()
        await runner.cleanup()

    assert stats["calls"] == 3 and stats["call_errors"] == 0
    assert stats["mean_forward_seconds"] >= DELAY
    assert stats["mean_agent_seconds"] >= stats["mean_forward_seconds"]
    assert 0 <= stats["mean_overhead_seconds"] < 1.0


def test_a_poll_without_the_secret_is_refused(sandbox) -> None:
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(f"{sandbox}/_relay/poll?timeout=0.1", timeout=5)
    assert refused.value.code == 403


async def _attach(session: aiohttp.ClientSession, url: str, sandbox: str, target: str) -> None:
    body = {"key": "sb:9876", "preview_url": sandbox, "secret": SECRET, "target": target}
    async with session.post(f"{url}/attach", json=body) as response:
        assert response.status == 200


@pytest.mark.asyncio
async def test_a_growing_history_crosses_as_deltas_and_arrives_whole(sandbox) -> None:
    runner, target = await _skycap()
    manager = SandboxRelayManager(host="127.0.0.1", advertise_host="127.0.0.1", poll_timeout=2.0)
    url = await asyncio.to_thread(manager.start)
    history = "turn-0 " + "x" * 20000
    try:
        async with aiohttp.ClientSession() as session:
            await _attach(session, url, sandbox, target)
            replies = []
            for n in range(4):
                history += f" turn-{n + 1} " + "y" * 3000
                replies.append(await asyncio.to_thread(_agent_call, sandbox, n, history))
            # A fresh poller (as after a re-attach) has none of the bases: it fetches the body whole.
            await _attach(session, url, sandbox, target)
            history += " turn-5"
            replies.append(await asyncio.to_thread(_agent_call, sandbox, 4, history))
            async with session.get(f"{url}/stats") as response:
                stats = await response.json()
    finally:
        manager.stop()
        await runner.cleanup()

    # skycap got every body whole, byte for byte.
    assert [r["body"]["n"] for r in replies] == [0, 1, 2, 3, 4]
    assert replies[-1]["body"]["messages"] == history
    # The first call went whole; the next three as deltas; the one after the re-attach was fetched whole.
    assert stats["delta_calls"] == 3 and stats["full_fetches"] == 1 and stats["call_errors"] == 0


def test_common_prefix() -> None:
    assert stub.common_prefix(b"abcdef", b"abcxyz") == 3
    assert stub.common_prefix(b"", b"abc") == 0
    assert stub.common_prefix(b"same", b"same") == 4
    assert stub.common_prefix(b"ab" * 5000, b"ab" * 5000 + b"c") == 10000
