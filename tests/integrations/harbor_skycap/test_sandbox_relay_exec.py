"""The sandbox relay over exec (``exec_transport``) on one machine: a fake sandbox runs each exec as a local shell
command, as a provider without preview URLs would run it in the sandbox."""

import asyncio
import hashlib
import json
import os
import threading
import time
import urllib.request

import aiohttp
import pytest
from aiohttp import web

from examples.train_integrations.harbor_skycap.sandbox_relay import stub
from examples.train_integrations.harbor_skycap.sandbox_relay.exec_transport import (
    ExecRelayManager,
    ExecStubTransport,
    parse_frames,
)

SECRET = "s3cret"


class _Result:
    def __init__(self, result: str, exit_code: int) -> None:
        self.result, self.exit_code = result, exit_code


class FakeSandbox:
    """``process.exec`` runs ``sh -c`` locally with stderr merged into the result, like Daytona's; ``fs.upload_file``
    writes the file. ``lose_poll_outputs``: that many non-empty poll answers run in the sandbox but never come back."""

    def __init__(self, lose_poll_outputs: int = 0) -> None:
        self.process, self.fs = self, self
        self.commands = []
        self.lose = lose_poll_outputs

    async def exec(self, command: str, timeout: int = None) -> _Result:
        self.commands.append(command)
        proc = await asyncio.create_subprocess_exec(
            "sh", "-c", command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        output = out.decode()
        if self.lose and " poll " in command and any(json.loads(b)["requests"] for _, _, b in parse_frames(output)):
            self.lose -= 1
            raise ConnectionError("exec response lost")
        return _Result(output, proc.returncode)

    async def upload_file(self, data: bytes, path: str) -> None:
        with open(path, "wb") as f:
            f.write(data)


@pytest.fixture
def sandbox_stub():
    relay = stub.Relay()
    server = stub.ThreadingServer(("127.0.0.1", 0), stub.make_handler(relay, SECRET.encode(), 30.0))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()


async def _skycap() -> tuple:
    async def answer(request: web.Request) -> web.Response:
        raw = await request.read()
        body = json.loads(raw)
        return web.json_response(
            {"n": body["n"], "sha": hashlib.sha256(raw).hexdigest(), "pad": os.urandom(body.get("pad", 0) // 2).hex()}
        )

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", answer)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


def _agent_call(port: int, payload: dict) -> tuple:
    raw = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/t/abc/v1/chat/completions", data=raw, headers={"Content-Type": "application/json"}
    )
    return hashlib.sha256(raw).hexdigest(), json.loads(urllib.request.urlopen(request, timeout=30).read())


async def _run(port: int, sandbox: FakeSandbox, tmp_path, payloads, **transport_kwargs) -> tuple:
    runner, target = await _skycap()

    async def lookup(sandbox_id: str) -> FakeSandbox:
        assert sandbox_id == "sb-1"
        return sandbox

    manager = ExecRelayManager(
        host="127.0.0.1",
        advertise_host="127.0.0.1",
        poll_timeout=2.0,
        sandbox_lookup=lookup,
        transport_kwargs={"client_dir": str(tmp_path / "rx"), **transport_kwargs},
    )
    url = await asyncio.to_thread(manager.start)
    try:
        async with aiohttp.ClientSession() as session:
            attach = {"key": "sb-1:9876", "sandbox_id": "sb-1", "port": port, "secret": SECRET, "target": target}
            async with session.post(f"{url}/attach", json=attach) as response:
                assert response.status == 200, await response.text()
            replies = [await asyncio.to_thread(_agent_call, port, p) for p in payloads]
            deadline = time.monotonic() + 10
            while True:
                async with session.get(f"{url}/stats") as response:
                    stats = await response.json()
                done = stats["calls"] + stats["call_errors"] >= len(payloads) and stats["mean_agent_seconds"] is not None
                if done or time.monotonic() > deadline:
                    break
                await asyncio.sleep(0.2)
    finally:
        manager.stop()
        await runner.cleanup()
    return replies, stats


@pytest.mark.asyncio
async def test_calls_deltas_and_large_replies_cross_by_exec(sandbox_stub, tmp_path) -> None:
    history = "h" * 20000
    payloads = []
    for n in range(4):
        history += "y" * 3000
        payloads.append({"messages": history, "n": n, "pad": 300_000 if n == 2 else 10})
    # A small inline limit sends the big (incompressible) reply through upload_file.
    sandbox = FakeSandbox()
    replies, stats = await _run(sandbox_stub, sandbox, tmp_path, payloads, max_inline_reply=100_000)
    for (sha, reply), payload in zip(replies, payloads):
        assert reply["sha"] == sha and reply["n"] == payload["n"] and len(reply["pad"]) == payload["pad"]
    assert stats["calls"] == 4 and stats["call_errors"] == 0
    assert stats["delta_calls"] == 3 and stats["uploads"] == 1
    assert stats["mean_agent_seconds"] is not None
    assert not [f for f in os.listdir(tmp_path / "rx") if f.startswith("reply.")]  # the uploaded reply is removed


@pytest.mark.asyncio
async def test_a_poll_answer_lost_on_the_way_back_is_replayed(sandbox_stub, tmp_path) -> None:
    sandbox = FakeSandbox(lose_poll_outputs=1)
    replies, stats = await _run(sandbox_stub, sandbox, tmp_path, [{"n": 0}, {"n": 1}])
    assert [r["n"] for _, r in replies] == [0, 1]
    assert stats["poll_errors"] == 1 and stats["replays"] == 1 and stats["call_errors"] == 0
    # Acknowledged answers leave the spool.
    assert len(os.listdir(tmp_path / "rx" / "spool")) <= 1


@pytest.mark.asyncio
async def test_setup_fails_cleanly_without_a_stub(tmp_path) -> None:
    transport = ExecStubTransport(FakeSandbox(), 1, SECRET, client_dir=str(tmp_path / "rx"))
    with pytest.raises(RuntimeError):
        await transport.setup()


def test_parse_frames_ignores_noise() -> None:
    import base64
    import gzip

    frame = base64.b64encode(gzip.compress(b'{"a": 1}')).decode()
    assert parse_frames(f"warning: x\n@@RX 3 200 {frame}@@\nmore") == [("3", 200, b'{"a": 1}')]
