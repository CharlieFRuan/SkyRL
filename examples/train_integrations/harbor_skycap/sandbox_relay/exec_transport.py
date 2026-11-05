"""The sandbox relay over the sandbox provider's exec API, for providers without preview URLs (Dune Cloud).

``manager.py`` reaches each sandbox's ``stub.py`` over HTTP through a signed preview URL. Some Daytona-compatible
providers have none (Dune: 422 "Preview URLs are not supported"), but every one of them can run a command in the
sandbox. Here the manager reaches the same stub, with the same protocol, by running a tiny client inside the sandbox:

    poll   sandbox.process.exec("<py> rx.py poll <timeout> <ack>")   held in the sandbox up to <timeout> s
    reply  sandbox.process.exec("<py> rx.py reply <<EOF\\n<base64 gzip json>\\nEOF")  (or upload the file, then exec)
    body   sandbox.process.exec("<py> rx.py body <id>")

``rx.py`` (``HELPER`` below, Python 3.6+ stdlib, the same interpreter requirement as the stub) talks to the stub on
127.0.0.1 and prints each answer as one ASCII frame, ``@@RX <tag> <status> <base64 of gzip of the body>@@``: exec
returns stdout and stderr merged, as text, so the frame is what makes the output parseable.

A poll that the sandbox answered but whose exec response never reached the manager (an API error, a timeout) would
lose those calls: the stub hands each call out once and the agent would wait for it until its call timeout. So
``rx.py`` spools every non-empty poll answer under a sequence number; the manager acknowledges the last one it
received on its next poll, and any spooled answer past that (lost on the way back, or taken by a poll orphaned by a
detach / re-attach) is replayed before the next long-poll. A replayed call that was answered after all is answered
again harmlessly (the stub returns 404 for it), at the cost of one duplicate forward.

``ExecRelayManager`` is ``SandboxRelayManager`` with this transport: ``/attach`` takes ``{"key", "sandbox_id",
"port", "secret", "target"}`` instead of a ``preview_url`` (with a ``preview_url`` it still polls over HTTP), and it
looks the sandbox up with its own ``AsyncDaytona`` client (``DAYTONA_API_URL`` / ``DAYTONA_API_KEY`` in its
environment). One poll exec per sandbox is always in flight, so the client's connection pool is unbounded.
"""

import asyncio
import base64
import gzip
import json
import re
import shlex
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

from aiohttp import web
from loguru import logger

from .manager import SKIP_HEADERS, SandboxRelayManager

#: Where the exec client lives in the sandbox (per stub port): writable by any user, unlike the stub's root-owned dir.
CLIENT_DIR = "/tmp/.skycap_relay_rx_{port}"
#: The stub's directory, for the uv-managed Python that ``stub_start_command`` installs on images without python3.
RELAY_DIR = "/var/lib/.skycap_relay"
#: Replies whose base64 frame is at most this long go inline in the exec command; larger ones are uploaded first.
#: (Dune took a 1 MB command without complaint; keep a margin for providers that pass it through argv.)
MAX_INLINE_REPLY = 512 * 1024

FRAME = re.compile(r"@@RX (\S+) (\d+) ([A-Za-z0-9+/=]*)@@")

#: The sandbox-side client. ``conf.json`` next to it holds the stub's port and secret.
HELPER = r'''
import base64, gzip, json, os, sys, urllib.error, urllib.request
D = os.path.dirname(os.path.abspath(__file__))
SPOOL = os.path.join(D, "spool")
with open(os.path.join(D, "conf.json")) as f:
    CONF = json.load(f)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never through the sandbox's http_proxy


def call(path, data=None, timeout=60.0):
    headers = {"X-Relay-Secret": CONF["secret"], "Accept-Encoding": "gzip"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request("http://127.0.0.1:%d%s" % (CONF["port"], path), data=data, headers=headers)
    try:
        response = OPENER.open(request, timeout=timeout)
        status, body, encoding = response.getcode(), response.read(), response.headers.get("Content-Encoding")
    except urllib.error.HTTPError as e:
        status, body, encoding = e.code, e.read(), e.headers.get("Content-Encoding")
    return status, (body if encoding == "gzip" else gzip.compress(body, 6))


def emit(tag, status, gz):
    sys.stdout.write("@@RX %s %d %s@@\n" % (tag, status, base64.b64encode(gz).decode("ascii")))


def spooled():
    return sorted(int(name) for name in os.listdir(SPOOL) if name.isdigit())


def poll(timeout, ack):
    for seq in spooled():
        if seq <= ack:
            os.remove(os.path.join(SPOOL, str(seq)))
    left = spooled()  # answered here but never acknowledged: lost on the way back, or taken by an orphaned poll
    for seq in left:
        with open(os.path.join(SPOOL, str(seq)), "rb") as f:
            emit("R%d" % seq, 200, f.read())
    if left:
        return
    status, gz = call("/_relay/poll?timeout=%s" % timeout, timeout=float(timeout) + 30)
    seq = 0
    if status == 200 and json.loads(gzip.decompress(gz).decode("utf-8")).get("requests"):
        seq = max([ack] + spooled()) + 1
        with open(os.path.join(SPOOL, str(seq)), "wb") as f:
            f.write(gz)
    emit(str(seq), status, gz)


def main():
    op = sys.argv[1]
    if op == "poll":
        poll(sys.argv[2], int(sys.argv[3]))
    elif op == "body":
        status, gz = call("/_relay/body?id=" + sys.argv[2], timeout=120)
        emit("body", status, gz)
    elif op == "reply":
        if len(sys.argv) > 2:
            with open(sys.argv[2], "rb") as f:
                frame = f.read()
            os.remove(sys.argv[2])
        else:
            frame = sys.stdin.read().encode("ascii")
        status, gz = call("/_relay/reply", data=gzip.decompress(base64.b64decode(frame)), timeout=60)
        emit("reply", status, gz)
    elif op == "health":
        status, gz = call("/_relay/health", timeout=10)
        emit("health", status, gz)


main()
'''


def setup_command(port: int, secret: str, client_dir: Optional[str] = None) -> str:
    """Shell that writes the client and its config, and prints the Python that will run it (``@@PY <path>``):
    the image's python3 if it is 3.6+, else the one ``stub_start_command`` installed for the stub."""
    d = client_dir or CLIENT_DIR.format(port=port)
    conf = json.dumps({"port": port, "secret": secret})
    return (
        f"set -e; umask 077; mkdir -p {d}/spool; cat > {d}/rx.py <<'SKYCAP_RX_EOF'\n{HELPER}\nSKYCAP_RX_EOF\n"
        f"printf '%s' {shlex.quote(conf)} > {d}/conf.json\n"
        "PY=''; if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; sys.exit(sys.version_info < (3, 6))'"
        " 2>/dev/null; then PY=$(command -v python3); fi\n"
        f'if [ -z "$PY" ] && [ -x {RELAY_DIR}/bin/uv ]; then PY=$({RELAY_DIR}/bin/uv python find 3.12 2>/dev/null || true); fi\n'
        'if [ -z "$PY" ]; then echo "@@NOPY@@"; exit 1; fi\n'
        f'"$PY" -S -E {d}/rx.py health\n'
        'echo "@@PY $PY@@"'
    )


def parse_frames(output: str) -> List[tuple]:
    """``[(tag, status, body bytes)]`` from a client run's output (stdout and stderr merged)."""
    return [(tag, int(status), gzip.decompress(base64.b64decode(b64))) for tag, status, b64 in FRAME.findall(output)]


class ExecError(RuntimeError):
    pass


class ExecStubTransport:
    """One sandbox's stub, reached by exec. ``sandbox`` is a Daytona ``AsyncSandbox`` (or anything with
    ``process.exec(command, timeout=)`` returning ``.result``/``.exit_code`` and ``fs.upload_file(bytes, path)``)."""

    def __init__(
        self,
        sandbox: Any,
        port: int,
        secret: str,
        max_inline_reply: int = MAX_INLINE_REPLY,
        client_dir: Optional[str] = None,
    ) -> None:
        self.sandbox = sandbox
        self.port = int(port)
        self.secret = secret
        self.dir = client_dir or CLIENT_DIR.format(port=self.port)
        self.max_inline_reply = max_inline_reply
        self.python: Optional[str] = None
        self._acked = 0  # the last spooled poll answer this side has received
        self.stats = {"execs": 0, "uploads": 0, "replays": 0}

    async def _exec(self, command: str, timeout: float) -> str:
        self.stats["execs"] += 1
        response = await asyncio.wait_for(
            self.sandbox.process.exec(command, timeout=int(timeout)), timeout=timeout + 30
        )
        return response.result or ""

    def _client(self, *args: str) -> str:
        return f"{shlex.quote(self.python)} -S -E {self.dir}/rx.py " + " ".join(shlex.quote(a) for a in args)

    async def setup(self) -> None:
        output = await self._exec(setup_command(self.port, self.secret, self.dir), timeout=60)
        found = re.search(r"@@PY (.+?)@@", output)
        frames = parse_frames(output)
        if found is None or not frames or frames[0][1] != 200:
            raise ExecError(f"sandbox relay exec client did not start: {output[-500:]}")
        self.python = found.group(1)

    async def poll(self, timeout: float) -> Dict[str, Any]:
        """One long-poll: the stub's poll answer (``requests``, ``answered``, ``answered_seconds``), with any
        replayed answers' requests merged in."""
        output = await self._exec(self._client("poll", str(timeout), str(self._acked)), timeout=timeout + 60)
        frames = parse_frames(output)
        if not frames:
            raise ExecError(f"poll: no answer from the exec client: {output[-500:]}")
        merged: Dict[str, Any] = {"requests": []}
        for tag, status, body in frames:
            if status != 200:
                raise ExecError(f"poll HTTP {status}: {body[:200]!r}")
            polled = json.loads(body.decode("utf-8"))
            merged["requests"].extend(polled.get("requests") or [])
            if not tag.startswith("R"):  # a replayed answer's counters are stale
                for counter in ("answered", "answered_seconds"):
                    if counter in polled:
                        merged[counter] = polled[counter]
            self._acked = max(self._acked, int(tag.lstrip("R")))
        if any(tag.startswith("R") for tag, _, _ in frames):
            self.stats["replays"] += 1
        return merged

    async def body(self, call_id: str) -> bytes:
        frames = parse_frames(await self._exec(self._client("body", str(call_id)), timeout=150))
        if not frames or frames[0][1] != 200:
            raise ExecError(f"body: {frames[0][1] if frames else 'no answer'}")
        return base64.b64decode(json.loads(frames[0][2].decode("utf-8"))["body_b64"])

    async def reply(self, reply: Dict[str, Any]) -> int:
        """Deliver one reply; the stub's status (200, or 404 when the agent already gave up on the call)."""
        frame = base64.b64encode(gzip.compress(json.dumps(reply).encode("utf-8"), 6)).decode("ascii")
        if len(frame) <= self.max_inline_reply:
            command = self._client("reply") + f" <<'SKYCAP_RX_EOF'\n{frame}\nSKYCAP_RX_EOF"
        else:
            self.stats["uploads"] += 1
            path = f"{self.dir}/reply.{reply.get('id')}.{self.stats['uploads']}"
            await self.sandbox.fs.upload_file(frame.encode("ascii"), path)
            command = self._client("reply", path)
        frames = parse_frames(await self._exec(command, timeout=90))
        if not frames:
            raise ExecError("reply: no answer from the exec client")
        return frames[0][1]


SandboxLookup = Callable[[str], Awaitable[Any]]


class ExecRelayManager(SandboxRelayManager):
    """``SandboxRelayManager`` that reaches stubs through the provider's exec API.

    ``/attach`` with ``{"key", "sandbox_id", "port", "secret", "target"}``; with a ``preview_url`` instead it is the
    HTTP manager. ``sandbox_lookup`` (async, sandbox id -> sandbox) defaults to an ``AsyncDaytona`` client made on
    the manager's loop."""

    def __init__(
        self,
        *args: Any,
        sandbox_lookup: Optional[SandboxLookup] = None,
        transport_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._transport_kwargs = transport_kwargs or {}
        self._lookup = sandbox_lookup
        self._daytona: Any = None
        self._transports: Dict[str, ExecStubTransport] = {}
        self._stats.update({"exec_attached": 0, "setup_errors": 0, "replays": 0, "uploads": 0})

    async def _sandbox(self, sandbox_id: str) -> Any:
        if self._lookup is not None:
            return await self._lookup(sandbox_id)
        if self._daytona is None:
            from daytona import AsyncDaytona, DaytonaConfig

            # One poll per sandbox is always held open: an unbounded pool, or polls queue behind each other.
            self._daytona = AsyncDaytona(DaytonaConfig(connection_pool_maxsize=None))
        return await self._daytona.get(sandbox_id)

    async def _shutdown(self) -> None:
        await super()._shutdown()
        if self._daytona is not None:
            await self._daytona.close()

    async def _attach(self, request: web.Request) -> web.Response:
        body = await request.json()
        if body.get("preview_url"):
            return await self._attach_body_http(body)
        key = body["key"]
        old = self._pollers.pop(key, None)
        if old is not None:
            old.cancel()
        try:
            transport = ExecStubTransport(
                await self._sandbox(body["sandbox_id"]), int(body["port"]), body["secret"], **self._transport_kwargs
            )
            await transport.setup()
        except Exception as error:  # noqa: BLE001 - the environment fails the trial on a non-200
            self._stats["setup_errors"] += 1
            return web.json_response({"ok": False, "error": f"{type(error).__name__}: {error}"}, status=502)
        self._transports[key] = transport
        self._pollers[key] = asyncio.ensure_future(self._exec_poll(key, transport, body["target"].rstrip("/")))
        self._stats["attached"] += 1
        self._stats["exec_attached"] += 1
        return web.json_response({"ok": True})

    async def _attach_body_http(self, body: Dict[str, Any]) -> web.Response:
        old = self._pollers.pop(body["key"], None)
        if old is not None:
            old.cancel()
        self._pollers[body["key"]] = asyncio.ensure_future(
            self._poll(body["key"], body["preview_url"].rstrip("/"), body["secret"], body["target"].rstrip("/"))
        )
        self._stats["attached"] += 1
        return web.json_response({"ok": True})

    async def _stats_route(self, request: web.Request) -> web.Response:
        self._stats["replays"] = sum(t.stats["replays"] for t in self._transports.values())
        self._stats["uploads"] = sum(t.stats["uploads"] for t in self._transports.values())
        return await super()._stats_route(request)

    # --- polling, as SandboxRelayManager._poll/_body/_answer, through the transport ---

    async def _exec_poll(self, key: str, transport: ExecStubTransport, target: str) -> None:
        failing_since: Optional[float] = None
        backoff = 1.0
        seen = (0, 0.0)
        bodies: Dict[str, bytes] = {}
        try:
            while True:
                try:
                    polled = await transport.poll(self.poll_timeout)
                    calls = polled["requests"]
                    failing_since, backoff = None, 1.0
                    if "answered" in polled:
                        answered, seconds = int(polled["answered"]), float(polled["answered_seconds"])
                        if answered >= seen[0]:
                            self._agent_calls += answered - seen[0]
                            self._agent_seconds += seconds - seen[1]
                        seen = (answered, seconds)
                except asyncio.CancelledError:
                    raise
                except Exception as error:  # noqa: BLE001 - the sandbox may be starting, busy or gone
                    self._stats["poll_errors"] += 1
                    now = time.monotonic()
                    failing_since = failing_since or now
                    if now - failing_since > self.give_up_after:
                        logger.warning(f"sandbox relay {key}: dropping after {self.give_up_after:.0f}s of failed polls: {error}")
                        self._stats["dropped"] += 1
                        return
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 15.0)
                    continue
                for call in calls:
                    asyncio.ensure_future(self._exec_answer(key, transport, target, call, bodies))
        finally:
            if self._pollers.get(key) is asyncio.current_task():
                self._pollers.pop(key, None)
                self._transports.pop(key, None)

    async def _exec_body(self, transport: ExecStubTransport, call: Dict[str, Any], bodies: Dict[str, bytes]) -> bytes:
        delta = call.get("delta")
        if delta is None:
            body = base64.b64decode(call.get("body_b64") or "")
        else:
            base = bodies.get(str(delta["base"]))
            if base is not None and len(base) >= int(delta["prefix"]):
                self._stats["delta_calls"] += 1
                body = base[: int(delta["prefix"])] + base64.b64decode(delta["suffix_b64"])
            else:
                self._stats["full_fetches"] += 1
                body = await transport.body(str(call["id"]))
        bodies[str(call["id"])] = body
        while len(bodies) > 2:
            bodies.pop(next(iter(bodies)))
        return body

    async def _exec_answer(
        self, key: str, transport: ExecStubTransport, target: str, call: Dict[str, Any], bodies: Dict[str, bytes]
    ) -> None:
        started = time.monotonic()
        try:
            data = await self._exec_body(transport, call, bodies)
            forwarded = {k: v for k, v in (call.get("headers") or {}).items() if k.lower() not in SKIP_HEADERS}
            forward_started = time.monotonic()
            async with self._session.request(call["method"], f"{target}{call['path']}", headers=forwarded, data=data) as response:
                body = await response.read()
                self._forward_seconds += time.monotonic() - forward_started
                reply = {
                    "id": call["id"],
                    "status": response.status,
                    "headers": {k: v for k, v in response.headers.items() if k.lower() not in SKIP_HEADERS},
                    "body_b64": base64.b64encode(body).decode("ascii"),
                }
        except Exception as error:  # noqa: BLE001 - answer the agent with an error rather than leave it hanging
            self._stats["call_errors"] += 1
            message = f"sandbox relay: skycap call failed: {type(error).__name__}: {error}"
            reply = {
                "id": call["id"],
                "status": 502,
                "headers": {"Content-Type": "application/json"},
                "body_b64": base64.b64encode(json.dumps({"error": {"message": message}}).encode()).decode("ascii"),
            }
        for attempt in range(5):
            try:
                status = await transport.reply(reply)
                if status in (200, 404):
                    break
                raise ExecError(f"reply HTTP {status}")
            except Exception as error:  # noqa: BLE001
                if attempt == 4:
                    logger.warning(f"sandbox relay {key}: could not deliver reply {call['id']}: {error}")
                    self._stats["call_errors"] += 1
                    return
                await asyncio.sleep(1 + attempt)
        self._stats["calls"] += 1
        self._call_seconds += time.monotonic() - started
