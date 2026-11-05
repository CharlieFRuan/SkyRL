"""Cluster side of the sandbox relay: polls each sandbox's stub and answers its calls through skycap.

For clusters that sandboxes can't reach (no ingress, tunnels forbidden). Every connection starts here:
the manager long-polls ``stub.py`` inside each sandbox through the sandbox provider's preview URL (an
HTTPS proxy into the sandbox), forwards each call it gets to its skycap server's own harness route,
and posts the reply back. One manager runs in each skycap server actor, next to the server it feeds,
so the relay scales with ``skycap.num_servers``.

Control routes, on the cluster network only (the sandbox's environment class calls them):
    POST /attach {"key", "preview_url", "secret", "target"}   start polling one sandbox
    POST /detach {"key"}                                      stop
    GET  /stats

``/stats`` splits each call's time: ``mean_forward_seconds`` is skycap's (generation included), and
``mean_agent_seconds`` is what the agent waited, as its stub measured it; the difference,
``mean_overhead_seconds``, is the relay's (the trips through the preview proxy).

A stub sends a call that extends the previous one on its path as a delta against it (``delta``: base id, shared
prefix length, the rest); the manager keeps each sandbox's last few bodies to rebuild it, and fetches the whole body
from the stub (``/_relay/body``) when it lacks the base.
"""

import asyncio
import base64
import json
import threading
import time
from typing import Any, Dict, Optional

import aiohttp
from aiohttp import web
from loguru import logger

#: Daytona's preview proxy shows browsers a warning page unless this is set.
PREVIEW_HEADERS = {"X-Daytona-Skip-Preview-Warning": "true"}
#: Headers not carried across the relay.
SKIP_HEADERS = {"host", "content-length", "connection", "transfer-encoding", "keep-alive", "content-encoding"}


class SandboxRelayManager:
    def __init__(
        self,
        host: str,
        advertise_host: str,
        poll_timeout: float = 20.0,
        give_up_after: float = 600.0,
    ) -> None:
        """
        Args:
            host / advertise_host: where the control routes bind, and the address handed out for them.
            poll_timeout: seconds each long-poll waits in the stub for a call.
            give_up_after: seconds of failed polls after which a sandbox is dropped (it is gone).
        """
        self.host = host
        self.advertise_host = advertise_host
        self.poll_timeout = poll_timeout
        self.give_up_after = give_up_after
        self.url: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._session: Optional[aiohttp.ClientSession] = None
        self._pollers: Dict[str, asyncio.Task] = {}
        self._stats = {
            "attached": 0,
            "detached": 0,
            "calls": 0,
            "call_errors": 0,
            "poll_errors": 0,
            "dropped": 0,
            "delta_calls": 0,
            "full_fetches": 0,
        }
        self._call_seconds = 0.0
        self._forward_seconds = 0.0
        # From the stubs' own counters: calls answered and the agent-side time they took.
        self._agent_calls = 0
        self._agent_seconds = 0.0

    # --- lifecycle (called from the actor's thread) ---

    def start(self, timeout: float = 60.0) -> str:
        self._thread = threading.Thread(target=self._run, name="sandbox-relay", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("sandbox relay manager did not start")
        return self.url

    def stop(self) -> None:
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop).result(30)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())
        self._loop.run_forever()

    async def _serve(self) -> None:
        self._session = aiohttp.ClientSession(
            # Replies go out on a second connection per sandbox (its poll holds the first); keep it open across the
            # agent's tool calls, or each reply after a long command pays a new TLS handshake to the proxy.
            connector=aiohttp.TCPConnector(limit=0, ttl_dns_cache=300, keepalive_timeout=300),
            timeout=aiohttp.ClientTimeout(total=None, connect=30),
        )
        app = web.Application()
        app.router.add_post("/attach", self._attach)
        app.router.add_post("/detach", self._detach)
        app.router.add_get("/stats", self._stats_route)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, self.host, 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.url = f"http://{self.advertise_host}:{port}"
        self._ready.set()

    async def _shutdown(self) -> None:
        for task in list(self._pollers.values()):
            task.cancel()
        if self._session is not None:
            await self._session.close()

    # --- control routes ---

    async def _attach(self, request: web.Request) -> web.Response:
        body = await request.json()
        key = body["key"]
        old = self._pollers.pop(key, None)
        if old is not None:
            old.cancel()
        self._pollers[key] = asyncio.ensure_future(
            self._poll(key, body["preview_url"].rstrip("/"), body["secret"], body["target"].rstrip("/"))
        )
        self._stats["attached"] += 1
        return web.json_response({"ok": True})

    async def _detach(self, request: web.Request) -> web.Response:
        body = await request.json()
        task = self._pollers.pop(body["key"], None)
        if task is not None:
            task.cancel()
            self._stats["detached"] += 1
        return web.json_response({"ok": task is not None})

    async def _stats_route(self, request: web.Request) -> web.Response:
        calls = self._stats["calls"]
        forward = self._forward_seconds / calls if calls else None
        agent = self._agent_seconds / self._agent_calls if self._agent_calls else None
        return web.json_response(
            {
                **self._stats,
                "polling": len(self._pollers),
                "mean_call_seconds": self._call_seconds / calls if calls else None,
                "mean_forward_seconds": forward,
                "mean_agent_seconds": agent,
                "mean_overhead_seconds": agent - forward if agent is not None and forward is not None else None,
            }
        )

    # --- polling ---

    async def _poll(self, key: str, preview_url: str, secret: str, target: str) -> None:
        headers = {**PREVIEW_HEADERS, "X-Relay-Secret": secret}
        failing_since: Optional[float] = None
        backoff = 1.0
        seen = (0, 0.0)  # the stub's counters as of the last poll
        bodies: Dict[str, bytes] = {}  # call id -> body, the sandbox's last few, the bases of its deltas
        try:
            while True:
                try:
                    async with self._session.get(
                        f"{preview_url}/_relay/poll?timeout={self.poll_timeout}",
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.poll_timeout + 60),
                    ) as response:
                        if response.status != 200:
                            raise RuntimeError(f"poll HTTP {response.status}: {(await response.text())[:200]}")
                        polled = await response.json()
                    calls = polled["requests"]
                    failing_since, backoff = None, 1.0
                    if "answered" in polled:
                        answered, seconds = int(polled["answered"]), float(polled["answered_seconds"])
                        if answered >= seen[0]:  # else the stub restarted: start over from its counters
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
                    asyncio.ensure_future(self._answer(key, preview_url, headers, target, call, bodies))
        finally:
            if self._pollers.get(key) is asyncio.current_task():
                self._pollers.pop(key, None)

    async def _body(self, preview_url: str, headers: Dict[str, str], call: Dict[str, Any], bodies: Dict[str, bytes]) -> bytes:
        """The call's whole body: sent whole, rebuilt from its delta, or fetched from the stub without the base."""
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
                async with self._session.get(
                    f"{preview_url}/_relay/body?id={call['id']}", headers=headers, timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(f"body HTTP {response.status}")
                    body = base64.b64decode((await response.json())["body_b64"])
        bodies[str(call["id"])] = body
        while len(bodies) > 2:  # the stub bases a delta on the last answered call on its path
            bodies.pop(next(iter(bodies)))
        return body

    async def _answer(
        self,
        key: str,
        preview_url: str,
        headers: Dict[str, str],
        target: str,
        call: Dict[str, Any],
        bodies: Dict[str, bytes],
    ) -> None:
        """One agent call: through skycap, then the reply back to the stub."""
        started = time.monotonic()
        try:
            data = await self._body(preview_url, headers, call, bodies)
            forwarded = {k: v for k, v in (call.get("headers") or {}).items() if k.lower() not in SKIP_HEADERS}
            forward_started = time.monotonic()
            async with self._session.request(
                call["method"],
                f"{target}{call['path']}",
                headers=forwarded,
                data=data,
            ) as response:
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
                async with self._session.post(
                    f"{preview_url}/_relay/reply", json=reply, headers=headers, timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status in (200, 404):  # 404: the agent gave up on this call already
                        break
                    raise RuntimeError(f"reply HTTP {response.status}")
            except Exception as error:  # noqa: BLE001
                if attempt == 4:
                    logger.warning(f"sandbox relay {key}: could not deliver reply {call['id']}: {error}")
                    self._stats["call_errors"] += 1
                    return
                await asyncio.sleep(1 + attempt)
        self._stats["calls"] += 1
        self._call_seconds += time.monotonic() - started
