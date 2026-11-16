"""Preview relay: a call survives a lost poll response, a lost acknowledgement and an undeliverable reply.

A real stub.py process, a real SandboxRelayManager, a fake model endpoint, and between manager and stub a proxy
that can be told to lose things.   Run from a checkout root:  python test_relay_redelivery.py"""
import asyncio, json, os, socket, subprocess, sys, time
import aiohttp
from aiohttp import web

sys.path.insert(0, ".")
from examples.train_integrations.harbor_skycap.sandbox_relay.manager import SandboxRelayManager

STUB = "examples/train_integrations/harbor_skycap/sandbox_relay/stub.py"


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


async def main():
    stub_port, secret = free_port(), "s3cret"
    stub = subprocess.Popen([sys.executable, STUB, "--port", str(stub_port), "--secret", secret], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    forwarded = []      # every request that reached the fake model endpoint
    plan = {"drop_poll_responses": 0, "drop_poll_requests": 0, "fail_replies": 0}
    seen = {"polls": 0, "poll_urls": []}

    async def model(request):
        body = await request.json()
        forwarded.append(body["n"])
        await asyncio.sleep(0.2)
        return web.json_response({"choices": [{"message": {"role": "assistant", "content": "answer %s" % body["n"]}}]})

    async def proxy(request):
        path = request.rel_url.path
        url = "http://127.0.0.1:%d%s" % (stub_port, request.rel_url)
        if path == "/_relay/poll":
            seen["polls"] += 1; seen["poll_urls"].append(str(request.rel_url))
            if plan["drop_poll_requests"] > 0 and "ack=" in str(request.rel_url):
                plan["drop_poll_requests"] -= 1      # the stub never sees this poll (so never the ack in it)
                return web.Response(status=502, text="lost request")
        if path == "/_relay/reply" and plan["fail_replies"] > 0:
            plan["fail_replies"] -= 1
            await request.read()
            return web.Response(status=500, text="reply lost")
        async with aiohttp.ClientSession() as session:
            async with session.request(request.method, url, data=await request.read(), headers={k: v for k, v in request.headers.items() if k.lower() in ("x-relay-secret", "content-type")}) as r:
                data = await r.read()   # plain: the proxy asks the stub for no compression
                if path == "/_relay/poll" and plan["drop_poll_responses"] > 0 and b'"requests": [{' in data:
                    plan["drop_poll_responses"] -= 1  # the stub handed the call out; the cluster never gets it
                    return web.Response(status=502, text="lost response")
                return web.Response(status=r.status, body=data, headers={k: v for k, v in r.headers.items() if k.lower() == "content-type"})

    async def serve(handler):
        app = web.Application(client_max_size=2**26); app.router.add_route("*", "/{tail:.*}", handler)
        runner = web.AppRunner(app, access_log=None); await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0); await site.start()
        return runner, "http://127.0.0.1:%d" % site._server.sockets[0].getsockname()[1]

    model_runner, model_url = await serve(model)
    proxy_runner, proxy_url = await serve(proxy)
    manager = SandboxRelayManager("127.0.0.1", "127.0.0.1", poll_timeout=2.0, redeliver_after=2.0)
    murl = await asyncio.to_thread(manager.start)
    await asyncio.sleep(0.5)

    async def agent_call(n, timeout=60):
        t0 = time.time()
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.post("http://127.0.0.1:%d/t/x/v1/chat/completions" % stub_port, json={"n": n, "messages": ["m"] * (n + 1)}) as r:
                return (await r.json())["choices"][0]["message"]["content"], time.time() - t0

    async def stats():
        async with aiohttp.ClientSession() as session:
            async with session.get(murl + "/stats") as r:
                return await r.json()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(murl + "/attach", json={"key": "sbx:1", "preview_url": proxy_url, "secret": secret, "target": model_url}) as r:
                assert r.status == 200, await r.text()
        # 1. Normal call.
        text, dt = await agent_call(1); assert text == "answer 1" and dt < 5, (text, dt)
        # 2. The poll response carrying the call is lost: before the fix the agent waited out its own timeout.
        plan["drop_poll_responses"] = 1
        text, dt = await agent_call(2); s = await stats()
        assert text == "answer 2" and dt < 15, (text, dt); assert s["recovered"] == 1, s
        print("lost poll response: answered in %.1f s, recovered=%d" % (dt, s["recovered"]))
        # 3. The acknowledgement is lost (the next poll request never reaches the stub): the stub hands the call out
        #    again while it is being answered; the manager must not forward it twice.
        async def slow():
            return await agent_call(3)
        plan["drop_poll_requests"] = 3
        text, dt = await slow(); await asyncio.sleep(3)
        s = await stats()
        assert text == "answer 3" and forwarded.count(3) == 1, (text, forwarded)
        print("lost acknowledgement: forwarded once, duplicates ignored=%d" % s["duplicates"])
        # 4. The reply cannot be delivered (5 tries): the call comes back and is answered on a second pass.
        plan["fail_replies"] = 5
        text, dt = await agent_call(4, timeout=90); s = await stats()
        assert text == "answer 4" and forwarded.count(4) == 2 and s["call_errors"] == 1, (text, forwarded, s)
        print("undeliverable reply: answered after %.1f s on the second pass, call_errors=%d" % (dt, s["call_errors"]))
        # 5. Sequence still intact, counters sane.
        text, dt = await agent_call(5); s = await stats()
        assert text == "answer 5" and dt < 5 and s["calls"] == 5 and s["dropped"] == 0, (text, dt, s)
        # 6. A poller that does not ask for redelivery (the exec transport's client) is never sent a call twice.
        async with aiohttp.ClientSession() as session:
            t = asyncio.ensure_future(agent_call(6, timeout=8))
            await asyncio.sleep(0.3)
        print("stats:", {k: s[k] for k in ("calls", "call_errors", "poll_errors", "recovered", "duplicates", "delta_calls")})
        try:
            await t
        except Exception:
            pass
        print("OK")
    finally:
        stub.terminate()
        await proxy_runner.cleanup(); await model_runner.cleanup()


asyncio.run(main())
