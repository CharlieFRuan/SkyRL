"""Sandbox side of the sandbox relay: a local stand-in for skycap that the cluster answers by polling.

Runs inside a remote sandbox (Daytona), standard library only, Python 3.6+. The agent is pointed at
``http://127.0.0.1:<port>/t/<trajectory>/v1``; every request it makes there is held open until the
cluster picks it up and answers it. The cluster never needs to be reachable from the sandbox: it reaches
this process through the sandbox provider's preview proxy, and every connection starts on its side.

Routes:
    anything else      the agent's calls: queued, answered when the cluster posts the reply
    GET  /_relay/poll  (cluster) wait up to ``timeout`` s for queued calls; returns {"requests": [...], "answered",
                       "answered_seconds"}: the calls answered so far and their total time as the agent saw it
    POST /_relay/reply (cluster) {"id", "status", "headers", "body_b64"}: answers one held call
    GET  /_relay/body?id=  (cluster) the whole body of a held call, for a delta whose base the cluster lacks
    GET  /_relay/health

A call whose body extends the last answered call's on the same path (an agent resending its history plus the new
turns) is sent as a delta: the length of the prefix it shares with that call, which the cluster has already
received, and the rest. Poll responses are gzipped when the cluster accepts it. Both matter: a long history crosses
the provider's proxy at a few hundred KB/s, so sending it whole on every turn costs seconds per call.

The /_relay routes need the ``X-Relay-Secret`` header given at startup, so only the cluster that started
this stub can read the agent's prompts or answer them.

    python3 stub.py --port 9876 --secret <hex>
"""

import argparse
import base64
import gzip
import hmac
import itertools
import json
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

#: Headers not carried across the relay: hop-by-hop, or recomputed by the receiving server.
SKIP_HEADERS = {"host", "content-length", "connection", "transfer-encoding", "keep-alive", "accept-encoding"}


#: Fewer shared leading bytes than this and a call is sent whole rather than as a delta.
MIN_DELTA_PREFIX = 4096


def common_prefix(a, b):
    """Length of the longest common prefix of two byte strings (bisection over C-speed slice comparisons)."""
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


class Call(object):
    def __init__(self, call_id, method, path, headers, body, base=None):
        """``base``: (id, body) of the last answered call on this path, which the cluster has."""
        self.id = call_id
        self.path = path
        self.body = body
        self.request = {"id": call_id, "method": method, "path": path, "headers": headers}
        shared = common_prefix(base[1], body) if base is not None else 0
        if shared >= MIN_DELTA_PREFIX:
            self.request["delta"] = {
                "base": base[0],
                "prefix": shared,
                "suffix_b64": base64.b64encode(body[shared:]).decode("ascii"),
            }
        else:
            self.request["body_b64"] = base64.b64encode(body).decode("ascii")
        self.done = threading.Event()
        self.reply = None
        self.arrived = time.time()


class Relay(object):
    def __init__(self):
        self.lock = threading.Condition()
        self.queue = []  # calls not yet handed to a poll
        self.held = {}  # id -> Call, handed out and not yet answered
        self.ids = itertools.count(1)
        self.last_poll = time.time()
        # Calls answered, and their total time from the agent's request to the reply: what the agent waited.
        self.answered = 0
        self.answered_seconds = 0.0
        self.last_answered = {}  # path -> (id, body) of the last call answered there: the cluster has its body

    def submit(self, method, path, headers, body):
        with self.lock:
            call = Call(str(next(self.ids)), method, path, headers, body, self.last_answered.get(path))
            self.queue.append(call)
            self.held[call.id] = call
            self.lock.notify_all()
        return call

    def poll(self, timeout):
        deadline = time.time() + timeout
        with self.lock:
            self.last_poll = time.time()
            while not self.queue:
                left = deadline - time.time()
                if left <= 0:
                    return []
                self.lock.wait(left)
            taken, self.queue = self.queue, []
            return taken

    def requeue(self, calls):
        """Calls whose poll response never went out (the cluster's connection broke): hand them out again."""
        with self.lock:
            self.queue[:0] = [call for call in calls if call.id in self.held]
            self.lock.notify_all()

    def body(self, call_id):
        with self.lock:
            call = self.held.get(call_id)
        return None if call is None else call.body

    def counters(self):
        with self.lock:
            return {"answered": self.answered, "answered_seconds": self.answered_seconds}

    def answer(self, call_id, reply):
        with self.lock:
            call = self.held.pop(call_id, None)
            if call is not None:
                self.answered += 1
                self.answered_seconds += time.time() - call.arrived
                self.last_answered[call.path] = (call.id, call.body)
        if call is None:
            return False
        call.reply = reply
        call.done.set()
        return True


def make_handler(relay, secret, call_timeout):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # quiet: the agent's own logs are what matter
            pass

        def _body(self):
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length else b""

        def _send(self, status, body, content_type="application/json", headers=None):
            self.send_response(status)
            sent = set()
            for name, value in (headers or {}).items():
                if name.lower() in SKIP_HEADERS or name.lower() == "content-type":
                    continue
                self.send_header(name, value)
                sent.add(name.lower())
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, obj):
            self._send(status, json.dumps(obj).encode("utf-8"))

        def _authorized(self):
            given = (self.headers.get("X-Relay-Secret") or "").encode("utf-8")
            return hmac.compare_digest(given, secret)

        def _relay_route(self, method):
            if not self._authorized():
                return self._json(403, {"error": "bad relay secret"})
            path = self.path.split("?", 1)[0]
            if method == "GET" and path == "/_relay/health":
                with relay.lock:
                    return self._json(200, {"ok": True, "queued": len(relay.queue), "held": len(relay.held)})
            if method == "GET" and path == "/_relay/poll":
                timeout = 20.0
                if "timeout=" in self.path:
                    try:
                        timeout = min(60.0, float(self.path.split("timeout=", 1)[1].split("&", 1)[0]))
                    except ValueError:
                        pass
                calls = relay.poll(timeout)
                data = json.dumps(dict(relay.counters(), requests=[call.request for call in calls])).encode("utf-8")
                try:
                    if "gzip" in (self.headers.get("Accept-Encoding") or ""):
                        return self._send(200, gzip.compress(data, 6), headers={"Content-Encoding": "gzip"})
                    return self._send(200, data)
                except (OSError, ValueError):
                    relay.requeue(calls)
                    raise
            if method == "GET" and path == "/_relay/body":
                body = relay.body(self.path.split("id=", 1)[-1].split("&", 1)[0])
                if body is None:
                    return self._json(404, {"error": "no such held call"})
                return self._json(200, {"body_b64": base64.b64encode(body).decode("ascii")})
            if method == "POST" and path == "/_relay/reply":
                reply = json.loads(self._body().decode("utf-8"))
                ok = relay.answer(str(reply.get("id")), reply)
                return self._json(200 if ok else 404, {"ok": ok})
            return self._json(404, {"error": "no such relay route"})

        def _agent_call(self, method):
            headers = {k: v for k, v in self.headers.items() if k.lower() not in SKIP_HEADERS}
            call = relay.submit(method, self.path, headers, self._body() if method == "POST" else b"")
            if not call.done.wait(call_timeout):
                with relay.lock:
                    relay.held.pop(call.id, None)
                return self._json(504, {"error": {"message": "sandbox relay: no reply from the cluster in time"}})
            reply = call.reply
            body = base64.b64decode(reply.get("body_b64") or "")
            headers = reply.get("headers") or {}
            content_type = headers.get("Content-Type") or headers.get("content-type") or "application/json"
            self._send(int(reply.get("status") or 502), body, content_type, headers)

        def do_GET(self):
            if self.path.startswith("/_relay/"):
                return self._relay_route("GET")
            return self._agent_call("GET")

        def do_POST(self):
            if self.path.startswith("/_relay/"):
                return self._relay_route("POST")
            return self._agent_call("POST")

    return Handler


class ThreadingServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--secret", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--call-timeout", type=float, default=7200.0, help="seconds an agent call may wait")
    args = parser.parse_args(argv)
    relay = Relay()
    server = ThreadingServer((args.host, args.port), make_handler(relay, args.secret.encode("utf-8"), args.call_timeout))
    sys.stdout.write("sandbox relay stub listening on %s:%d\n" % (args.host, args.port))
    sys.stdout.flush()
    server.serve_forever()


if __name__ == "__main__":
    main()
