"""A labelled Daytona environment that starts the sandbox relay's stub (``skycap.sandbox_relay``).

With the environment kwarg ``skycap_relay`` (``{"manager", "target", "port"}``; the generator sets it per trial),
``start`` launches ``stub.py`` in the sandbox, makes a signed Daytona preview URL for its port, and asks the relay
manager to poll it; ``stop`` detaches. The agent calls the stub on ``127.0.0.1`` and never needs to reach the
cluster. Without ``skycap_relay`` it is ``LabelledDaytonaEnvironment`` (as a stock ``harbor run`` would use it,
with ``skycap_relay`` passed through ``--ek`` instead).

    harbor_trial_config.environment.import_path=examples.train_integrations.harbor_skycap.sandbox_relay.environment:RelayDaytonaEnvironment
    harbor_trial_config.environment.kwargs.labels.owner=<you> harbor_trial_config.environment.kwargs.labels.run=<run>
"""

import os
import secrets
import shlex
from typing import Any, Dict, Optional

from ..daytona import LabelledDaytonaEnvironment

#: Where the relay stub lives in the sandbox.
RELAY_DIR = "/var/lib/.skycap_relay"
#: The signed preview URL's lifetime when no ``ttl_minutes`` bounds the sandbox's.
DEFAULT_PREVIEW_TTL_MINUTES = 360


def _stub_source() -> str:
    path = os.environ.get("SKYCAP_RELAY_STUB") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "stub.py")
    with open(path) as f:
        return f.read()


def stub_start_command(port: int, secret: str, stub: str) -> str:
    """Shell that writes the stub, starts it on ``port`` and waits for its health route; prints RELAY_STUB_UP.

    It runs the image's python3 if it has one (the stub is 3.6+ stdlib), else a uv-managed Python, installing curl
    first on images that have neither curl nor wget (some Terminal-Bench tasks)."""
    return (
        f"set -e; mkdir -p {RELAY_DIR}; cat > {RELAY_DIR}/stub.py <<'SKYCAP_RELAY_EOF'\n{stub}\nSKYCAP_RELAY_EOF\n"
        "PY=''; if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; sys.exit(sys.version_info < (3, 6))'; then PY=$(command -v python3); fi\n"
        "if [ -z \"$PY\" ]; then\n"
        "  if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then\n"
        "    { (apt-get update && apt-get install -y curl) || apk add --no-cache curl || yum install -y curl"
        " || dnf install -y curl; } >/dev/null 2>&1 || true\n"
        "  fi\n"
        # Images with curl but no CA bundle (curl: (77)): the installer's download needs one.
        "  if [ ! -s /etc/ssl/certs/ca-certificates.crt ] && command -v apt-get >/dev/null 2>&1; then\n"
        "    (apt-get update && apt-get install -y ca-certificates) >/dev/null 2>&1 || true\n"
        "  fi\n"
        # UV_UNMANAGED_INSTALL: into that directory only. The installer otherwise adds a line to the task's
        # ~/.profile and ~/.bashrc (a change to the task's sandbox that the model can see, and has deleted).
        f"  export UV_UNMANAGED_INSTALL={RELAY_DIR}/bin UV_PYTHON_INSTALL_DIR={RELAY_DIR}/python\n"
        "  (command -v curl >/dev/null && curl -LsSf https://astral.sh/uv/0.7.13/install.sh || wget -qO- https://astral.sh/uv/0.7.13/install.sh) | sh >/dev/null 2>&1\n"
        f"  {RELAY_DIR}/bin/uv python install 3.12 >/dev/null 2>&1; PY=$({RELAY_DIR}/bin/uv python find 3.12)\n"
        "fi\n"
        # Under a name of its own: the model's `pkill python3` / `killall python` then leaves the stub alone (in
        # CalibForge runs that took the agent's only route to the model down in ~0.5% of trials).
        f"ln -sf \"$PY\" {RELAY_DIR}/relay-stub 2>/dev/null && PY={RELAY_DIR}/relay-stub\n"
        f"nohup \"$PY\" {RELAY_DIR}/stub.py --port {port} --secret {shlex.quote(secret)} > {RELAY_DIR}/stub.log 2>&1 < /dev/null &\n"
        f"for i in $(seq 1 50); do \"$PY\" -c \"import urllib.request as u; r=u.Request('http://127.0.0.1:{port}/_relay/health', headers={{'X-Relay-Secret': '{secret}'}}); u.urlopen(r, timeout=2)\" 2>/dev/null && {{ echo RELAY_STUB_UP; exit 0; }}; sleep 0.2; done\n"
        f"cat {RELAY_DIR}/stub.log; exit 1"
    )


class RelayDaytonaEnvironment(LabelledDaytonaEnvironment):
    def __init__(
        self, *args: Any, skycap_relay: Optional[Dict[str, Any]] = None, ephemeral: bool = True, **kwargs: Any
    ) -> None:
        """``skycap_relay``: the relay manager that polls this sandbox, the skycap server its calls go to, and the
        stub's port. ``ephemeral``: create ephemeral sandboxes (some Daytona orgs permit nothing else; Harbor sets it
        on its snapshot path only)."""
        super().__init__(*args, **kwargs)
        self._relay = skycap_relay
        self._ephemeral = ephemeral
        self._relay_key: Optional[str] = None

    async def _create_sandbox(self, params: Any, daytona: Any = None) -> None:
        if self._ephemeral:
            params.ephemeral = True
        await super()._create_sandbox(params=params, daytona=daytona)

    async def start(self, force_build: bool) -> None:
        await super().start(force_build)
        if self._relay:
            await self._start_relay()

    async def stop(self, delete: bool):
        if self._relay and self._relay_key:
            try:
                await self._relay_control("/detach", {"key": self._relay_key})
            except Exception as e:  # noqa: BLE001 - the manager drops a gone sandbox on its own
                self.logger.warning("sandbox relay detach failed: %r", e)
        return await super().stop(delete)

    async def _relay_control(self, route: str, body: Dict[str, Any]) -> None:
        import aiohttp

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post(self._relay["manager"] + route, json=body) as response:
                if response.status != 200:
                    raise RuntimeError(f"relay manager {route}: HTTP {response.status} {await response.text()}")

    async def _start_relay(self) -> None:
        """Start the stub, then hand its signed preview URL to the relay manager. Fails the trial if the stub doesn't
        come up: the agent would have no way to reach the model. As root, before the agent's install."""
        port = int(self._relay["port"])
        secret = secrets.token_hex(16)
        result = await self.exec(stub_start_command(port, secret, _stub_source()), timeout_sec=300, user="root")
        if "RELAY_STUB_UP" not in (result.stdout or ""):
            raise RuntimeError(f"sandbox relay stub did not start: {result.stdout} {result.stderr}")
        ttl_minutes = self._ttl_minutes or DEFAULT_PREVIEW_TTL_MINUTES
        preview = await self._sandbox.create_signed_preview_url(port, expires_in_seconds=ttl_minutes * 60)
        self._relay_key = f"{self._sandbox.id}:{port}"
        await self._relay_control(
            "/attach",
            {"key": self._relay_key, "preview_url": preview.url, "secret": secret, "target": self._relay["target"]},
        )
