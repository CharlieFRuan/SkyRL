"""``RelayDaytonaEnvironment`` for providers without preview URLs (Dune Cloud): the relay manager reaches the stub
through the exec API (``exec_transport.ExecRelayManager``) instead of a signed preview URL.

    harbor_trial_config.environment.import_path=examples.train_integrations.harbor_skycap.sandbox_relay.exec_environment:ExecRelayDaytonaEnvironment

``skycap_relay["manager"]`` must be an ``ExecRelayManager``'s URL, and that manager needs ``DAYTONA_API_URL`` /
``DAYTONA_API_KEY`` for the same provider account (it looks the sandbox up by id).
"""

import secrets

from .environment import RelayDaytonaEnvironment, _stub_source, stub_start_command


class ExecRelayDaytonaEnvironment(RelayDaytonaEnvironment):
    async def _start_relay(self) -> None:
        port = int(self._relay["port"])
        secret = secrets.token_hex(16)
        result = await self.exec(stub_start_command(port, secret, _stub_source()), timeout_sec=300, user="root")
        if "RELAY_STUB_UP" not in (result.stdout or ""):
            raise RuntimeError(f"sandbox relay stub did not start: {result.stdout} {result.stderr}")
        self._relay_key = f"{self._sandbox.id}:{port}"
        await self._relay_control(
            "/attach",
            {
                "key": self._relay_key,
                "sandbox_id": self._sandbox.id,
                "port": port,
                "secret": secret,
                "target": self._relay["target"],
            },
        )
