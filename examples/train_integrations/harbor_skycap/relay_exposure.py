"""Exposing skycap's harness routes through an HTTPS relay that routes by path.

``RelayPathExposure`` binds the harness listener on ``0.0.0.0:<port>`` of the
server's node, as ``external_host`` does, but agents reach it at

    https://<relay_host>/skycap/<node ip>/<port>

on a public relay (a small VM with a TLS certificate) that forwards each such
prefix to ``<node ip>:<port>`` over the private network and strips the prefix,
so skycap sees its own ``/t/{trajectory id}/v1/...`` routes. One relay hostname
serves every node and port, and the sandbox only ever talks HTTPS to it: a
sandbox with an agent-phase allowlist needs only ``<relay_host>`` allowed (the
generator adds it to ``agent.extra_allowed_hosts``).

Use it with ``skycap.exposure.type`` set to this class's import path:

    skycap.exposure.type=examples.train_integrations.harbor_skycap.relay_exposure:RelayPathExposure
    skycap.exposure.kwargs.relay_host=relay.example.com
    skycap.exposure.kwargs.port=11500

Server ``i`` binds ``port + i`` (``servers.exposure_for``). ``node_ip`` defaults
to the address Ray knows the server's node by; set it only to pin one.

The relay, with Caddy (prefix stripped, no timeouts, responses flushed as they
come, since a turn can take minutes)::

    relay.example.com {
        @skycap path_regexp skycap ^/skycap/(10\\.0\\.0\\.(?:1|2))/(115[0-4][0-9])(/.*)$
        handle @skycap {
            rewrite * {re.skycap.3}
            reverse_proxy {re.skycap.1}:{re.skycap.2} {
                flush_interval -1
                transport http {
                    dial_timeout 30s
                    response_header_timeout 0
                    read_timeout 0
                    write_timeout 0
                }
            }
        }
        respond 404
    }

or, for one node and port, ``handle_path /skycap/10.0.0.1/11500/* { reverse_proxy
10.0.0.1:11500 { flush_interval -1 } }``. Restrict the regexp to your nodes'
addresses and ports, so the relay is not an open proxy into the network. Only
the harness routes are served on the listener, and with ``skycap.require_api_key``
(on whenever an exposure is set) each trajectory's routes answer only its own key.
"""

import ipaddress
import socket
from typing import Optional

from skycap.exposure import Exposure

#: The path prefix the relay routes on.
PREFIX = "skycap"


class RelayPathExposure(Exposure):
    def __init__(self, relay_host: str, port: int, node_ip: Optional[str] = None, scheme: str = "https") -> None:
        if not relay_host or "/" in relay_host:
            raise ValueError(f"relay_host must be a host name, not {relay_host!r}")
        port = int(port)
        if not 0 < port < 65536:
            raise ValueError(f"RelayPathExposure needs a TCP port, got {port}")
        if scheme not in ("http", "https"):
            raise ValueError(f"scheme must be http or https, not {scheme!r}")
        self.relay_host = relay_host
        self.port = port
        self.node_ip = node_ip
        self.scheme = scheme

    def bind(self) -> tuple[str, int]:
        return "0.0.0.0", self.port

    def start(self, harness_url: str) -> str:
        node_ip = self.node_ip or node_address()
        return f"{self.scheme}://{self.relay_host}/{PREFIX}/{node_ip}/{self.port}"


def node_address() -> str:
    """This node's address on the cluster network: Ray's, or else the one the default route leaves from."""
    try:
        import ray

        if ray.is_initialized():
            return ray.util.get_node_ip_address()
    except Exception:  # noqa: BLE001 - fall back to the routing table
        pass
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        # No packet is sent: connecting a UDP socket only picks the outgoing interface.
        probe.connect(("10.255.255.255", 1))
        address = probe.getsockname()[0]
    if ipaddress.ip_address(address).is_loopback:
        raise RuntimeError("can't tell this node's address; set skycap.exposure.kwargs.node_ip")
    return address
