"""Agents inside remote sandboxes reach skycap through its exposure: the config, the per-server
exposures, and the URL and environment the generator hands each kind of agent."""

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("skycap")
pytest.importorskip("harbor")

from examples.train_integrations.harbor_skycap.engine import SkyRLEngine  # noqa: E402
from examples.train_integrations.harbor_skycap.entrypoints.main_harbor_skycap import (  # noqa: E402
    HarborSkycapConfig,
    _exposure,
    _require_api_key,
)
from examples.train_integrations.harbor_skycap.harbor_generator import (  # noqa: E402
    PLACEHOLDER_API_KEY,
    HarborSkycapGenerator,
    UnexposedSkycapError,
    runs_in_sandbox,
)
from examples.train_integrations.harbor_skycap.servers import exposure_for  # noqa: E402
from skycap import CaptureService  # noqa: E402
from skycap.exposure import Exposure  # noqa: E402
from tests.integrations.harbor_skycap import test_harbor_skycap  # noqa: E402
from tests.integrations.harbor_skycap.fakes import TOP_K, FakeRenderer  # noqa: E402
from tests.integrations.harbor_skycap.test_harbor_skycap import (  # noqa: E402
    batch,
    generator_cfg,
    harbor_cfg,
)

pytestmark = pytest.mark.integrations

# The Harbor tests' fixtures: a mock SkyRL router, and Harbor's Trial replaced by the fake.
router = test_harbor_skycap.router
trials = test_harbor_skycap.trials


class LoopbackExposure(Exposure):
    """Exposes the harness listener at its own loopback URL: the agent reaches it, the control plane doesn't."""

    def start(self, harness_url: str) -> str:
        return harness_url


class FailingExposure(Exposure):
    """An exposure whose way in can't open."""

    def start(self, harness_url: str) -> str:
        raise RuntimeError("no way in")


@pytest.fixture
def exposed(router, tmp_path):
    service = CaptureService(
        router.url,
        mode="tokens",
        renderer=FakeRenderer(),
        engine=SkyRLEngine(),
        model="policy",
        sampling_overrides={"top_k": TOP_K},
        record_dir=str(tmp_path / "record"),
        host="127.0.0.1",
        exposure=LoopbackExposure(),
        require_api_key=True,
    )
    service.start()
    yield service
    service.stop()


def agent_cfg(name: str) -> dict:
    cfg = harbor_cfg()
    cfg["agent"]["name"] = name
    return cfg


# -- config ------------------------------------------------------------------------
def test_nothing_is_exposed_by_default() -> None:
    cfg = HarborSkycapConfig.from_cli_overrides([])
    assert cfg.skycap.exposure.type == "none" and _exposure(cfg) is None


def test_the_exposure_is_configured_by_name_and_kwargs() -> None:
    cfg = HarborSkycapConfig.from_cli_overrides(
        ["skycap.exposure.type=external_host", "skycap.exposure.kwargs.host=203.0.113.7"]
    )
    assert _exposure(cfg) == "external_host"
    assert cfg.skycap.exposure.kwargs == {"host": "203.0.113.7"}


@pytest.mark.parametrize(
    "overrides, match",
    [
        (["skycap.exposure.type=cloudfare"], "is one of"),
        (["skycap.exposure.type=external_host"], "takes"),
        (["skycap.exposure.type=cloudflare", "skycap.exposure.kwargs.region=us"], "takes"),
        (["skycap.exposure.kwargs.host=203.0.113.7"], "type is none"),
    ],
)
def test_a_bad_exposure_is_refused_before_anything_starts(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        _exposure(HarborSkycapConfig.from_cli_overrides(overrides))


def test_keys_are_required_whenever_the_servers_are_exposed() -> None:
    def required(*overrides: str) -> bool:
        return _require_api_key(HarborSkycapConfig.from_cli_overrides(list(overrides)))

    assert not required()
    assert required("skycap.exposure.type=cloudflare")
    assert not required("skycap.exposure.type=cloudflare", "skycap.require_api_key=false")
    assert required("skycap.require_api_key=true")


def test_each_server_gets_its_own_exposure() -> None:
    assert exposure_for(None, None, 2) is None
    assert exposure_for("cloudflare", {"timeout": 30.0}, 2) == ("cloudflare", {"timeout": 30.0})
    # external_host: server i listens on port + i, from 11500 unless a port is given.
    assert exposure_for("external_host", {"host": "h"}, 0) == ("external_host", {"host": "h", "port": 11500})
    assert exposure_for("external_host", {"host": "h", "port": 12000}, 3) == (
        "external_host",
        {"host": "h", "port": 12003},
    )
    # Any exposure given a port binds one per server.
    assert exposure_for(RELAY, {"relay_host": "r", "port": "11500"}, 2) == (RELAY, {"relay_host": "r", "port": 11502})


# -- which agents are in a sandbox -----------------------------------------------------
def test_installed_agents_run_in_the_sandbox_and_terminus_does_not() -> None:
    assert runs_in_sandbox({"name": "mini-swe-agent"}) and runs_in_sandbox({"name": "claude-code"})
    assert not runs_in_sandbox({"name": "terminus-2"}) and not runs_in_sandbox({})
    assert runs_in_sandbox({"import_path": "harbor.agents.installed.mini_swe_agent:MiniSweAgent"})
    assert not runs_in_sandbox({"import_path": "harbor.agents.terminus_2:Terminus2"})
    # As Harbor decides: a known name wins over an import path, as with the default config's terminus-2.
    mini = "harbor.agents.installed.mini_swe_agent:MiniSweAgent"
    assert not runs_in_sandbox({"name": "terminus-2", "import_path": mini})
    assert runs_in_sandbox({"name": "not-an-agent-name", "import_path": mini})


#: The HTTPS relay exposure, by import path as ``skycap.exposure.type`` names it.
RELAY = "examples.train_integrations.harbor_skycap.relay_exposure:RelayPathExposure"


def test_the_relay_exposure_binds_its_port_and_is_reached_by_path_on_the_relay(monkeypatch) -> None:
    from examples.train_integrations.harbor_skycap import relay_exposure
    from skycap.exposure import load_exposure

    cfg = HarborSkycapConfig.from_cli_overrides(
        [
            f"skycap.exposure.type={RELAY}",
            "skycap.exposure.kwargs.relay_host=relay.example",
            "skycap.exposure.kwargs.port=11500",
        ]
    )
    assert _exposure(cfg) == RELAY and _require_api_key(cfg)
    name, kwargs = exposure_for(RELAY, dict(cfg.skycap.exposure.kwargs), 3)
    exposure = load_exposure(name, **kwargs)
    assert exposure.bind() == ("0.0.0.0", 11503)
    # The node's address is Ray's, or the default route's; one given pins it.
    monkeypatch.setattr(relay_exposure, "node_address", lambda: "10.180.15.202")
    assert exposure.start("http://127.0.0.1:11503") == "https://relay.example/skycap/10.180.15.202/11503"
    pinned = load_exposure(name, **kwargs, node_ip="10.0.0.9")
    assert pinned.start("http://127.0.0.1:11503") == "https://relay.example/skycap/10.0.0.9/11503"
    assert not relay_exposure.node_address().startswith("127.")
    for bad in ({"relay_host": "https://relay.example", "port": 1}, {"relay_host": "r", "port": 0}):
        with pytest.raises(ValueError):
            relay_exposure.RelayPathExposure(**bad)


def test_raw_content_defaults_per_agent_and_an_installed_agent_needs_an_exposure() -> None:
    from examples.train_integrations.harbor_skycap.entrypoints.main_harbor_skycap import (
        _check_agent_reach,
        _use_raw_content,
    )

    terminus = HarborSkycapConfig.from_cli_overrides(["harbor_trial_config.agent.name=terminus-2"])
    mini = HarborSkycapConfig.from_cli_overrides(["harbor_trial_config.agent.name=mini-swe-agent"])
    assert _use_raw_content(terminus) and not _use_raw_content(mini)
    mini.skycap.use_raw_content = True
    assert _use_raw_content(mini)
    _check_agent_reach(terminus)
    with pytest.raises(ValueError, match="skycap.exposure.type"):
        _check_agent_reach(mini)
    mini.skycap.exposure.type = "cloudflare"
    _check_agent_reach(mini)


def test_the_servers_get_the_renderer_and_thinking_options(monkeypatch) -> None:
    from examples.train_integrations.harbor_skycap.entrypoints import main_harbor_skycap

    started = []
    monkeypatch.setattr(main_harbor_skycap, "start_servers", lambda settings, **_: started.append(settings))
    cfg = HarborSkycapConfig.from_cli_overrides(
        [
            "harbor_trial_config.agent.name=mini-swe-agent",
            "trainer.algorithm.max_seq_len=1024",
            "skycap.renderer=qwen3.6",
            "skycap.chat_template_kwargs.enable_thinking=true",
            "skycap.chat_template_kwargs.preserve_thinking=true",
        ]
    )
    main_harbor_skycap.start_skycap(cfg, "http://router")
    (settings,) = started
    assert settings["chat_template_kwargs"] == {"enable_thinking": True, "preserve_thinking": True}
    assert settings["thinking_retention"] == "all" and settings["renderer_name"] == "qwen3.6"
    assert settings["use_raw_content"] is False


# -- the URL each agent gets ----------------------------------------------------------------
@pytest.mark.asyncio
async def test_an_agent_in_the_sandbox_calls_skycap_on_its_exposed_url(exposed, trials, monkeypatch) -> None:
    monkeypatch.delenv("MSWEA_API_KEY", raising=False)
    cfg = agent_cfg("mini-swe-agent")
    cfg["agent"]["extra_allowed_hosts"] = ["example.org"]
    gen = HarborSkycapGenerator(generator_cfg(), cfg, [exposed.url], SimpleNamespace(weight_version=7))
    try:
        out = await gen.generate(batch("linear", "summarize"), disable_tqdm=True)
    finally:
        await gen.close()

    assert out["rewards"] and all(sum(mask) > 0 for mask in out["loss_masks"])
    for config in trials.configs:
        env = config["agent"]["env"]
        url = env["OPENAI_API_BASE"]
        # The trajectory's route on the exposed listener, never the server's own URL.
        assert url.startswith(f"{exposed.exposed_url}/t/") and not url.startswith(exposed.url)
        assert env["HOSTED_VLLM_API_BASE"] == env["OPENAI_BASE_URL"] == config["agent"]["kwargs"]["api_base"] == url
        # LiteLLM's OpenAI provider, as a stock Harbor run against vLLM; mini-swe-agent prices nothing.
        assert config["agent"]["model_name"] == "openai/policy"
        assert env["MSWEA_COST_TRACKING"] == "ignore_errors" and env["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
        # An agent phase with an allowlist lets the agent reach skycap's host, besides what was configured.
        assert config["agent"]["extra_allowed_hosts"] == ["example.org", "127.0.0.1"]
        # The server requires keys: each agent got its own trajectory's key, which it called with.
        key = env["OPENAI_API_KEY"]
        assert key.startswith("sk-skycap-") and env["HOSTED_VLLM_API_KEY"] == env["MSWEA_API_KEY"] == key
    assert len({c["agent"]["env"]["OPENAI_API_KEY"] for c in trials.configs}) == len(trials.configs)
    # Without a key (a server that requires none), the placeholder; Harbor's mini-swe-agent, which checks this
    # process's environment for a key before it starts, finds one.
    env = gen._trial_config("/tasks/t", "https://edge.example/t/tr_x/v1", None)["agent"]["env"]
    assert env["OPENAI_API_KEY"] == env["HOSTED_VLLM_API_KEY"] == PLACEHOLDER_API_KEY
    assert os.environ["MSWEA_API_KEY"] == PLACEHOLDER_API_KEY


@pytest.mark.asyncio
async def test_terminus_keeps_the_servers_own_url_when_it_is_exposed(exposed, trials) -> None:
    gen = HarborSkycapGenerator(
        generator_cfg(), agent_cfg("terminus-2"), [exposed.url], SimpleNamespace(weight_version=7)
    )
    try:
        out = await gen.generate(batch("linear"), disable_tqdm=True)
    finally:
        await gen.close()

    assert out["rewards"] == [1.0]
    (config,) = trials.configs
    assert config["agent"]["kwargs"]["api_base"].startswith(f"{exposed.url}/t/")
    assert "OPENAI_API_BASE" not in (config["agent"].get("env") or {})
    assert config["agent"]["model_name"] == "hosted_vllm/policy" and not config["agent"]["extra_allowed_hosts"]
    # Its key is the trajectory's, through llm_kwargs, and the keyed server took its calls.
    assert config["agent"]["kwargs"]["llm_kwargs"]["api_key"].startswith("sk-skycap-")


@pytest.mark.asyncio
async def test_without_exposure_an_agent_in_the_sandbox_is_refused(router, tmp_path, trials) -> None:
    service = CaptureService(
        router.url, mode="tokens", renderer=FakeRenderer(), engine=SkyRLEngine(), model="policy",
        sampling_overrides={"top_k": TOP_K}, host="127.0.0.1",
    )  # fmt: skip
    service.start()
    gen = HarborSkycapGenerator(generator_cfg(), agent_cfg("mini-swe-agent"), [service.url])
    try:
        # A configuration error: the batch fails rather than every trial being retried and masked.
        with pytest.raises(ExceptionGroup) as raised:
            await gen.generate(batch("linear"), disable_tqdm=True)
    finally:
        await gen.close()
        service.stop()
    assert raised.group_contains(UnexposedSkycapError)
    assert trials.configs == []
