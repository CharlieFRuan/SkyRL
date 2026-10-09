"""Train on Harbor tasks, with skycap capturing each rollout's exact tokens.

The sibling ``main_harbor`` with two changes: a pool of skycap servers (Ray
actors, see ``servers.py``) starts in front of the inference router, and the
generator points each trial at its own trajectory on one of them. Agents that
run inside their sandbox (mini-swe-agent, ...) reach it through
``skycap.exposure``.

    uv run --isolated --extra fsdp --extra harbor --extra skycap \\
        -m examples.train_integrations.harbor_skycap.entrypoints.main_harbor_skycap \\
        trainer.policy.model.path=Qwen/Qwen3-8B generator.inference_engine.served_model_name=policy \\
        generator.step_wise_trajectories=true data.train_data="['/path/to/harbor/tasks']" ...
"""

import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import ray
import yaml
from loguru import logger
from skycap.exposure import load_exposure
from skycap.paths import BUILTIN_RULES, load_rule

from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from ...harbor.entrypoints.main_harbor import (
    HARBOR_DEFAULT_CONFIG,
    HarborExp,
    HarborSkyRLConfig,
    _deep_merge,
)
from ..harbor_generator import HarborSkycapGenerator, runs_in_sandbox
from ..record_index import RecordLog, SkycapRecordIndex
from ..servers import SkycapServers, exposure_for, start_servers


@dataclass
class ExposureConfig:
    type: str = "none"
    """How agents that run inside a remote sandbox (Harbor's installed agents: mini-swe-agent, Claude Code, ...)
    reach skycap; Terminus-2 calls from this cluster and never needs it. ``none`` (default); ``cloudflare``, a
    Cloudflare quick tunnel per server (development: at most 200 calls in flight per tunnel, ~125 s to a
    response's first byte); ``external_host``, server ``i`` listens on ``kwargs.port + i`` (default 11500) on its
    node and is reached at ``kwargs.host``: a relay that forwards each port to its server's node (frp), or the
    node's own address with every server on that node (``skycap.placement_strategy=STRICT_PACK``); or an
    ``Exposure`` subclass, ``"pkg.module:Class"``.
    Only the harness routes are exposed; see skycap's README."""
    kwargs: Dict[str, Any] = field(default_factory=dict)
    """The exposure's constructor arguments: ``host`` and ``port`` for ``external_host``, ``timeout`` and
    ``attempts`` for ``cloudflare``, or a custom class's own."""


@dataclass
class SkycapWandbConfig:
    enabled: bool = True
    """Index each step's skycap records in W&B when ``trainer.logger`` is wandb.

    One version per step of the artifact ``skycap-records-<phase>-<run id>``, aliased ``<phase>-step-N`` and
    ``latest``, holding a ``step.json`` (every attempt, trained or superseded, and where its record is) and,
    for records in ``record_mirror``, a reference to each record file. No record bytes are uploaded.
    Logging runs off the step and fails open."""
    phases: List[str] = field(default_factory=lambda: ["train"])
    """The training phases to index: ``train``, ``eval``. Each gets its own artifact."""


@dataclass
class SkycapConfig:
    num_servers: int = 1
    """skycap servers, one Ray actor each. The generator spreads trajectories over them round-robin."""
    num_cpus_per_server: float = 2.0
    """CPUs reserved per server, for rendering and HTTP."""
    placement_strategy: str = "SPREAD"
    """Ray placement-group strategy for the servers: SPREAD (default, for availability), STRICT_SPREAD, PACK."""
    record_dir: Optional[str] = None
    """Where ended trajectories are written. Defaults to ``{trainer.export_path}/skycap``; each server writes
    on its own node, so point it at a shared filesystem to have one directory for the run."""
    record_mirror: Optional[str] = None
    """Where every server also copies its records, as an fsspec URL (``s3://bucket/prefix``).

    The copy is made in the background after the record is written to ``record_dir``, and fails open: a slow
    or failing store never fails a rollout. Needs the store's fsspec implementation (``s3fs``, ``gcsfs``)."""
    record_mirror_config: Dict[str, Any] = field(default_factory=dict)
    """The mirror's options (``skycap.mirror.RecordMirror``), e.g. ``{exclude: [experts, sampling_mask]}`` to
    leave sidecars out of the remote copy, or ``timeout``, ``attempts``, ``queue_size``, ``storage_options``.
    Needs ``record_mirror``."""
    wandb: SkycapWandbConfig = field(default_factory=SkycapWandbConfig)
    """The per-step index of the records in W&B."""
    ttl: float = 3600.0
    """Seconds an open trajectory may be idle before skycap writes it as abandoned and releases it."""
    renderer_pool_size: int = 8
    """Renderers (tokenizer copies) each server renders prompts with in parallel."""
    renderer: Optional[str] = None
    """The ``renderers`` renderer skycap renders with (``qwen3.6``, ``qwen3``, ...). By default it is picked from
    ``trainer.policy.model.path`` when that is a Hub name it knows; a local checkout's path needs it named."""
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)
    """The chat template's options for skycap's renderer, e.g. ``{enable_thinking: true, preserve_thinking: true}``
    for Qwen3.6 with its thinking kept in the history. skycap renders every prompt itself, so a harness's own
    ``chat_template_kwargs`` (in ``extra_body``) don't apply."""
    thinking_retention: Optional[str] = "all"
    """What skycap's renderer does with earlier turns' thinking. ``all`` (default) keeps it: each turn extends the
    previous prompt and completion token for token. ``tool_cycle`` drops it once a new user query arrives, as
    Qwen's template does by default, re-rendering the prompt. None follows ``chat_template_kwargs`` (Qwen3.6's
    ``preserve_thinking``); an explicit value that contradicts them is refused."""
    use_raw_content: Optional[bool] = None
    """Answer with the completion's own text as ``content``, a thinking model's reasoning inline, as
    SkyRL's vLLM does (it runs no reasoning parser). Terminus-2 replays ``content``, and LiteLLM's
    ``hosted_vllm/`` provider strips ``reasoning_content`` from what it sends back; with parsed
    replies every replayed turn would lose its reasoning, edit history and fork the graph.
    None (default) picks per agent: on for one that calls the model from this process (Terminus-2), off for
    one installed in its sandbox (mini-swe-agent), which acts through tool calls and needs them parsed, as a
    vLLM with reasoning and tool-call parsers would; its LiteLLM (``openai/``) replays ``reasoning_content``."""
    max_tokens_policy: str = "clamp"
    """When a call's ``max_tokens`` doesn't fit the room the prompt leaves in ``max_model_len``: ``clamp`` samples into
    the room left; ``reject`` refuses it as a stock vLLM endpoint does (context-length error), for a parity eval."""
    train_paths: str = "all"
    """Which captured paths train (skycap's path rule). ``all``: every root-to-leaf path of a rollout's graph,
    each sampled message trained once, so a reply the harness discarded and asked again for trains with the
    rollout's advantage too. ``final``: only the path to the reply of the rollout's last model call, the
    conversation the harness ended with: one row per rollout, and nothing off it trains. Or a custom rule, ``"pkg.module:function"``:
    a function of skycap's ``MessageGraph`` to ``skycap.paths.Row``s (a path and the model nodes on it to
    train), importable on every node; the skycap servers are started with it."""
    exposure: ExposureConfig = field(default_factory=ExposureConfig)
    require_api_key: Optional[bool] = None
    """Whether a trajectory's harness routes answer only its own key, which the generator hands its agent. ``None``
    (default): whenever ``skycap.exposure`` is set, so routes reachable from outside the cluster can't be written to
    by whoever learns a URL."""
    """How agents inside remote sandboxes reach the servers."""


@dataclass
class HarborSkycapConfig(HarborSkyRLConfig):
    skycap: SkycapConfig = field(default_factory=SkycapConfig)


def start_skycap(cfg: Any, engine_url: str) -> SkycapServers:
    """skycap servers in token mode, in front of SkyRL's router."""
    ie = cfg.generator.inference_engine
    sampling = cfg.generator.sampling_params
    engine_init = dict(ie.engine_init_kwargs or {})
    train_paths = cfg.skycap.train_paths
    # Here first, so a rule that won't import fails on the driver rather than in every server actor.
    load_rule(train_paths)
    settings = {
        "upstream_url": engine_url,
        "tokenizer": cfg.trainer.policy.model.path,
        "renderer_pool_size": cfg.skycap.renderer_pool_size,
        "model": ie.served_model_name,
        "max_model_len": engine_init.get("max_model_len") or cfg.trainer.algorithm.max_seq_len,
        # The trainer computes logprobs with these, so every rollout is sampled with them,
        # whatever the harness asks for.
        "sampling_overrides": {
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "min_p": sampling.min_p,
        },
        "sampling_mask": ie.enable_return_sample_support_set,
        "use_raw_content": _use_raw_content(cfg),
        "chat_template_kwargs": dict(cfg.skycap.chat_template_kwargs) if cfg.skycap.chat_template_kwargs else None,
        "thinking_retention": cfg.skycap.thinking_retention,
        "renderer_name": cfg.skycap.renderer,
        "max_tokens_policy": cfg.skycap.max_tokens_policy,
        "require_api_key": _require_api_key(cfg),
        # A custom rule is imported by each server, under the name the generator finishes with.
        "path_rules": {} if train_paths in BUILTIN_RULES else {train_paths: train_paths},
        "record_mirror": cfg.skycap.record_mirror,
        "record_mirror_config": dict(cfg.skycap.record_mirror_config or {}) or None,
    }
    return start_servers(
        settings,
        num_servers=cfg.skycap.num_servers,
        num_cpus_per_server=cfg.skycap.num_cpus_per_server,
        placement_strategy=cfg.skycap.placement_strategy,
        record_dir=cfg.skycap.record_dir or os.path.join(cfg.trainer.export_path, "skycap"),
        ttl=cfg.skycap.ttl,
        exposure=_exposure(cfg),
        exposure_kwargs=dict(cfg.skycap.exposure.kwargs),
    )


def _use_raw_content(cfg: Any) -> bool:
    """``skycap.use_raw_content``, by default on unless the agent runs inside its sandbox."""
    if cfg.skycap.use_raw_content is not None:
        return bool(cfg.skycap.use_raw_content)
    return not runs_in_sandbox(cfg.harbor_trial_config.get("agent") or {})


def _check_agent_reach(cfg: Any) -> None:
    """An agent installed in its sandbox reaches skycap only through an exposure: refuse to start without one."""
    agent = cfg.harbor_trial_config.get("agent") or {}
    if cfg.skycap.exposure.type == "none" and runs_in_sandbox(agent):
        raise ValueError(
            f"Harbor agent {agent.get('name') or agent.get('import_path')!r} calls the model from inside its "
            "sandbox, which can't reach skycap's own URL: set skycap.exposure.type (external_host, cloudflare, "
            "or an Exposure class such as relay_exposure:RelayPathExposure)"
        )


def _require_api_key(cfg: Any) -> bool:
    """``skycap.require_api_key``, on by default whenever the servers are exposed."""
    required = cfg.skycap.require_api_key
    return cfg.skycap.exposure.type != "none" if required is None else bool(required)


def _exposure(cfg: Any) -> Optional[str]:
    """``skycap.exposure.type``, or None for ``none``. Raises ``ValueError`` on a config that can't be built."""
    kind = cfg.skycap.exposure.type
    if kind == "none":
        if cfg.skycap.exposure.kwargs:
            raise ValueError("skycap.exposure.kwargs is set but skycap.exposure.type is none")
        return None
    # Built here once, as the first server's would be, so a bad config fails before any actor starts.
    name, kwargs = exposure_for(kind, dict(cfg.skycap.exposure.kwargs), 0)
    load_exposure(name, **kwargs)
    return kind


class HarborSkycapExp(HarborExp):
    skycap: Optional[SkycapServers] = None
    generator: Optional[HarborSkycapGenerator] = None
    records: Optional[RecordLog] = None

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        if self.skycap is None:
            self.skycap = start_skycap(cfg, inference_engine_client.get_endpoint_url())
        if self.records is None and cfg.skycap.wandb.enabled and cfg.trainer.logger == "wandb":
            self.records = RecordLog()
        self.generator = HarborSkycapGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,
            capture_urls=self.skycap.urls,
            inference_engine_client=inference_engine_client,
            train_paths=cfg.skycap.train_paths,
            records=self.records,
        )
        return self.generator

    def get_trainer(self, *args, **kwargs):
        trainer = super().get_trainer(*args, **kwargs)
        if self.records is not None:
            trainer.add_callback(SkycapRecordIndex(self.records, self.cfg.skycap.wandb.phases))
        return trainer

    def run(self):
        try:
            super().run()
        finally:
            if self.skycap is not None:
                logger.info("stopping skycap, writing the trajectories still in memory")
                self.skycap.stop()
            if self.generator is not None:
                asyncio.run(self.generator.close())


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    HarborSkycapExp(cfg).run()


def main() -> None:
    cfg = HarborSkycapConfig.from_cli_overrides(sys.argv[1:])
    with open(HARBOR_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)
    validate_cfg(cfg)
    _exposure(cfg)
    _check_agent_reach(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError("trainer.algorithm.max_seq_len must be set for Harbor training")
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
