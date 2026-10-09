"""Harbor trials, captured by skycap.

The sibling ``harbor`` integration asks Harbor for per-turn token ids
(``collect_rollout_details``), which is why it has to ban summarization: a
rewritten history breaks the harness's token accounting. Here Harbor runs
unmodified, in text space. Each trial is pointed at its own skycap trajectory
URL; skycap renders every prompt, calls the engine with token ids, and keeps a
context graph, so a rewritten history is a branch rather than a hole and
summarization is allowed.

Per trial: create a trajectory, point the agent at it, run it, and ``finish``
with the reward to get a sample per path the ``train_paths`` rule picks.
``compose`` turns those into the step-wise ``GeneratorOutput``.

Terminus-2 calls the model from this process and gets ``trajectory.base_url``.
An installed agent (mini-swe-agent, Claude Code, ...) calls it from inside its
sandbox, so it gets the trajectory's route on the server's exposed URL
(``trajectory.exposed_base_url``, with ``skycap.exposure``), through the
sandbox environment that LiteLLM- and OpenAI-based agents read, as the model
``openai/<served name>``. The exposed URL's host is added to the agent phase's
network allowlist (``agent.extra_allowed_hosts``) for tasks that have one.
"""

import asyncio
import importlib
import os
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import litellm
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.models.agent.name import AgentName
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from loguru import logger
from skycap import CapturePool, PathRuleError
from skycap.paths import load_rule
from skycap.trajectory import CONTEXT_LENGTH_EXCEEDED
from tqdm import tqdm

from skyrl.backends.skyrl_train.inference_servers.base import ConversationType
from skyrl.train.generators.base import (
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
    TrajectoryID,
)
from skyrl.train.generators.utils import build_vllm_cache_salt
from skyrl.train.utils.rate_limiter import create_rate_limiter

from ..harbor.trial_metrics import TrialAttempts, trial_metrics
from .compose import TrialOutcome, compose, split
from .record_index import RecordLog

litellm.suppress_debug_info = True

# Attempts per trial. A Harbor failure is often environmental (a sandbox that didn't come up).
# Attempts per trial. An eval that must score like a single `harbor run` sets SKYCAP_TRIAL_ATTEMPTS=1.
MAX_NUM_RETRIES_PER_TRIAL = int(os.environ.get("SKYCAP_TRIAL_ATTEMPTS", "2"))

#: skycap authenticates nothing, but LiteLLM won't build a client without a key.
PLACEHOLDER_API_KEY = "skycap"

#: The sandbox environment an installed agent's LiteLLM/OpenAI client is pointed at its trajectory with, besides
#: the URL and key: mini-swe-agent carries on without a price for the model, and LiteLLM uses its bundled cost
#: map rather than fetching one, which an agent phase allowed only skycap's host couldn't.
SANDBOX_ENV_DEFAULTS = {"MSWEA_COST_TRACKING": "ignore_errors", "LITELLM_LOCAL_MODEL_COST_MAP": "True"}


class UnexposedSkycapError(RuntimeError):
    """An agent that runs in its sandbox was given no exposed URL: a configuration error, never retried."""


class HarborSkycapGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: Any,
        harbor_cfg: Dict[str, Any],
        capture_urls: List[str],
        inference_engine_client: Any = None,
        train_paths: str = "all",
        records: Optional[RecordLog] = None,
    ) -> None:
        """
        Args:
            generator_cfg: the run's generator config.
            harbor_cfg: Harbor's ``TrialConfig`` template.
            capture_urls: the skycap servers to spread trajectories over.
            inference_engine_client: read for its ``weight_version``, which keys the prefix-cache salt.
            train_paths: the skycap path rule every trajectory is finished with: ``all``, ``final``, or a custom
                rule's ``"pkg.module:function"``, which the servers must have been started with.
            records: where every trajectory opened is logged, per phase, for the W&B record index.
        """
        # Imported here too, so a bad import path fails at startup rather than at the first finish.
        load_rule(train_paths)
        self.train_paths = train_paths
        if not getattr(generator_cfg, "step_wise_trajectories", False):
            raise ValueError(
                "HarborSkycapGenerator emits one row per captured path, grouped per rollout the step-wise way. "
                "Set generator.step_wise_trajectories=true."
            )
        if getattr(generator_cfg, "merge_stepwise_output", False):
            raise ValueError(
                "Set generator.merge_stepwise_output=false: each row is already a complete multi-turn path, and "
                "prefix merging could fuse two paths that merely share a prefix."
            )
        self.generator_cfg = generator_cfg
        self._routed_experts = bool(getattr(generator_cfg.inference_engine, "enable_return_routed_experts", False))
        self.capture_urls = list(capture_urls)
        self.pool = CapturePool(self.capture_urls)
        self.inference_engine_client = inference_engine_client
        self.records = records
        served = generator_cfg.inference_engine.served_model_name
        if served is None or "/" in served:
            raise ValueError("generator.inference_engine.served_model_name must be set, without '/'")
        self._served_model_name = served

        self._template = deepcopy(harbor_cfg)
        agent = self._template.setdefault("agent", {})
        self._in_sandbox = runs_in_sandbox(agent)
        if self._in_sandbox:
            # Harbor's mini-swe-agent refuses to start unless this process's environment holds a key, which it
            # copies into the sandbox; the trial's own key in the agent's env overrides it there.
            os.environ.setdefault("MSWEA_API_KEY", PLACEHOLDER_API_KEY)
        agent["model_name"] = model_name(served, self._in_sandbox)
        kwargs = agent.setdefault("kwargs", {})
        # skycap has the tokens exactly; asking Harbor for them too is what forces the sibling to ban summarization.
        kwargs.pop("collect_rollout_details", None)
        self._rate_limiter = create_rate_limiter(getattr(generator_cfg, "rate_limit", None))

    async def close(self) -> None:
        """Close the pool's HTTP session. The generator can't generate afterwards."""
        await self.pool.close()

    def _cache_salt(self) -> Optional[str]:
        if not getattr(self.generator_cfg, "use_cache_salt", False):
            return None
        weight_version = getattr(self.inference_engine_client, "weight_version", None)
        if weight_version is None:
            return None
        return build_vllm_cache_salt(weight_version, self._served_model_name)

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        outcomes, attempts = await self.run_trials(input_batch, disable_tqdm=disable_tqdm)
        output = compose(
            outcomes,
            overlong_filtering=self.generator_cfg.apply_overlong_filtering,
            top_k=self.generator_cfg.sampling_params.top_k,
            sample_support=getattr(self.generator_cfg.inference_engine, "enable_return_sample_support_set", False),
            routed_experts=self._routed_experts,
        )
        output["rollout_metrics"].update(trial_metrics(attempts))
        return output

    async def run_trials(
        self, input_batch: GeneratorInput, disable_tqdm: bool = False
    ) -> Tuple[List[TrialOutcome], List[TrialAttempts]]:
        """Every trial of the batch, one outcome each (retries included), in the batch's order."""
        prompts = input_batch["prompts"]
        trajectory_ids = input_batch["trajectory_ids"]
        if trajectory_ids is None:
            raise ValueError("`trajectory_ids` is required in the input batch")
        if len(prompts) != len(trajectory_ids):
            raise ValueError(f"Prompt count ({len(prompts)}) doesn't match trajectory_ids ({len(trajectory_ids)})")
        metadata = input_batch.get("batch_metadata")
        step = getattr(metadata, "global_step", None)
        phase = getattr(metadata, "training_phase", "train")
        cache_salt = self._cache_salt()

        outcomes: List[Optional[TrialOutcome]] = [None] * len(prompts)
        attempts = [TrialAttempts() for _ in prompts]
        progress = tqdm(
            disable=disable_tqdm,
            total=len(prompts),
            desc="Generating Trajectories",
            miniters=max(1, len(prompts) // 10),
            mininterval=5,
        )

        async def worker(index: int, prompt: ConversationType, trajectory_id: TrajectoryID) -> None:
            outcomes[index] = await self._trial(prompt, trajectory_id, cache_salt, step, phase, attempts[index])
            progress.update(1)

        try:
            async with asyncio.TaskGroup() as group:
                for index, (prompt, trajectory_id) in enumerate(zip(prompts, trajectory_ids)):
                    group.create_task(worker(index, prompt, trajectory_id))
        finally:
            progress.close()
        return outcomes, attempts

    async def _trial(
        self,
        prompt: ConversationType,
        trajectory_id: TrajectoryID,
        cache_salt: Optional[str],
        step: Optional[int],
        phase: str,
        attempts: TrialAttempts,
    ) -> TrialOutcome:
        """One rollout, retried on unknown errors. Never raises: one failure must not cancel the batch.

        A path rule that raises is not retried: the rule runs on what the trial captured, so another run of the
        trial would mostly pay for the sandbox and the agent again to fail the same way. The rollout is masked.
        """
        started = time.monotonic()
        missing_routes = False
        for attempt in range(MAX_NUM_RETRIES_PER_TRIAL):
            prefix = f"Trajectory {trajectory_id} attempt {attempt + 1}/{MAX_NUM_RETRIES_PER_TRIAL}"
            attempts.start()
            try:
                outcome = await self._attempt(prompt, trajectory_id, cache_salt, step, phase, attempt, attempts)
            except PathRuleError as error:
                logger.error(f"{prefix}: path rule {self.train_paths!r} failed, not retrying: {error}")
                attempts.fail(error)
                break
            except UnexposedSkycapError:
                raise
            except Exception as error:  # noqa: BLE001 - retried, then masked
                logger.warning(f"{prefix} failed: {type(error).__name__}: {error}")
                attempts.fail(error)
                continue
            outcome.e2e_time = time.monotonic() - started
            if outcome.stop_reason != "error":
                return outcome
            missing_routes = missing_routes or outcome.missing_routes
            logger.warning(f"{prefix} produced nothing to train on")
        return TrialOutcome(
            trajectory_id=trajectory_id,
            stop_reason="error",
            e2e_time=time.monotonic() - started,
            missing_routes=missing_routes,
        )

    async def _attempt(
        self,
        prompt: ConversationType,
        trajectory_id: TrajectoryID,
        cache_salt: Optional[str],
        step: Optional[int],
        phase: str,
        attempt: int,
        attempts: TrialAttempts,
    ) -> TrialOutcome:
        """One attempt on its own trajectory, so a retry never shares a graph with the attempt it replaces."""
        meta = {
            "task": str(prompt),
            "instance_id": str(trajectory_id.instance_id),
            "repetition_id": trajectory_id.repetition_id,
            "step": step,
            "attempt": attempt,
        }
        trajectory = None
        try:
            async with self.pool.trajectory(meta, paths=self.train_paths) as trajectory:
                config = self._trial_config(prompt, self._agent_url(trajectory), cache_salt, trajectory.api_key)
                async with self._rate_limiter:
                    results = await (await Trial.create(TrialConfig.model_validate(config))).run()

                attempts.record(results)
                exception = results.exception_info.exception_type if results.exception_info else None
                # As stock Harbor scores the trial: the verifier runs after an agent timeout or a failed agent too.
                verifier_reward = _verifier_reward(results)
                if exception == "AgentTimeoutError":
                    # Masked, not retried, as the sibling does.
                    reward, stop_reason = 0.0, "agent_timeout"
                elif exception == "ContextLengthExceededError":
                    # Trains with reward 0, as the sibling does.
                    reward, stop_reason = 0.0, "context_length"
                elif exception == "NonZeroAgentExitCodeError" and await _context_length_exceeded(trajectory):
                    # An installed agent has no ContextLengthExceededError of its own: skycap refused its prompt
                    # as too long, its LiteLLM raised ContextWindowExceededError, and it exited non-zero. A
                    # context-length stop like Terminus-2's (overlong filtering applies), scored by the verifier
                    # Harbor still ran, as stock Harbor would.
                    reward, stop_reason = verifier_reward or 0.0, "context_length"
                elif not results.verifier_result:
                    reward, stop_reason = 0.0, "error"
                    logger.warning(f"Trajectory {trajectory_id} has no verifier result: {results.exception_info}")
                else:
                    reward, stop_reason = float(results.verifier_result.rewards["reward"]), "complete"
                finished = await trajectory.finish({"reward": reward, "stop_reason": stop_reason})
        finally:
            # Every attempt that got a trajectory, including one that raised: the pool finished it.
            if self.records is not None and trajectory is not None:
                self.records.add(phase, trajectory_id, attempt, trajectory)

        # How stock Harbor scored the trial, kept whatever it trains as (for evaluation).
        scored = dict(
            verifier_reward=verifier_reward,
            exception=exception,
            trial_name=getattr(results, "trial_name", None),
            skycap_id=trajectory.id,
        )
        if finished.status != "finished":
            # The trajectory failed inside skycap (e.g. an unattributable prompt): its samples may miss a turn.
            logger.warning(f"Trajectory {trajectory_id}: skycap status {finished.status!r}, not training on it")
            return TrialOutcome(trajectory_id=trajectory_id, stop_reason="error", **scored)
        if not any(split(sample) is not None for sample in finished.samples):
            # Nothing generated (e.g. the harness failed before its first call): retried, then masked,
            # so the trial's reward doesn't enter the group without tokens behind it.
            logger.warning(f"Trajectory {trajectory_id}: skycap captured no trainable tokens")
            return TrialOutcome(trajectory_id=trajectory_id, stop_reason="error", **scored)
        # Overlong filtering clears a context-length trial's loss mask, so it trains nothing and needs no routes.
        filtered = stop_reason == "context_length" and self.generator_cfg.apply_overlong_filtering
        if (
            self._routed_experts
            and not filtered
            and any((row := split(sample)) is not None and row.routes is None for sample in finished.samples)
        ):
            # skycap drops a path's routes when any node on it lacks them; replaying the rest would
            # train those tokens on routes the rollout never took. Retried, then masked.
            logger.warning(f"Trajectory {trajectory_id}: a trained path has no routed experts")
            return TrialOutcome(trajectory_id=trajectory_id, stop_reason="error", missing_routes=True, **scored)
        return TrialOutcome(
            trajectory_id=trajectory_id,
            samples=finished.samples,
            reward=reward,
            stop_reason=stop_reason,
            unbridged_calls=finished.unbridged_calls,
            **scored,
        )

    def _agent_url(self, trajectory: Any) -> str:
        """Where the agent calls its trajectory: on the server's exposed URL when it runs in its sandbox."""
        if not self._in_sandbox:
            return trajectory.base_url
        if trajectory.exposed_base_url is None:
            # A sandbox can't reach the server's own URL. One on this network (Docker) can use external_host
            # with this node's address.
            raise UnexposedSkycapError(
                f"{self._template['agent'].get('name') or self._template['agent'].get('import_path')} runs inside "
                "its sandbox, but skycap isn't exposed: set skycap.exposure.type"
            )
        return trajectory.exposed_base_url

    def _trial_config(
        self,
        prompt: ConversationType,
        base_url: str,
        cache_salt: Optional[str],
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """The trial's Harbor config. ``api_key`` is the trajectory's own key (``skycap.require_api_key``)."""
        config = deepcopy(self._template)
        config["task"] = {"path": prompt}
        key = api_key or PLACEHOLDER_API_KEY
        if self._in_sandbox:
            # Installed agents read their endpoint and key from the sandbox's environment, not from agent kwargs:
            # LiteLLM's `openai/` provider OPENAI_API_BASE, the OpenAI SDK OPENAI_BASE_URL. mini-swe-agent sends
            # MSWEA_API_KEY over a provider's key, so it carries the trajectory's key too.
            env = config["agent"].setdefault("env", {})
            for provider in ("OPENAI", "HOSTED_VLLM"):
                env[f"{provider}_API_BASE"] = base_url
                env[f"{provider}_API_KEY"] = key
            env["OPENAI_BASE_URL"] = base_url
            env["MSWEA_API_KEY"] = key
            for name, value in SANDBOX_ENV_DEFAULTS.items():
                env.setdefault(name, value)
            # A task whose agent phase is allowlisted must let the agent reach skycap. Harbor ignores the
            # extra host (with a warning) for a task whose agent phase has the public network.
            host = urlsplit(base_url).hostname
            if host is not None and ":" not in host:
                allowed = config["agent"].setdefault("extra_allowed_hosts", [])
                if host not in allowed:
                    allowed.append(host)
        kwargs = config["agent"]["kwargs"]
        kwargs["api_base"] = base_url
        llm_kwargs = kwargs.setdefault("llm_kwargs", {})
        # Terminus-2 takes `api_base` itself but passes a key only through `llm_kwargs`.
        llm_kwargs["api_key"] = key
        if cache_salt is not None:
            # LiteLLM merges `extra_body` into the request body, where skycap reads `cache_salt` and forwards it.
            extra_body = llm_kwargs.setdefault("extra_body", {})
            if not isinstance(extra_body, dict):
                raise TypeError("harbor_trial_config.agent.kwargs.llm_kwargs.extra_body must be a mapping")
            extra_body["cache_salt"] = cache_salt
        return config


def model_name(served: str, in_sandbox: bool) -> str:
    """The LiteLLM model Harbor hands the agent.

    An installed agent gets ``openai/``: LiteLLM's OpenAI provider reads ``OPENAI_API_BASE`` and sends an
    assistant message back as the agent replays it, ``reasoning_content`` included, so a thinking model's
    turns extend each other; and a stock Harbor run against vLLM uses it too, so the agent behaves alike in
    both. Terminus-2 keeps ``hosted_vllm/``, against which ``skycap.use_raw_content`` is tuned.
    """
    return f"openai/{served}" if in_sandbox else f"hosted_vllm/{served}"


def _verifier_reward(results: Any) -> Optional[float]:
    """The verifier's reward, or None when it didn't run."""
    verifier = getattr(results, "verifier_result", None)
    rewards = getattr(verifier, "rewards", None) if verifier else None
    if not rewards or rewards.get("reward") is None:
        return None
    return float(rewards["reward"])


async def _context_length_exceeded(trajectory: Any) -> bool:
    """Whether skycap refused one of the trajectory's prompts as too long for the model's context."""
    document = await trajectory.document()
    return any(failure.get("code") == CONTEXT_LENGTH_EXCEEDED for failure in document.get("failures") or ())


def runs_in_sandbox(agent: Dict[str, Any]) -> bool:
    """Whether Harbor's agent ``agent`` (a ``TrialConfig.agent``) is installed in the sandbox and calls the model
    from there, as mini-swe-agent and Claude Code do, rather than from this process, as Terminus-2 does."""
    # As Harbor picks the agent (AgentFactory.create_agent_from_config): a known name wins over an import path.
    name, import_path = agent.get("name"), agent.get("import_path")
    if name is not None and name in AgentName.values():
        cls = AgentFactory._AGENT_MAP.get(AgentName(name))
    elif import_path:
        module, _, attribute = import_path.partition(":")
        cls = getattr(importlib.import_module(module), attribute)
    else:
        return False
    return isinstance(cls, type) and issubclass(cls, BaseInstalledAgent)
