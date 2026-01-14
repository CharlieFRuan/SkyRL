import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from loguru import logger
from uuid import uuid4
from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput, TrajectoryID
from skyrl_train.generators.utils import get_rollout_metrics, get_response_ids_and_loss_mask_from_messages
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import ConversationType
from skyrl_train.utils.reward_shaping import shape_reward_from_output
from omegaconf import DictConfig
from pathlib import Path

# Harbor orchestrator and trial imports
from harbor.orchestrators.queue import QueueOrchestrator
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult

# Schema-driven Harbor config mapping
from examples.terminal_bench.harbor_config import HarborConfigBuilder

@dataclass
class TerminalBenchAgentOutput:
    response_ids: List[int]
    reward: float
    stop_reason: str
    loss_mask: List[int]
    prompt_ids: List[int]
    trajectory_id: TrajectoryID
    summarization_count: Optional[int] = None

class TerminalBenchGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: DictConfig,
        terminal_bench_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
    ):
        """
        Args:
            generator_cfg: DictConfig object containing the generator configuration
            terminal_bench_cfg: DictConfig object containing the terminal bench configuration
            inference_engine_client: InferenceEngineClient object for interacting with the inference engines
            tokenizer: tokenizer object for encoding and decoding text
        """
        self.base_url = f"http://{generator_cfg.http_endpoint_host}:{generator_cfg.http_endpoint_port}"
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = generator_cfg.model_name

        # Core terminal bench config
        self.trials_dir = terminal_bench_cfg.trials_dir

        # Schema-driven Harbor config builder
        # Automatically maps YAML fields to Harbor's TrialConfig with validation
        self._harbor_config_builder = HarborConfigBuilder(terminal_bench_cfg)

        # Configure Harbor log level (default WARNING to reduce noise)
        harbor_log_level = self._harbor_config_builder.get_log_level(default="WARNING")
        self._configure_harbor_logging(harbor_log_level)

        # Store model_info for external access (e.g., metrics)
        self.model_info = self._harbor_config_builder.model_info

        # Build retry config for QueueOrchestrator (handles backoff, exception filtering)
        self._retry_config = self._harbor_config_builder.build_retry_config()
        self._n_concurrent_trials = self._harbor_config_builder.get_n_concurrent_trials(
            default=16  # Reasonable default for parallel trial execution
        )

        # Reward shaping config (parses test output for partial credit)
        self._reward_shaping_config = self._harbor_config_builder.get_reward_shaping_config()

        logger.info(
            f"TerminalBenchGenerator initialized with HarborConfigBuilder. "
            f"Exposed fields: {list(self._harbor_config_builder._harbor_cfg.keys())}. "
            f"Retry config: max_retries={self._retry_config.max_retries}, "
            f"backoff={self._retry_config.min_wait_sec}-{self._retry_config.max_wait_sec}s. "
            f"Concurrent trials: {self._n_concurrent_trials}. "
            f"Reward shaping: enabled={self._reward_shaping_config.get('enable_reward_shaping', True)}, "
            f"shaper={self._reward_shaping_config.get('reward_shaper', 'pass_ratio')}"
        )

        # Read custom chat template
        custom_chat_template_path = generator_cfg.engine_init_kwargs.get("custom_chat_template_chat_completion_path", None)
        if custom_chat_template_path:
            with open(custom_chat_template_path, "r") as f:
                self.custom_chat_template_content = f.read()
            logger.info(f"TerminalBenchGenerator initialized with custom chat template read from: {custom_chat_template_path}")
        else:
            self.custom_chat_template_content = None

    def _configure_harbor_logging(self, level: str) -> None:
        """
        Configure Harbor's logging level.

        Args:
            level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        log_level = getattr(logging, level.upper(), logging.WARNING)

        # Set level for Harbor's main logger and all child loggers
        harbor_loggers = [
            "harbor",
            "harbor.trial",
            "harbor.agents",
            "harbor.verifier",
            "harbor.orchestrators",
            "harbor.environments",
            "harbor.utils.logger",
        ]

        for logger_name in harbor_loggers:
            logging.getLogger(logger_name).setLevel(log_level)

        # Also set the root harbor logger
        logging.getLogger("harbor").setLevel(log_level)

        logger.info(f"Harbor logging level set to {level}")

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """
        Generate rollouts for a batch of prompts using Harbor's QueueOrchestrator.

        The QueueOrchestrator handles:
        - Concurrency control (n_concurrent_trials workers)
        - Retry logic with exponential backoff
        - Exception filtering (retry transient errors, skip permanent ones)
        """
        num_trials = len(input_batch["prompts"])
        logger.info(f"Starting batch generation for {num_trials} trials")

        # Build all TrialConfigs upfront
        trial_configs: List[TrialConfig] = []
        trajectory_ids: List[TrajectoryID] = []

        # Harbor expects hosted_vllm model names with exactly one '/'.
        # Convert HuggingFace-style "org/model" to just "model" for the alias.
        model_alias = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name

        for i in range(num_trials):
            prompt = input_batch["prompts"][i]
            trajectory_id = input_batch["trajectory_ids"][i]

            # Generate session_id for sticky routing to inference engines
            session_id = uuid4().hex

            trial_config = self._harbor_config_builder.build_trial_config(
                task_path=prompt,
                trials_dir=self.trials_dir,
                model_name=f"hosted_vllm/{model_alias}",
                api_base=f"{self.base_url}/v1",
                session_id=session_id,
            )
            trial_configs.append(trial_config)
            trajectory_ids.append(trajectory_id)

        # Create QueueOrchestrator with retry config and concurrency control
        orchestrator = QueueOrchestrator(
            trial_configs=[],  # We'll submit dynamically
            n_concurrent_trials=min(self._n_concurrent_trials, num_trials),
            metrics={},  # SkyRL handles its own metrics
            quiet=True,
            retry_config=self._retry_config,
        )

        # Start the orchestrator worker pool
        await orchestrator.start()

        try:
            # Submit all trials and collect futures
            futures = await orchestrator.submit_batch(trial_configs)

            # Wait for all trials to complete
            results: List[TrialResult | Exception] = await asyncio.gather(
                *futures, return_exceptions=True
            )
        finally:
            # Always shutdown the orchestrator
            await orchestrator.shutdown(wait=True)

        # Process results into TerminalBenchAgentOutput
        all_outputs: List[TerminalBenchAgentOutput] = []
        for i, result in enumerate(results):
            trajectory_id = trajectory_ids[i]
            output = self._process_trial_result(result, trajectory_id)
            all_outputs.append(output)

        # For a group of trajectories (n_samples_per_prompt trajectories for the same prompt), if one
        # of the trajectories fails, we skip the entire group. We also skip the group for rollout metric aggregation
        failed_instance_ids = set()
        num_failed_trajectories = 0  # per-trajectory, rather than per-instance
        successful_outputs: List[TerminalBenchAgentOutput] = []  # only for metrics purpose
        for output in all_outputs:
            if output.stop_reason == "error":
                failed_instance_ids.add(output.trajectory_id.instance_id)
                num_failed_trajectories += 1

        for output in all_outputs:
            if output.trajectory_id.instance_id in failed_instance_ids:
                output.response_ids = [0]
                output.stop_reason = "error"
                output.loss_mask = [0]
                output.prompt_ids = [0]
                output.reward = 0
            else:
                successful_outputs.append(output)

        # Calculate rollout metrics for successful outputs
        if len(successful_outputs) > 0:
            rollout_metrics = get_rollout_metrics(
                [output.response_ids for output in successful_outputs],
                [output.reward for output in successful_outputs],
            )
            rollout_metrics["generate/trajectories_summarized"] = sum(1 for output in successful_outputs if output.summarization_count > 0)
            rollout_metrics["generate/trajectories_truncated"] = sum(1 for output in successful_outputs if output.stop_reason == "length")
        else:
            rollout_metrics = {}
        rollout_metrics["generate/num_failed_instances"] = len(failed_instance_ids)
        rollout_metrics["generate/num_failed_trajectories"] = num_failed_trajectories

        logger.info(
            f"Batch generation complete: {num_trials - num_failed_trajectories}/{num_trials} successful, "
            f"{len(failed_instance_ids)} failed instances"
        )

        generator_output: GeneratorOutput = {
            "prompt_token_ids": [output.prompt_ids for output in all_outputs],
            "response_ids": [output.response_ids for output in all_outputs],
            "rewards": [output.reward for output in all_outputs],
            "loss_masks": [output.loss_mask for output in all_outputs],
            "stop_reasons": [output.stop_reason for output in all_outputs],
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
        }

        return generator_output

    def _process_trial_result(
        self,
        result: TrialResult | Exception,
        trajectory_id: TrajectoryID,
    ) -> TerminalBenchAgentOutput:
        """
        Process a TrialResult from QueueOrchestrator into TerminalBenchAgentOutput.

        Args:
            result: TrialResult from Harbor or an Exception if the trial failed completely.
            trajectory_id: The trajectory ID for this trial.

        Returns:
            TerminalBenchAgentOutput with processed rollout data.
        """
        # Handle exceptions from the orchestrator
        if isinstance(result, Exception):
            logger.warning(f"Trajectory {trajectory_id} failed with exception: {result}")
            return TerminalBenchAgentOutput(
                response_ids=[0],
                reward=0,
                stop_reason="error",
                loss_mask=[0],
                prompt_ids=[0],
                trajectory_id=trajectory_id,
            )

        # Check for missing verifier result (trial ran but didn't produce valid output)
        if not result.verifier_result:
            logger.warning(
                f"Trajectory {trajectory_id} failed: No verifier result. "
                f"Exception info: {result.exception_info}"
            )
            return TerminalBenchAgentOutput(
                response_ids=[0],
                reward=0,
                stop_reason="error",
                loss_mask=[0],
                prompt_ids=[0],
                trajectory_id=trajectory_id,
            )

        # Extract data from successful trial
        try:
            original_reward = result.verifier_result.rewards["reward"]
            chat_history = result.agent_result.metadata['all_messages']
            summarization_count = result.agent_result.metadata['summarization_count']
        except (KeyError, AttributeError, TypeError) as e:
            logger.warning(
                f"Trajectory {trajectory_id} failed: Could not extract results. "
                f"Error: {e}, Result: {result}"
            )
            return TerminalBenchAgentOutput(
                response_ids=[0],
                reward=0,
                stop_reason="error",
                loss_mask=[0],
                prompt_ids=[0],
                trajectory_id=trajectory_id,
            )

        # Apply reward shaping if enabled
        if self._reward_shaping_config.get("enable_reward_shaping", True):
            verifier_stdout = getattr(result.verifier_result, "stdout", None)
            reward = shape_reward_from_output(
                stdout=verifier_stdout,
                original_reward=original_reward,
                parser_name=self._reward_shaping_config.get("reward_parser"),
                shaper_name=self._reward_shaping_config.get("reward_shaper", "pass_ratio"),
                shaper_kwargs=self._reward_shaping_config.get("shaper_kwargs", {}),
                fallback_to_original=self._reward_shaping_config.get("reward_shaping_fallback", True),
            )
            if reward != original_reward:
                logger.debug(
                    f"Trajectory {trajectory_id}: reward shaped {original_reward:.3f} -> {reward:.3f}"
                )
        else:
            reward = original_reward

        # Validate chat history structure
        if not chat_history or len(chat_history) < 2 or chat_history[0]["role"] != "user":
            logger.warning(
                f"Trajectory {trajectory_id} failed: Invalid chat history structure. "
                f"chat_history: {chat_history}"
            )
            return TerminalBenchAgentOutput(
                response_ids=[0],
                reward=0,
                stop_reason="error",
                loss_mask=[0],
                prompt_ids=[0],
                trajectory_id=trajectory_id,
            )

        # Process successful trial
        # Use the first message as the prompt (assume no system messages)
        prompt = [chat_history[0]]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=False,
            tokenize=True,
            chat_template=self.custom_chat_template_content,
        )
        initial_prompt_length = len(prompt_ids)

        # Process response messages (everything after the first message)
        response_messages = chat_history[1:]
        assistant_logprobs = getattr(result.agent_result, "output_logprobs", None)
        response_ids, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
            response_messages, self.tokenizer, assistant_logprobs, custom_chat_template=self.custom_chat_template_content
        )

        # Determine stop reason
        max_response_tokens = (
            self.generator_cfg.sampling_params.max_generate_length
            + self.generator_cfg.max_input_length
            - initial_prompt_length
        )
        stop_reason = "complete"  # Default for trial completion
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"

        # Truncate to maximum allowed length
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]

        return TerminalBenchAgentOutput(
            response_ids=response_ids,
            reward=reward,
            stop_reason=stop_reason,
            loss_mask=loss_mask,
            prompt_ids=prompt_ids,
            trajectory_id=trajectory_id,
            summarization_count=summarization_count,
        )
