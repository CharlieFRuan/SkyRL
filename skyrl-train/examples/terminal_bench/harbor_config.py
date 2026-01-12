"""
Schema-driven Harbor configuration mapping for SkyRL terminal bench.

This module provides automatic mapping from YAML config to Harbor's TrialConfig,
with validation and warnings for unknown/unsupported fields.

Usage:
    from examples.terminal_bench.harbor_config import HarborConfigBuilder

    builder = HarborConfigBuilder(terminal_bench_cfg)
    trial_config = builder.build_trial_config(
        task_path=prompt,
        trials_dir=self.trials_dir,
        agent_name="terminus",
        model_name="hosted_vllm/Qwen3-8B",
        api_base="http://localhost:8000/v1",
        session_id=session_id,
    )
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Set

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from harbor.models.trial.config import (
    TrialConfig,
    AgentConfig,
    TaskConfig,
    EnvironmentConfig,
    VerifierConfig,
)
from harbor.models.job.config import RetryConfig
from harbor.models.environment_type import EnvironmentType
from harbor.models.agent.name import AgentName


# =============================================================================
# Schema Definition: Which Harbor fields are exposed in SkyRL YAML
# =============================================================================
#
# This schema defines the mapping between YAML config keys and Harbor's
# Pydantic models. To expose a new Harbor field:
#   1. Add it to the appropriate section below
#   2. That's it - the mapping is automatic
#
# Field types:
#   - "direct": Maps directly to a Pydantic model field
#   - "kwargs": Passed through agent.kwargs dict (agent-specific params)
#
# =============================================================================

@dataclass
class FieldMapping:
    """Defines how a YAML field maps to Harbor config."""
    harbor_field: str  # Field name in Harbor's Pydantic model
    field_type: str = "direct"  # "direct" or "kwargs"
    default: Any = None  # Default value if not specified


@dataclass
class SectionSchema:
    """Schema for a Harbor config section (agent, environment, etc.)."""
    fields: Dict[str, FieldMapping] = field(default_factory=dict)

    def get_all_field_names(self) -> Set[str]:
        return set(self.fields.keys())


# Agent config fields
AGENT_SCHEMA = SectionSchema(
    fields={
        # Direct fields on AgentConfig
        "override_timeout_sec": FieldMapping("override_timeout_sec"),
        "override_setup_timeout_sec": FieldMapping("override_setup_timeout_sec"),
        "max_timeout_sec": FieldMapping("max_timeout_sec"),
        # Kwargs fields (passed to agent.kwargs)
        "max_episodes": FieldMapping("max_episodes", field_type="kwargs", default=16),
        "enable_summarize": FieldMapping("enable_summarize", field_type="kwargs", default=True),
        "store_all_messages": FieldMapping("store_all_messages", field_type="kwargs", default=True),
    }
)

# Environment config fields
ENVIRONMENT_SCHEMA = SectionSchema(
    fields={
        "override_cpus": FieldMapping("override_cpus"),
        "override_memory_mb": FieldMapping("override_memory_mb"),
        "override_storage_mb": FieldMapping("override_storage_mb"),
        "override_gpus": FieldMapping("override_gpus"),
        "environment_type": FieldMapping("type"),  # Maps to EnvironmentConfig.type
    }
)

# Verifier config fields
VERIFIER_SCHEMA = SectionSchema(
    fields={
        "verifier_override_timeout_sec": FieldMapping("override_timeout_sec"),
        "verifier_max_timeout_sec": FieldMapping("max_timeout_sec"),
        "verifier_disable": FieldMapping("disable"),
    }
)

# Trial-level config fields
TRIAL_SCHEMA = SectionSchema(
    fields={
        "timeout_multiplier": FieldMapping("timeout_multiplier", default=1.0),
    }
)

# Retry config fields (for QueueOrchestrator)
RETRY_SCHEMA = SectionSchema(
    fields={
        "max_retries": FieldMapping("max_retries", default=2),
        "min_wait_sec": FieldMapping("min_wait_sec", default=1.0),
        "max_wait_sec": FieldMapping("max_wait_sec", default=60.0),
        "wait_multiplier": FieldMapping("wait_multiplier", default=2.0),
        # Exception filtering - comma-separated strings in YAML, converted to sets
        "include_exceptions": FieldMapping("include_exceptions"),
        "exclude_exceptions": FieldMapping("exclude_exceptions"),
    }
)

# Orchestrator config fields
ORCHESTRATOR_SCHEMA = SectionSchema(
    fields={
        "n_concurrent_trials": FieldMapping("n_concurrent_trials"),
    }
)

# Complete schema registry
HARBOR_SCHEMA = {
    "agent": AGENT_SCHEMA,
    "environment": ENVIRONMENT_SCHEMA,
    "verifier": VERIFIER_SCHEMA,
    "trial": TRIAL_SCHEMA,
    "retry": RETRY_SCHEMA,
    "orchestrator": ORCHESTRATOR_SCHEMA,
}


def _get_all_known_harbor_fields() -> Set[str]:
    """Get all field names that Harbor's Pydantic models accept."""
    known = set()
    # From AgentConfig
    known.update(AgentConfig.model_fields.keys())
    # From EnvironmentConfig
    known.update(EnvironmentConfig.model_fields.keys())
    # From VerifierConfig
    known.update(VerifierConfig.model_fields.keys())
    # From TrialConfig (excluding nested configs)
    known.update({"timeout_multiplier", "trial_name"})
    return known


def _get_all_exposed_fields() -> Set[str]:
    """Get all field names exposed in our schema."""
    exposed = set()
    for schema in HARBOR_SCHEMA.values():
        exposed.update(schema.get_all_field_names())
    return exposed


# =============================================================================
# HarborConfigBuilder: Main interface for building TrialConfig from YAML
# =============================================================================

class HarborConfigBuilder:
    """
    Builds Harbor TrialConfig from SkyRL YAML configuration.

    Provides automatic field mapping with validation and warnings.
    """

    def __init__(self, terminal_bench_cfg: DictConfig):
        """
        Initialize the builder with terminal bench configuration.

        Args:
            terminal_bench_cfg: The terminal_bench_config section from Hydra config.
        """
        self._cfg = terminal_bench_cfg
        self._warnings_issued: Set[str] = set()

        # Extract harbor-specific config if present, otherwise use flat structure
        # This supports both new nested style and legacy flat style
        if "harbor" in terminal_bench_cfg:
            self._harbor_cfg = OmegaConf.to_container(
                terminal_bench_cfg.harbor, resolve=True
            ) or {}
        else:
            # Legacy: extract harbor fields from flat config
            self._harbor_cfg = self._extract_harbor_fields_legacy(terminal_bench_cfg)

        # Extract model_info (special handling - nested dict passed to agent kwargs)
        model_info_cfg = terminal_bench_cfg.get("model_info", {})
        if isinstance(model_info_cfg, DictConfig):
            model_info_cfg = OmegaConf.to_container(model_info_cfg, resolve=True)
        self._model_info = {
            "max_input_tokens": model_info_cfg.get("max_input_tokens", 32768),
            "max_output_tokens": model_info_cfg.get("max_output_tokens", 8192),
            "input_cost_per_token": model_info_cfg.get("input_cost_per_token", 0),
            "output_cost_per_token": model_info_cfg.get("output_cost_per_token", 0),
        }

        # Validate config and issue warnings
        self._validate_config()

    def _extract_harbor_fields_legacy(self, cfg: DictConfig) -> Dict[str, Any]:
        """Extract harbor-related fields from legacy flat config structure."""
        harbor_fields = {}
        all_exposed = _get_all_exposed_fields()

        for key in all_exposed:
            if key in cfg and cfg[key] is not None:
                harbor_fields[key] = cfg[key]

        return harbor_fields

    def _validate_config(self) -> None:
        """Validate config and issue warnings for unknown/unsupported fields."""
        all_exposed = _get_all_exposed_fields()
        all_known_harbor = _get_all_known_harbor_fields()

        for key, value in self._harbor_cfg.items():
            if value is None:
                continue

            if key not in all_exposed:
                if key in all_known_harbor:
                    # Known Harbor field but not exposed in SkyRL
                    self._warn_once(
                        f"Harbor config '{key}' is a valid Harbor field but not exposed "
                        f"in SkyRL. Add to HARBOR_SCHEMA in harbor_config.py to enable."
                    )
                else:
                    # Completely unknown field
                    self._warn_once(
                        f"Unknown harbor config key '{key}' - ignoring. "
                        f"Check spelling or Harbor version compatibility."
                    )

    def _warn_once(self, message: str) -> None:
        """Issue a warning only once per message."""
        if message not in self._warnings_issued:
            self._warnings_issued.add(message)
            logger.warning(message)
            warnings.warn(message, UserWarning, stacklevel=3)

    def _get_field_value(
        self,
        yaml_key: str,
        mapping: FieldMapping,
        fallback_cfg: Optional[DictConfig] = None,
    ) -> Any:
        """Get field value from config with fallback to default."""
        # Check harbor config first
        if yaml_key in self._harbor_cfg:
            return self._harbor_cfg[yaml_key]

        # Check fallback (legacy flat config)
        if fallback_cfg is not None and yaml_key in fallback_cfg:
            value = fallback_cfg.get(yaml_key)
            if value is not None:
                return value

        # Return default
        return mapping.default

    def _build_agent_fields(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Build agent direct fields and kwargs from config."""
        direct_fields = {}
        kwargs_fields = {}

        for yaml_key, mapping in AGENT_SCHEMA.fields.items():
            value = self._get_field_value(yaml_key, mapping, self._cfg)
            if value is not None:
                if mapping.field_type == "kwargs":
                    kwargs_fields[mapping.harbor_field] = value
                else:
                    direct_fields[mapping.harbor_field] = value

        return direct_fields, kwargs_fields

    def _build_environment_config(self) -> EnvironmentConfig:
        """Build EnvironmentConfig from config."""
        env_fields = {}

        for yaml_key, mapping in ENVIRONMENT_SCHEMA.fields.items():
            value = self._get_field_value(yaml_key, mapping, self._cfg)
            if value is not None:
                if mapping.harbor_field == "type":
                    # Special handling for environment type
                    if isinstance(value, str):
                        value = EnvironmentType(value)
                env_fields[mapping.harbor_field] = value

        # Default to Daytona if not specified
        if "type" not in env_fields:
            env_fields["type"] = EnvironmentType.DAYTONA

        return EnvironmentConfig(**env_fields)

    def _build_verifier_config(self) -> VerifierConfig:
        """Build VerifierConfig from config."""
        verifier_fields = {}

        for yaml_key, mapping in VERIFIER_SCHEMA.fields.items():
            value = self._get_field_value(yaml_key, mapping, self._cfg)
            if value is not None:
                verifier_fields[mapping.harbor_field] = value

        return VerifierConfig(**verifier_fields)

    def _get_trial_fields(self) -> Dict[str, Any]:
        """Get trial-level fields from config."""
        trial_fields = {}

        for yaml_key, mapping in TRIAL_SCHEMA.fields.items():
            value = self._get_field_value(yaml_key, mapping, self._cfg)
            if value is not None:
                trial_fields[mapping.harbor_field] = value

        return trial_fields

    def build_retry_config(self) -> RetryConfig:
        """
        Build RetryConfig for QueueOrchestrator from YAML config.

        Returns:
            Configured RetryConfig with exponential backoff and exception filtering.
        """
        retry_fields = {}

        for yaml_key, mapping in RETRY_SCHEMA.fields.items():
            value = self._get_field_value(yaml_key, mapping, self._cfg)
            if value is not None:
                # Handle exception sets (YAML lists -> Python sets)
                if yaml_key in ("include_exceptions", "exclude_exceptions"):
                    if isinstance(value, (list, tuple)):
                        value = set(value)
                    elif isinstance(value, str):
                        # Support comma-separated string
                        value = {s.strip() for s in value.split(",") if s.strip()}
                retry_fields[mapping.harbor_field] = value

        return RetryConfig(**retry_fields)

    def get_n_concurrent_trials(self, default: int = 16) -> int:
        """
        Get the number of concurrent trials for QueueOrchestrator.

        Args:
            default: Default concurrency if not specified in config.

        Returns:
            Number of concurrent trials to run.
        """
        mapping = ORCHESTRATOR_SCHEMA.fields.get("n_concurrent_trials")
        if mapping:
            value = self._get_field_value("n_concurrent_trials", mapping, self._cfg)
            if value is not None:
                return int(value)
        return default

    def build_trial_config(
        self,
        task_path: str,
        trials_dir: str,
        agent_name: str,
        model_name: str,
        api_base: str,
        session_id: str,
    ) -> TrialConfig:
        """
        Build a complete TrialConfig for a Harbor trial.

        Args:
            task_path: Path to the task directory.
            trials_dir: Directory for trial outputs.
            agent_name: Agent type ("terminus" or "oracle").
            model_name: Model name for Harbor (e.g., "hosted_vllm/Qwen3-8B").
            api_base: Base URL for the inference API.
            session_id: Session ID for sticky routing.

        Returns:
            Configured TrialConfig ready for Trial execution.
        """
        # Build component configs
        environment_config = self._build_environment_config()
        verifier_config = self._build_verifier_config()
        agent_direct_fields, agent_kwargs = self._build_agent_fields()
        trial_fields = self._get_trial_fields()

        # Add required agent kwargs
        agent_kwargs.update({
            "api_base": api_base,
            "key": "fake_key",
            "session_id": session_id,
            "model_info": self._model_info,
        })

        # Determine agent name enum
        if agent_name == "terminus":
            harbor_agent_name = AgentName.TERMINUS_2.value
        elif agent_name == "oracle":
            harbor_agent_name = AgentName.ORACLE.value
        else:
            raise ValueError(f"Unknown agent name: {agent_name}")

        # Build AgentConfig
        agent_config = AgentConfig(
            name=harbor_agent_name,
            model_name=model_name,
            kwargs=agent_kwargs,
            **agent_direct_fields,
        )

        # Build TrialConfig
        return TrialConfig(
            task=TaskConfig(path=task_path),
            trials_dir=Path(trials_dir),
            environment=environment_config,
            verifier=verifier_config,
            agent=agent_config,
            **trial_fields,
        )

    @property
    def model_info(self) -> Dict[str, Any]:
        """Get the model_info dict for external use."""
        return self._model_info.copy()


# =============================================================================
# Utility functions
# =============================================================================

def get_exposed_harbor_fields() -> Dict[str, list[str]]:
    """
    Get a summary of all exposed Harbor fields for documentation.

    Returns:
        Dict mapping section names to lists of field names.
    """
    return {
        section_name: list(schema.get_all_field_names())
        for section_name, schema in HARBOR_SCHEMA.items()
    }


def print_harbor_schema() -> None:
    """Print the current Harbor schema for debugging."""
    print("SkyRL Terminal Bench - Exposed Harbor Fields")
    print("=" * 50)
    for section_name, schema in HARBOR_SCHEMA.items():
        print(f"\n{section_name.upper()}:")
        for yaml_key, mapping in schema.fields.items():
            field_type = f" (kwargs)" if mapping.field_type == "kwargs" else ""
            default = f" [default: {mapping.default}]" if mapping.default is not None else ""
            print(f"  - {yaml_key} -> {mapping.harbor_field}{field_type}{default}")
