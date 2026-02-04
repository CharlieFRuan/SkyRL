"""
Built-in callbacks for common training operations.

These callbacks provide default implementations for checkpointing, evaluation,
model saving, and other periodic actions that were previously inline in the
training loop.

Supports two configuration styles:
1. Legacy interval configs (ckpt_interval, eval_interval, etc.)
2. New explicit callback configs in YAML:
   ```yaml
   trainer:
     callbacks:
       - type: checkpoint
         save_steps: 10
       - type: evaluation
         eval_steps: 20
   ```
"""

from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

from loguru import logger

from .base import TrainerCallback, TrainerState, TrainerControl, CallbackHandler

if TYPE_CHECKING:
    from omegaconf import DictConfig


# Registry mapping callback type names to classes
# This enables YAML-based callback configuration
CALLBACK_REGISTRY: Dict[str, Type[TrainerCallback]] = {}


def register_callback(name: str):
    """
    Decorator to register a callback class in the registry.

    Args:
        name: The type name to use in YAML configs (e.g., "checkpoint")

    Example:
        @register_callback("my_callback")
        class MyCallback(TrainerCallback):
            ...
    """
    def decorator(cls: Type[TrainerCallback]) -> Type[TrainerCallback]:
        CALLBACK_REGISTRY[name] = cls
        return cls
    return decorator


@register_callback("checkpoint")
class CheckpointCallback(TrainerCallback):
    """
    Callback for saving training checkpoints at regular intervals.

    This replaces the inline `ckpt_interval` logic in the training loop.
    Checkpoints include model weights, optimizer state, and training state
    for resumable training.

    Args:
        save_steps: Save a checkpoint every N steps. Set to -1 or 0 to disable.
        save_on_train_end: Whether to save a final checkpoint when training ends.
    """

    def __init__(self, save_steps: int = 10, save_on_train_end: bool = True):
        self.save_steps = save_steps
        self.save_on_train_end = save_on_train_end

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.save_steps > 0 and state.global_step % self.save_steps == 0:
            control.should_save = True
        return control

    def on_train_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.save_on_train_end and self.save_steps > 0:
            control.should_save = True
        return control


@register_callback("evaluation")
class EvaluationCallback(TrainerCallback):
    """
    Callback for running evaluation at regular intervals.

    This replaces the inline `eval_interval` logic in the training loop.
    Evaluation runs on the validation dataset and logs metrics.

    Args:
        eval_steps: Run evaluation every N steps. Set to -1 or 0 to disable.
        eval_on_train_end: Whether to run evaluation when training ends.
        eval_before_train: Whether to run evaluation before training starts.
    """

    def __init__(
        self,
        eval_steps: int = 5,
        eval_on_train_end: bool = True,
        eval_before_train: bool = True,
    ):
        self.eval_steps = eval_steps
        self.eval_on_train_end = eval_on_train_end
        self.eval_before_train = eval_before_train

    def on_train_begin(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.eval_before_train and self.eval_steps > 0:
            control.should_evaluate = True
        return control

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.eval_steps > 0:
            if state.global_step % self.eval_steps == 0 or state.is_last_step:
                control.should_evaluate = True
        return control


@register_callback("hf_model_save")
class HFModelSaveCallback(TrainerCallback):
    """
    Callback for saving models in HuggingFace format at regular intervals.

    This replaces the inline `hf_save_interval` logic in the training loop.
    HF format models can be loaded directly with transformers and pushed to
    the HuggingFace Hub.

    Args:
        save_steps: Save HF model every N steps. Set to -1 or 0 to disable.
        save_on_train_end: Whether to save final HF model when training ends.
    """

    def __init__(self, save_steps: int = -1, save_on_train_end: bool = True):
        self.save_steps = save_steps
        self.save_on_train_end = save_on_train_end

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.save_steps > 0 and state.global_step % self.save_steps == 0:
            control.should_save_hf_model = True
        return control

    def on_train_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.save_on_train_end and self.save_steps > 0:
            control.should_save_hf_model = True
        return control


@register_callback("ref_model_update")
class RefModelUpdateCallback(TrainerCallback):
    """
    Callback for updating the reference model with policy weights at epoch boundaries.

    This replaces the inline `update_ref_every_epoch` logic in the training loop.
    The reference model is used for KL divergence calculations in algorithms
    like PPO and GRPO.

    Args:
        update_every_epoch: Whether to update the reference model at the end of each epoch.
    """

    def __init__(self, update_every_epoch: bool = False):
        self.update_every_epoch = update_every_epoch
        self._should_update_ref = False

    def on_epoch_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        # Mark that we should update ref model
        # The actual update is handled by the trainer when it processes this flag
        if self.update_every_epoch and not state.is_last_step:
            # Skip updating ref at the end of the last epoch (as the original code did)
            self._should_update_ref = True
        return control

    @property
    def should_update_ref(self) -> bool:
        """Check if ref model should be updated and reset the flag."""
        result = self._should_update_ref
        self._should_update_ref = False
        return result


@register_callback("progress")
class ProgressCallback(TrainerCallback):
    """
    Callback for tracking and displaying training progress.

    This provides a central place for progress tracking without modifying
    the core training loop.

    Args:
        log_interval: Log progress every N steps. Default is every step.
    """

    def __init__(self, log_interval: int = 1):
        self.log_interval = log_interval

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.log_interval > 0 and state.global_step % self.log_interval == 0:
            logger.info(
                f"Step {state.global_step}/{state.total_steps} "
                f"(Epoch {state.epoch + 1}, Step {state.step_in_epoch})"
            )
        return control


@register_callback("logging")
class LoggingCallback(TrainerCallback):
    """
    Callback for logging metrics to tracking systems (WandB, MLflow).

    This callback handles the actual logging to external tracking systems.
    It's always enabled by default.

    Args:
        log_every_step: Whether to log after every step. Default True.
    """

    def __init__(self, log_every_step: bool = True):
        self.log_every_step = log_every_step

    def on_step_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if self.log_every_step:
            control.should_log = True
        return control


def create_default_callbacks(cfg: "DictConfig") -> List[TrainerCallback]:
    """
    Create the default set of callbacks based on trainer configuration.

    Supports two configuration styles:

    1. **New style** (explicit callbacks list):
       ```yaml
       trainer:
         callbacks:
           - type: checkpoint
             save_steps: 10
           - type: evaluation
             eval_steps: 20
       ```

    2. **Legacy style** (interval configs):
       ```yaml
       trainer:
         ckpt_interval: 10
         eval_interval: 20
       ```

    If explicit 'callbacks' config is present, it takes precedence.
    Otherwise, callbacks are created from legacy interval configs.

    Args:
        cfg: Training configuration (OmegaConf DictConfig)

    Returns:
        List of configured callbacks
    """
    # Check for new-style explicit callback configuration
    callbacks_config = getattr(cfg.trainer, "callbacks", None)
    if callbacks_config is not None and len(callbacks_config) > 0:
        logger.info("Using explicit callback configuration from YAML")
        callbacks = create_callbacks_from_config(cfg)
        # Always add logging callback if not explicitly configured
        has_logging = any(isinstance(cb, LoggingCallback) for cb in callbacks)
        if not has_logging:
            callbacks.append(LoggingCallback())
        return callbacks

    # Fall back to legacy interval-based configuration
    logger.debug("Using legacy interval-based callback configuration")
    callbacks = []

    # Checkpoint callback
    ckpt_interval = getattr(cfg.trainer, "ckpt_interval", 10)
    if ckpt_interval > 0:
        callbacks.append(CheckpointCallback(save_steps=ckpt_interval))

    # Evaluation callback
    eval_interval = getattr(cfg.trainer, "eval_interval", 5)
    eval_before_train = getattr(cfg.trainer, "eval_before_train", True)
    if eval_interval > 0:
        callbacks.append(
            EvaluationCallback(
                eval_steps=eval_interval,
                eval_before_train=eval_before_train,
            )
        )

    # HF model save callback
    hf_save_interval = getattr(cfg.trainer, "hf_save_interval", -1)
    if hf_save_interval > 0:
        callbacks.append(HFModelSaveCallback(save_steps=hf_save_interval))

    # Reference model update callback
    update_ref_every_epoch = getattr(cfg.trainer, "update_ref_every_epoch", False)
    if update_ref_every_epoch:
        callbacks.append(RefModelUpdateCallback(update_every_epoch=True))

    # Logging callback (always enabled)
    callbacks.append(LoggingCallback())

    return callbacks


class DefaultCallbackHandler(CallbackHandler):
    """
    A callback handler that initializes with default callbacks based on config.

    This provides backward compatibility by recreating the original inline
    behavior through callbacks.

    Example:
        ```python
        handler = DefaultCallbackHandler(cfg)
        # Adds all default callbacks based on config intervals
        ```
    """

    def __init__(
        self,
        cfg: Optional["DictConfig"] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ):
        """
        Initialize with default callbacks from config, plus any custom callbacks.

        Args:
            cfg: Training configuration. If provided, creates default callbacks.
            callbacks: Additional custom callbacks to add after defaults.
        """
        default_callbacks = []
        if cfg is not None:
            default_callbacks = create_default_callbacks(cfg)

        all_callbacks = default_callbacks + (callbacks or [])
        super().__init__(all_callbacks)

    @classmethod
    def from_config(
        cls,
        cfg: "DictConfig",
        additional_callbacks: Optional[List[TrainerCallback]] = None,
    ) -> "DefaultCallbackHandler":
        """
        Create a handler from config with optional additional callbacks.

        Args:
            cfg: Training configuration
            additional_callbacks: Custom callbacks to add after defaults

        Returns:
            Configured callback handler
        """
        return cls(cfg=cfg, callbacks=additional_callbacks)


def create_callback_from_config(callback_config: Dict[str, Any]) -> TrainerCallback:
    """
    Create a callback instance from a YAML config dictionary.

    Args:
        callback_config: Dictionary with 'type' key and callback-specific params.
            Example: {"type": "checkpoint", "save_steps": 10}

    Returns:
        Instantiated callback

    Raises:
        ValueError: If callback type is unknown or missing
    """
    if "type" not in callback_config:
        raise ValueError(f"Callback config missing 'type' key: {callback_config}")

    callback_type = callback_config["type"]
    if callback_type not in CALLBACK_REGISTRY:
        available = ", ".join(CALLBACK_REGISTRY.keys())
        raise ValueError(
            f"Unknown callback type '{callback_type}'. "
            f"Available types: {available}"
        )

    # Get the callback class and instantiate with remaining params
    callback_cls = CALLBACK_REGISTRY[callback_type]
    params = {k: v for k, v in callback_config.items() if k != "type"}

    try:
        return callback_cls(**params)
    except TypeError as e:
        raise ValueError(
            f"Invalid parameters for callback '{callback_type}': {e}"
        ) from e


def create_callbacks_from_config(cfg: "DictConfig") -> List[TrainerCallback]:
    """
    Create callbacks from explicit YAML configuration.

    This supports the new-style callback configuration:
    ```yaml
    trainer:
      callbacks:
        - type: checkpoint
          save_steps: 10
        - type: evaluation
          eval_steps: 20
          eval_before_train: false
    ```

    Args:
        cfg: Training configuration with optional 'callbacks' list

    Returns:
        List of instantiated callbacks (empty if no callbacks configured)
    """
    callbacks_config = getattr(cfg.trainer, "callbacks", None)
    if callbacks_config is None:
        return []

    callbacks = []
    for callback_config in callbacks_config:
        # Convert OmegaConf to dict if needed
        if hasattr(callback_config, "items"):
            config_dict = dict(callback_config)
        else:
            config_dict = callback_config

        try:
            callback = create_callback_from_config(config_dict)
            callbacks.append(callback)
            logger.debug(f"Created callback: {callback.__class__.__name__}")
        except ValueError as e:
            logger.error(f"Failed to create callback: {e}")
            raise

    return callbacks


def get_available_callback_types() -> List[str]:
    """Get list of available callback type names for YAML configs."""
    return list(CALLBACK_REGISTRY.keys())
