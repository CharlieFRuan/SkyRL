"""
Main entrypoint for training on Harbor tasks.
"""

import asyncio
import signal

import hydra
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir
from skyrl_train.utils import validate_cfg
from skyrl_train.utils.utils import initialize_ray
from examples.harbor.harbor_generator import HarborGenerator
from examples.harbor.dataset import HarborTaskDataset


async def _run_with_graceful_shutdown(coro):
    """Run *coro* with SIGINT wired to cancel the current task.

    Unlike the default asyncio SIGINT handler (_on_sigint) which raises
    KeyboardInterrupt — potentially interrupting cleanup code mid-execution
    — this uses ``loop.add_signal_handler`` so the cancellation is delivered
    as a normal event-loop callback.  The result is a *single* CancelledError
    that propagates through the task hierarchy, giving every ``finally``
    block (sandbox deletion, etc.) time to complete before the process exits.

    See harbor#656 / SkyRL#1160.
    """
    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()

    def _cancel_main():
        if not main_task.done():
            main_task.cancel()

    loop.add_signal_handler(signal.SIGINT, _cancel_main)
    try:
        await coro
    finally:
        try:
            loop.remove_signal_handler(signal.SIGINT)
        except Exception:
            pass


class HarborExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """
        Initializes the HarborGenerator.
        """
        return HarborGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,  # Pass harbor config to the generator
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            HarborTaskDataset: The training dataset.
        """
        prompts_dataset = HarborTaskDataset(
            data_files=self.cfg.data.train_data,
        )
        # make sure the dataset is large enough to train on
        assert (
            len(prompts_dataset) >= self.cfg.trainer.train_batch_size
        ), f"dataset should be atleast as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        return prompts_dataset

    def get_eval_dataset(self):
        """Initializes the evaluation dataset.

        Returns:
            HarborTaskDataset: The evaluation dataset.
        """
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            prompts_dataset = HarborTaskDataset(
                data_files=self.cfg.data.val_data,
            )
            return prompts_dataset
        return None

    def run(self):
        # Override BasePPOExp.run() to handle signals for the in-process
        # approach.
        #
        # SIGTERM: Ignore it.  When Ctrl+C sends SIGINT to the process
        # group, `uv` also dies and sends SIGTERM to us.  BasePPOExp.run()
        # maps SIGTERM→KeyboardInterrupt (for @ray.remote workers), but
        # here that would disrupt cleanup.
        #
        # SIGINT: Handled inside the event loop via add_signal_handler
        # (see _run_with_graceful_shutdown).  We set SIG_IGN here so
        # asyncio.run() does NOT install its own _on_sigint handler
        # (which raises KeyboardInterrupt and can interrupt cleanup code).
        #
        # See harbor#656 / SkyRL#1160.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        trainer = self._setup_trainer()
        try:
            asyncio.run(_run_with_graceful_shutdown(trainer.train()))
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_cfg(cfg)
    initialize_ray(cfg)

    # Ray's C code masks SIGINT (SIG_IGN) during initialization.
    # Restore it so signals are delivered.  The actual handler for
    # asyncio.run() is installed in _run_with_graceful_shutdown().
    signal.signal(signal.SIGINT, signal.default_int_handler)

    exp = HarborExp(cfg)
    exp.run()


if __name__ == "__main__":
    main()
