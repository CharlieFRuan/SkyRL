"""Evaluate on Harbor tasks through skycap, without training.

The inference engines and the skycap servers start as for training (``main_harbor_skycap``), but no trainer:
every task of ``data.val_data`` runs ``generator.eval_n_samples_per_prompt`` trials once, at most
``generator.rate_limit.max_concurrency`` at a time. Harbor writes each trial's directory under
``harbor_trial_config.trials_dir`` as ``harbor run`` would (score those the way a stock Harbor job is scored), and
skycap each trajectory's record under ``skycap.record_dir``. A line per trial with what skycap made of it goes to
``{trainer.export_path}/eval/results.jsonl``.

``SKYCAP_TRIAL_ATTEMPTS=1`` makes one attempt per trial, as ``harbor run`` does.

    uv run --isolated --extra fsdp --extra harbor --extra skycap \\
        -m examples.train_integrations.harbor_skycap.entrypoints.main_harbor_skycap_eval \\
        data.val_data="['/path/to/harbor/tasks']" trainer.policy.model.path=... ...
"""

import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

import ray
import yaml
from loguru import logger

from skyrl.train.generators.base import BatchMetadata, GeneratorInput, TrajectoryID
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from ...harbor.dataset import HarborTaskDataset
from ...harbor.entrypoints.main_harbor import HARBOR_DEFAULT_CONFIG, _deep_merge
from .main_harbor_skycap import HarborSkycapConfig, HarborSkycapExp, _exposure


class HarborSkycapEvalExp(HarborSkycapExp):
    def get_train_dataset(self):
        return None

    def get_eval_dataset(self):
        return HarborTaskDataset(data_files=list(self.cfg.data.val_data))

    def run(self):
        try:
            client = self.get_inference_client()
            if self.cfg.trainer.placement.colocate_all:
                # Colocated engines start asleep, for a trainer that isn't there.
                asyncio.run(client.wake_up())
            generator = self.get_generator(self.cfg, self.tokenizer, client)
            asyncio.run(self._evaluate(generator))
        finally:
            if self.skycap is not None:
                logger.info("stopping skycap, writing the trajectories still in memory")
                self.skycap.stop()

    async def _evaluate(self, generator) -> None:
        try:
            tasks = list(self.eval_dataset)
            if not tasks:
                raise ValueError(f"no Harbor tasks in data.val_data={list(self.cfg.data.val_data)}")
            samples = self.cfg.generator.eval_n_samples_per_prompt
            prompts, ids = [], []
            for task in tasks:
                for repetition in range(samples):
                    prompts.append(task["prompt"])
                    ids.append(TrajectoryID(instance_id=Path(task["prompt"]).name, repetition_id=repetition))
            logger.info(f"evaluating {len(tasks)} tasks x {samples} samples")
            batch = GeneratorInput(
                prompts=prompts,
                env_classes=None,
                env_extras=None,
                sampling_params=None,
                trajectory_ids=ids,
                batch_metadata=BatchMetadata(global_step=0, training_phase="eval"),
            )
            outcomes, _ = await generator.run_trials(batch)
        finally:
            await generator.close()
            self.generator = None
        out_dir = os.path.join(self.cfg.trainer.export_path, "eval")
        os.makedirs(out_dir, exist_ok=True)
        rows = [
            {
                "task": Path(prompt).name,
                "repetition": outcome.trajectory_id.repetition_id,
                "stop_reason": outcome.stop_reason,
                "train_reward": outcome.reward,
                "num_paths": len(outcome.samples),
                "unbridged_calls": outcome.unbridged_calls,
                "e2e_time": outcome.e2e_time,
            }
            for prompt, outcome in zip(prompts, outcomes)
        ]
        with open(os.path.join(out_dir, "results.jsonl"), "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        summary = {
            "trials": len(rows),
            "stop_reasons": dict(Counter(row["stop_reason"] for row in rows)),
            "mean_train_reward": sum(row["train_reward"] for row in rows) / max(1, len(rows)),
            "unbridged_calls": sum(row["unbridged_calls"] for row in rows),
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"eval done ({out_dir}): {json.dumps(summary)}")


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    HarborSkycapEvalExp(cfg).run()


def main() -> None:
    cfg = HarborSkycapConfig.from_cli_overrides(sys.argv[1:])
    with open(HARBOR_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)
    validate_cfg(cfg)
    _exposure(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
