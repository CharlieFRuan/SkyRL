"""Evaluate on Harbor tasks through skycap, without training.

The inference engines and the skycap servers start as for training (``main_harbor_skycap``), but no
trainer: every task of ``data.val_data`` runs ``generator.eval_n_samples_per_prompt`` trials once, at
most ``generator.rate_limit.max_concurrency`` at a time. Harbor writes each trial's directory under
``harbor_trial_config.trials_dir`` as ``harbor run`` would, and skycap each trajectory's record under
``skycap.record_dir``.

Each trial is scored as stock Harbor scores it: the verifier's reward, which Harbor computes after an
agent timeout or a failed agent too, and 0 when the verifier didn't run. A trial that failed before
its agent ran (a sandbox that didn't start) is retried once, as in training. The scores go to
``{trainer.export_path}/eval/results.jsonl`` (one line per trial) and ``summary.json``.

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
from typing import Any, Dict, List

import ray
import yaml
from loguru import logger

from skyrl.train.generators.base import BatchMetadata, GeneratorInput, TrajectoryID
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from ...harbor.dataset import HarborTaskDataset
from ...harbor.entrypoints.main_harbor import HARBOR_DEFAULT_CONFIG, _deep_merge
from ..compose import TrialOutcome
from .main_harbor_skycap import (
    HarborSkycapConfig,
    HarborSkycapExp,
    _check_agent_reach,
    _exposure,
)


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
        rows = [result_row(prompt, outcome) for prompt, outcome in zip(prompts, outcomes)]
        out_dir = os.path.join(self.cfg.trainer.export_path, "eval")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.jsonl"), "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        summary = summarize(rows)
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"eval summary ({out_dir}): {json.dumps(summary)}")


def result_row(prompt: str, outcome: TrialOutcome) -> Dict[str, Any]:
    """One trial, scored as stock Harbor scores it (``reward``), with what skycap made of it."""
    return {
        "task": Path(prompt).name,
        "task_path": prompt,
        "repetition": outcome.trajectory_id.repetition_id,
        "trial_name": outcome.trial_name,
        "reward": outcome.verifier_reward if outcome.verifier_reward is not None else 0.0,
        "verified": outcome.verifier_reward is not None,
        "exception": outcome.exception,
        "stop_reason": outcome.stop_reason,
        "train_reward": outcome.reward,
        "skycap_id": outcome.skycap_id,
        "num_paths": len(outcome.samples),
        "unbridged_calls": outcome.unbridged_calls,
        "e2e_time": outcome.e2e_time,
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    return {
        "num_trials": n,
        "mean_reward": sum(row["reward"] for row in rows) / n if n else 0.0,
        "num_verified": sum(row["verified"] for row in rows),
        "exceptions": dict(Counter(row["exception"] for row in rows if row["exception"])),
        "stop_reasons": dict(Counter(row["stop_reason"] for row in rows)),
        "num_unbridged_trials": sum(row["unbridged_calls"] > 0 for row in rows),
    }


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
    _check_agent_reach(cfg)
    if not cfg.data.val_data:
        raise ValueError("data.val_data must name the Harbor task directories to evaluate")
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError("trainer.algorithm.max_seq_len must be set")
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
