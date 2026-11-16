"""Fully-async training on Harbor tasks with skycap: ``main_harbor_skycap`` with ``FullyAsyncRayPPOTrainer``.

The skycap counterpart of ``harbor/entrypoints/main_harbor_fully_async.py``. The skycap records are written as usual;
their per-step W&B index (``skycap.wandb``) is turned off, as the fully-async trainer's step loop doesn't run the
callback that writes it.

    uv run --isolated --extra megatron --extra harbor --extra skycap \\
        -m examples.train_integrations.harbor_skycap.entrypoints.main_harbor_skycap_fully_async \\
        trainer.fully_async.enabled=true trainer.placement.colocate_all=false ...
"""

import sys

import ray
import yaml
from loguru import logger

from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from ...harbor.entrypoints.main_harbor import HARBOR_DEFAULT_CONFIG, _deep_merge
from .main_harbor_skycap import HarborSkycapConfig, HarborSkycapExp, _exposure


class HarborSkycapFullyAsyncExp(HarborSkycapExp):
    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        return FullyAsyncRayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    HarborSkycapFullyAsyncExp(cfg).run()


def main() -> None:
    cfg = HarborSkycapConfig.from_cli_overrides(sys.argv[1:])
    with open(HARBOR_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)
    validate_cfg(cfg)
    _exposure(cfg)
    if cfg.skycap.wandb.enabled:
        logger.warning("skycap.wandb is not supported with the fully-async trainer; turning it off")
        cfg.skycap.wandb.enabled = False
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError("trainer.algorithm.max_seq_len must be set for Harbor training")
    if not cfg.trainer.fully_async.enabled:
        raise ValueError("set trainer.fully_async.enabled=true (or use main_harbor_skycap for synchronous training)")
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
