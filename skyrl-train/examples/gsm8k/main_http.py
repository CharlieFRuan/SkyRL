"""
Custom entrypoint for running SkyRL with HTTP endpoint generator.

This uses SkyRLGymHTTPGenerator instead of SkyRLGymGenerator to send
requests via HTTP endpoint instead of direct engine calls.
"""

import os
import sys
import hydra
import ray
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import (
    BasePPOExp,
    config_dir,
    initialize_ray,
    validate_cfg,
)


class MainHTTP(BasePPOExp):
    """Custom entrypoint that uses SkyRLGymHTTPGenerator."""

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """Initializes the HTTP-based generator.

        Returns:
            SkyRLGymHTTPGenerator: The generator that uses HTTP endpoint.
        """
        # Import from the examples directory
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from skyrl_gym_http_generator import SkyRLGymHTTPGenerator

        return SkyRLGymHTTPGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            model_name=cfg.trainer.policy.model.path,
        )


@ray.remote(num_cpus=1)
def skyrl_http_entrypoint(cfg: DictConfig):
    """Ray remote function to run the HTTP-based experiment."""
    exp = MainHTTP(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entrypoint for HTTP-based generator."""
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_http_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
