"""
A thin wrapper that defers the import of `SGLangInferenceEngine` to `__init__()`.

Otherwise, importing `SGLangInferenceEngine` in `create_ray_wrapped_inference_engines()` means that
we are importing SGLang in the CPU-only driver process.

This is to avoid sglang from importing vllm when GPU not detected:
https://github.com/sgl-project/sglang/blob/c797322280b48b9cf88bd18992f3f628f85d86b5/python/sglang/srt/layers/quantization/utils.py#L11-L17
Similar comment: https://github.com/volcengine/verl/blob/9cc307767b0c787e8f5ef581dac929f7bde044ef/verl/workers/fsdp_workers.py#L520-L527

By importing sglang in the actor (which has GPU resources), we avoid the above issue.
"""

import ray
from skyrl_train.inference_engines.base import InferenceEngineInterface

class _LazySGLangInferenceEngine(InferenceEngineInterface):
    """An engine wrapper that imports SGLangInferenceEngine only after the actor starts."""

    def __init__(self, *args, **kwargs):
        # The real engine import happens here, inside the actor.
        from skyrl_train.inference_engines.sglang.sglang_engine import (
            SGLangInferenceEngine,
        )

        self._engine = SGLangInferenceEngine(*args, **kwargs)

    # ------------------------------------------------------------------
    # Simple pass-throughs to the underlying engine implementation.
    # ------------------------------------------------------------------
    def tp_size(self):
        return self._engine.tp_size()

    async def generate(self, *args, **kwargs):
        return await self._engine.generate(*args, **kwargs)

    async def wake_up(self, *args, **kwargs):
        return await self._engine.wake_up(*args, **kwargs)

    async def sleep(self, *args, **kwargs):
        return await self._engine.sleep(*args, **kwargs)

    async def init_weight_update_communicator(self, *args, **kwargs):
        return await self._engine.init_weight_update_communicator(*args, **kwargs)

    async def update_named_weight(self, *args, **kwargs):
        return await self._engine.update_named_weight(*args, **kwargs)

    async def teardown(self):
        return await self._engine.teardown()

    async def reset_prefix_cache(self):
        return await self._engine.reset_prefix_cache()

# Create a Ray actor class from the lazy wrapper.
SGLangRayActor = ray.remote(_LazySGLangInferenceEngine)
