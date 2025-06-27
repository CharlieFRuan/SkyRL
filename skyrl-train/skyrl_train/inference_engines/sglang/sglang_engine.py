import os
import asyncio
import warnings
from typing import List, Any, Optional, Union, Dict
import torch
import ray
from uuid import uuid4
import threading
import multiprocessing as mp

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    assert_pkg_version,
    is_cuda,
    maybe_set_triton_cache_manager,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightUpdateRequest,
)


# Patch SGLang's _set_envs_and_config to avoid signal handler issues in Ray actors
# Based on VERL's solution: https://github.com/sgl-project/sglang/issues/6723
def _patched_set_envs_and_config(server_args):
    """Patched version of SGLang's _set_envs_and_config that removes signal handler registration."""
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(getattr(server_args, 'enable_nccl_nvls', False)))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    # Set prometheus env vars
    if getattr(server_args, 'enable_metrics', False):
        try:
            set_prometheus_multiproc_dir()
        except:
            pass  # Ignore if not available

    # Set ulimit
    try:
        set_ulimit()
    except:
        pass  # Ignore if not available

    # Fix triton bugs
    tp_size = getattr(server_args, 'tp_size', 1)
    dp_size = getattr(server_args, 'dp_size', 1)
    if tp_size * dp_size > 1:
        try:
            maybe_set_triton_cache_manager()
        except:
            pass  # Ignore if not available

    # Check flashinfer version
    attention_backend = getattr(server_args, 'attention_backend', None)
    if attention_backend == "flashinfer":
        try:
            assert_pkg_version(
                "flashinfer_python",
                "0.2.5",
                "Please uninstall the old version and reinstall the latest version by following the instructions at https://docs.flashinfer.ai/installation.html.",
            )
        except:
            pass  # Ignore if not available
    
    try:
        if is_cuda():
            assert_pkg_version(
                "sgl-kernel",
                "0.1.1",
                "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
            )
    except:
        pass  # Ignore if not available

    # Set mp start method
    try:
        mp.set_start_method("spawn", force=True)
    except:
        pass  # Ignore if already set

    # We do NOT register signal handlers here to avoid Ray actor issues
    # Original SGLang code had: signal.signal(signal.SIGCHLD, sigchld_handler)
    # But this fails in Ray actors since signal handlers only work in main thread


# Apply the patch
import sglang.srt.entrypoints.engine
sglang.srt.entrypoints.engine._set_envs_and_config = _patched_set_envs_and_config


def setup_envvars_for_sglang(kwargs, bundle_indices):
    """Setup environment variables for SGLang similar to vLLM setup."""
    noset_visible_devices = kwargs.pop("noset_visible_devices", False)
    
    if noset_visible_devices:
        # Set CUDA_VISIBLE_DEVICES to the ray assigned GPU when needed
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

    num_gpus = kwargs.pop("num_gpus", 1)
    if bundle_indices is not None:
        print(f"creating SGLang Engine with bundle_indices={bundle_indices}")


class SGLangInferenceEngine(InferenceEngineInterface):
    """SGLang inference engine that implements InferenceEngineInterface."""

    def __init__(self, *args, bundle_indices: Optional[List[int]] = None, **kwargs):
        setup_envvars_for_sglang(kwargs, bundle_indices)
        
        # Store common attributes
        self._tp_size = kwargs.get("tensor_parallel_size", 1)
        self.tokenizer = kwargs.pop("tokenizer", None)
        
        # Extract sampling params
        sampling_params_dict = kwargs.pop("sampling_params", None)
        self.sampling_params = sampling_params_dict or {}
        
        # Create SGLang ServerArgs from kwargs
        server_args_dict = self._convert_kwargs_to_server_args(args, kwargs)
        server_args = ServerArgs(**server_args_dict)
        
        # Create the SGLang engine (signal handler issue is now fixed by patching)
        print("Creating SGLang engine with patched signal handler...")
        self.engine = Engine(server_args=server_args)
        print("SGLang engine created successfully")

    def _convert_kwargs_to_server_args(self, args, kwargs) -> Dict[str, Any]:
        """Convert vLLM-style kwargs to SGLang ServerArgs format."""
        server_args = {}
        
        # Map common parameters
        if "model" in kwargs:
            server_args["model_path"] = kwargs["model"]
        elif len(args) > 0:
            server_args["model_path"] = args[0]
            
        if "tensor_parallel_size" in kwargs:
            server_args["tp_size"] = kwargs["tensor_parallel_size"]
        if "dtype" in kwargs:
            server_args["dtype"] = kwargs["dtype"]
        if "trust_remote_code" in kwargs:
            server_args["trust_remote_code"] = kwargs["trust_remote_code"]
        if "max_model_len" in kwargs:
            server_args["context_length"] = kwargs["max_model_len"]
        if "seed" in kwargs:
            server_args["random_seed"] = kwargs["seed"]
            
        # SGLang specific defaults
        server_args["log_level"] = "error"  # Keep quiet by default
        
        return server_args

    def tp_size(self):
        """Return the tensor parallel size."""
        return self._tp_size

    def _preprocess_prompts(self, input_batch: InferenceEngineInput):
        """Preprocess prompts for SGLang generation."""
        prompts = input_batch.get("prompts")
        prompt_token_ids = input_batch.get("prompt_token_ids")
        request_sampling_params = input_batch.get("sampling_params")

        if (prompts is None and prompt_token_ids is None) or (prompts is not None and prompt_token_ids is not None):
            raise ValueError("Either `prompts` or `prompt_token_ids` must be provided, but not both.")

        # Use request sampling params if provided, otherwise use defaults
        sampling_params = request_sampling_params if request_sampling_params is not None else self.sampling_params

        # Handle prompts vs token_ids
        if prompts is not None:
            # Convert chat format to text if needed
            if isinstance(prompts[0], list):  # List of conversation messages
                text_prompts = []
                for prompt in prompts:
                    if self.tokenizer:
                        text_prompt = self.tokenizer.apply_chat_template(
                            prompt,
                            add_generation_prompt=True,
                            tokenize=False
                        )
                        text_prompts.append(text_prompt)
                    else:
                        # Fallback: just concatenate messages
                        text_prompt = " ".join([msg.get("content", "") for msg in prompt])
                        text_prompts.append(text_prompt)
                return text_prompts, None, sampling_params
            else:
                return prompts, None, sampling_params
        else:
            return None, prompt_token_ids, sampling_params

    def _postprocess_outputs(self, outputs):
        """Process SGLang outputs to match expected format."""
        responses: List[str] = []
        stop_reasons: List[str] = []
        
        # Handle both single output and batch outputs
        if not isinstance(outputs, list):
            outputs = [outputs]
            
        for output in outputs:
            if isinstance(output, dict):
                # SGLang returns dict with 'text' field
                responses.append(output.get("text", ""))
                # Map SGLang finish reasons to our expected format
                finish_reason = output.get("finish_reason", "stop")
                stop_reasons.append(finish_reason)
            else:
                # Fallback for unexpected format
                responses.append(str(output))
                stop_reasons.append("stop")

        return InferenceEngineOutput(
            responses=responses,
            stop_reasons=stop_reasons,
        )

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        """Generate responses using SGLang engine."""
        text_prompts, token_ids_prompts, sampling_params = self._preprocess_prompts(input_batch)
        
        # Generate using SGLang
        if text_prompts is not None:
            outputs = await asyncio.to_thread(
                self.engine.generate,
                prompt=text_prompts,
                sampling_params=sampling_params
            )
        else:
            outputs = await asyncio.to_thread(
                self.engine.generate,
                input_ids=token_ids_prompts,
                sampling_params=sampling_params
            )

        return self._postprocess_outputs(outputs)

    async def wake_up(self, *args: Any, **kwargs: Any):
        """Wake up the engine."""
        # TODO(Charlie): Check resume_memory_occupation and release_memory_occupation; understand which one to use,
        # tokenizer manager or the engine.
        pass

    async def sleep(self, *args: Any, **kwargs: Any):
        """Put engine to sleep."""
        # SGLang doesn't have explicit sleep/wake functionality
        # This is a no-op for compatibility
        pass

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        """Initialize weight update communicator for SGLang."""
        return await asyncio.to_thread(
            self.engine.init_weights_update_group,
            master_address=master_addr,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend
        )

    async def update_named_weight(self, request: NamedWeightUpdateRequest):
        """Update named weights in SGLang engine."""
        return await asyncio.to_thread(
            self.engine.update_weights_from_distributed,
            name=request["name"],
            dtype=request["dtype"],
            shape=request["shape"]
        )

    async def teardown(self):
        """Shutdown the SGLang engine."""
        await asyncio.to_thread(self.engine.shutdown)

    async def reset_prefix_cache(self):
        """Reset prefix cache in SGLang engine."""
        await asyncio.to_thread(self.engine.flush_cache)


# Create Ray actor for SGLang engine
SGLangRayActor = ray.remote(SGLangInferenceEngine)
