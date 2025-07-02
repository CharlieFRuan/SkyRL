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
# from sglang.srt.managers.io_struct import UpdateWeightsFromDistributedReqInput, InitWeightsUpdateGroupReqInput
from sglang.srt.managers.tokenizer_manager import (
    UpdateWeightsFromDistributedReqInput,
    InitWeightsUpdateGroupReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
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


# TODO(charlie): duplicate of setup_envvars_for_vllm
def setup_envvars_for_sglang(kwargs, bundle_indices):
    noset_visible_devices = kwargs.pop("noset_visible_devices")
    if kwargs.get("distributed_executor_backend") == "ray":
        # a hack to make the script work.
        # stop ray from manipulating *_VISIBLE_DEVICES
        # at the top-level when the distributed_executor_backend is ray.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
    elif noset_visible_devices:
        # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
        # when the distributed_executor_backend is not rayargs and
        # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])


class SGLangInferenceEngine(InferenceEngineInterface):
    """SGLang inference engine that implements InferenceEngineInterface."""

    def __init__(self, *args, bundle_indices: Optional[List[int]] = None, **kwargs):
        setup_envvars_for_sglang(kwargs, bundle_indices)
        
        # default to use dummy load format, which need to reload weights in first time
        self._need_reload = True

        # Store common attributes
        self._tp_size = kwargs.get("tensor_parallel_size", 1)
        self.tokenizer = kwargs.pop("tokenizer", None)
        
        # Extract sampling params
        sampling_params_dict = kwargs.pop("sampling_params", None)
        self.sampling_params = sampling_params_dict or {}
        
        # Create SGLang ServerArgs from kwargs
        server_args_dict = self._convert_kwargs_to_server_args(args, kwargs)
        # server_args = ServerArgs(**server_args_dict)
        
        # Create the SGLang engine (signal handler issue is now fixed by patching)
        print("Creating SGLang engine with patched signal handler...")
        # self.engine = Engine(server_args=server_args)
        self.engine = Engine(**server_args_dict)
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
        server_args["disable_cuda_graph"] = True  # Disable CUDA graph to avoid JIT compilation issues
        
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
        
        # Generate using SGLang's async method
        # Otherwise using `await asyncio.to_thread(self.engine.generate)` will cause issues
        # as SGLang's `generate()` method calls `loop = asyncio.get_event_loop()`, raising error
        # `RuntimeError: There is no current event loop in thread 'ThreadPoolExecutor-0_0'.`
        if text_prompts is not None:
            outputs = await self.engine.async_generate(
                prompt=text_prompts,
                sampling_params=sampling_params
            )
        else:
            outputs = await self.engine.async_generate(
                input_ids=token_ids_prompts,
                sampling_params=sampling_params
            )

        return self._postprocess_outputs(outputs)

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        """Initialize weight update communicator for SGLang."""
        obj = InitWeightsUpdateGroupReqInput(
            master_address=master_addr,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend
        )
        
        # NOTE(charlie): Call the async method on tokenizer_manager directly to avoid event loop
        # conflicts since `sgl.Engine.init_weights_update_group` is sync, yet uses
        # `asyncio.get_event_loop()` which prevents us from using `asyncio.to_thread`. We cannot
        # call the sync method directly either because it runs into `RuntimeError: this event loop
        # is already running.`
        success, message = await self.engine.tokenizer_manager.init_weights_update_group(obj, None)
        return success, message

    async def update_named_weight(self, request: NamedWeightUpdateRequest):
        """Update named weights in SGLang engine."""
        obj = UpdateWeightsFromDistributedReqInput(
            name=request["name"],
            dtype=request["dtype"],
            shape=request["shape"]
        )
        
        # Call the underlying async method for the same reason as in `init_weight_update_communicator`
        success, message = await self.engine.tokenizer_manager.update_weights_from_distributed(obj, None)
        return success, message

    async def wake_up(self, tags: Optional[List[str]] = None):
        """Wake up the engine. For multi-stage waking up, pass in `"weight"` or `"kv_cache"` to tags."""
        dummy_obj = ResumeMemoryOccupationReqInput(tags=["weight"])
        print("CHARLIE DUMMY OBJ: ", dummy_obj)
        # # because __init__ is a sync method, it can not call the async release_memory_occupation
        # # have to move release_memory_occupation from __init__ to here
        # # For multi-stage awake, we run release weight and kv_cache when we resume weights for the first time.
        # if self._need_reload:
        #     await self.sleep()
        #     self._need_reload = False

        if tags is None:
            obj = ResumeMemoryOccupationReqInput()
        else:
            obj = ResumeMemoryOccupationReqInput(tags=tags)
        print("CHARLIE SGLANG ENGINE WAKE UP WITH TAGS: ", tags)
        # Call the underlying async method for the same reason as in `init_weight_update_communicator`
        return await self.engine.tokenizer_manager.resume_memory_occupation(obj, None)

    async def sleep(self, tags: Optional[List[str]] = None):
        """Put engine to sleep."""
        dummy_obj = ReleaseMemoryOccupationReqInput(tags=["weight"])
        print("CHARLIE DUMMY OBJ: ", dummy_obj)
        if tags is None:
            obj = ReleaseMemoryOccupationReqInput()
        else:
            obj = ReleaseMemoryOccupationReqInput(tags=tags)
        # Call the underlying async method for the same reason as in `init_weight_update_communicator`
        print("CHARLIE SGLANG ENGINE SLEEP WITH TAGS: ", tags)
        return await self.engine.tokenizer_manager.release_memory_occupation(obj, None)


    async def teardown(self):
        """Shutdown the SGLang engine."""
        self.engine.shutdown()

    async def reset_prefix_cache(self):
        """Reset prefix cache in SGLang engine."""
        # Call the underlying async method for the same reason as in `init_weight_update_communicator`
        return await self.engine.tokenizer_manager.flush_cache()


# Create Ray actor for SGLang engine. Note there is no Sync/Async distinction for SGLang.
SGLangRayActor = ray.remote(SGLangInferenceEngine)
