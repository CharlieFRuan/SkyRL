import os
import ray
import torch
import asyncio
import vllm
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from ray.util.placement_group import placement_group, PlacementGroupSchedulingStrategy
from typing import List, Any, Optional


def initialize_ray():
    """Initialize Ray with proper environment variables for VLLM"""
    env_vars = {
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_P2P_DISABLE": "0",
        "CUDA_LAUNCH_BLOCKING": "1",
        "VLLM_USE_V1": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    }
    ray.init(runtime_env={"env_vars": env_vars})


def setup_envvars_for_vllm(kwargs, bundle_indices):
    """Setup environment variables for VLLM with Ray backend"""
    noset_visible_devices = kwargs.pop("noset_visible_devices", False)
    if kwargs.get("distributed_executor_backend") == "ray":
        # Stop ray from manipulating *_VISIBLE_DEVICES
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
    elif noset_visible_devices:
        # Set CUDA_VISIBLE_DEVICES to the ray assigned GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

    num_gpus = kwargs.pop("num_gpus")
    if bundle_indices is not None:
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
        print(f"creating LLM with bundle_indices={bundle_indices}")


class VLLMInferenceEngine:
    """Simple VLLM inference engine"""
    
    def __init__(self, model_path: str, tensor_parallel_size: int = 1, bundle_indices: Optional[List[int]] = None, **kwargs):
        setup_envvars_for_vllm(kwargs, bundle_indices)
        
        # Disable multiprocessing for VLLM v1
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        
        # Set distributed executor backend
        distributed_executor_backend = "ray" if tensor_parallel_size > 1 else "uni"
        
        self.llm = vllm.LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=distributed_executor_backend,
            trust_remote_code=True,
            **{k: v for k, v in kwargs.items() if k != 'distributed_executor_backend'}
        )
        
        self.default_sampling_params = SamplingParams(temperature=0.0, max_tokens=30)
    
    async def generate(self, prompts: List[str], sampling_params: SamplingParams = None) -> List[str]:
        """Generate text for given prompts"""
        if sampling_params is None:
            sampling_params = self.default_sampling_params
            
        outputs = await asyncio.to_thread(
            self.llm.generate,
            prompts=prompts,
            sampling_params=sampling_params,
        )
        
        responses = []
        for output in outputs:
            # Each prompt should have only one response
            resp = output.outputs[0]
            responses.append(resp.text)
            
        return responses


def check_gpu_availability():
    """Check how many GPUs are available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"Found {gpu_count} GPUs available")
            return gpu_count
        else:
            print("CUDA is not available")
            return 0
    except Exception as e:
        print(f"Error checking GPU availability: {e}")
        return 0


def get_ray_pg_ready_with_timeout(pg, timeout: int = 60):
    """Wait for placement group to be ready with timeout"""
    try:
        ray.get(pg.ready(), timeout=timeout)
    except Exception as e:
        # Extract resource demands from the placement group
        bundles = pg.bundle_specs
        total_gpus = sum(bundle.get("GPU", 0) for bundle in bundles)
        total_cpus = sum(bundle.get("CPU", 0) for bundle in bundles)

        raise RuntimeError(
            f"Failed to create placement group with {len(bundles)} bundles "
            f"(requiring {total_gpus} GPUs, {total_cpus} CPUs total) in {timeout} seconds. "
            f"This might indicate insufficient GPU resources.\n"
            f"Error: {e}"
        )


# Create Ray actor for VLLM
VLLMRayActor = ray.remote(VLLMInferenceEngine)


async def main():
    # Initialize Ray
    initialize_ray()
    
    # Sample prompts
    prompts = [
        "Hello, my name is",
        "The president of the United States is", 
        "The capital of France is",
        "The future of AI is",
    ]
    
    # Check GPU availability
    available_gpus = check_gpu_availability()
    
    # Configuration - Choose TP size based on available GPUs
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"  
    
    # Try TP=4 if we have enough GPUs, otherwise fall back to TP=1
    if available_gpus >= 4:
        # Use a model that works with TP=4 (needs attention heads divisible by 4)
        model_path = "Qwen/Qwen2.5-1.5B-Instruct"  # 16 attention heads, divisible by 4
        tensor_parallel_size = 4
        print(f"Using TP=4 with {available_gpus} available GPUs")
    else:
        tensor_parallel_size = 1
        print(f"Only {available_gpus} GPUs available, using TP=1")
    
    print(f"Testing with model: {model_path}, TP size: {tensor_parallel_size}")
    
    try:
        # Create placement group
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(tensor_parallel_size)]
        pg = placement_group(bundles, strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=30)  # Reduced timeout
        
        print(f"Placement group created successfully with {tensor_parallel_size} GPUs")
        
        # Create bundle indices for the GPUs
        bundle_indices = list(range(tensor_parallel_size)) if tensor_parallel_size > 1 else None
        
        # Create scheduling strategy
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        
        print("Creating VLLM Ray actor...")
        
        # Create VLLM Ray actor with timeout protection
        engine = VLLMRayActor.options(
            num_cpus=1,
            num_gpus=1,  # This is per-actor, not total
            scheduling_strategy=scheduling_strategy,
        ).remote(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            bundle_indices=bundle_indices,
            num_gpus=1,  # This gets passed to setup_envvars_for_vllm
            distributed_executor_backend="ray" if tensor_parallel_size > 1 else "uni",
            dtype="bfloat16",
            gpu_memory_utilization=0.8,  # Reduced memory utilization
            max_model_len=1024,  # Reduced max length
            enforce_eager=True,  # Disable CUDA graphs for faster startup
        )
        
        print("Waiting for VLLM engine to initialize (timeout: 120 seconds)...")
        
        # Test if engine is ready with timeout
        try:
            ready_check = await asyncio.wait_for(
                engine.__ray_ready__.remote(),
                timeout=120.0  # 2 minute timeout for initialization
            )
            ray.get(ready_check)
            print("VLLM engine created successfully")
        except asyncio.TimeoutError:
            print("ERROR: VLLM engine initialization timed out after 120 seconds")
            ray.shutdown()
            return
        
        # Create sampling parameters
        sampling_params = SamplingParams(temperature=0.0, max_tokens=20)  # Reduced max tokens
        
        # Generate responses with timeout
        print("Generating responses...")
        try:
            responses_future = engine.generate.remote(prompts, sampling_params)
            responses = await asyncio.wait_for(
                ray.get(responses_future),
                timeout=60.0  # 1 minute timeout for generation
            )
            
            # Print results
            for prompt, response in zip(prompts, responses):
                print("=" * 50)
                print(f"Prompt: {prompt}")
                print(f"Generated text: {response}")
                
        except asyncio.TimeoutError:
            print("ERROR: Text generation timed out after 60 seconds")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("Shutting down Ray...")
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
