#!/usr/bin/env python3

import asyncio
import torch
import torch.distributed as dist
import time
import ray
import os
from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine
from skyrl_train.inference_engines.base import NamedWeightUpdateRequest

def init_process_group():
    """Initialize a simple process group for testing"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    
    dist.init_process_group(
        backend="nccl",
        rank=0,
        world_size=2,
        init_method="env://"
    )

async def test_sglang_weight_update():
    """Test SGLang weight update in isolation"""
    
    print("Testing SGLang weight update...")
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Create SGLang engine
    engine = SGLangInferenceEngine(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=1536,
    )
    
    print("SGLang engine created")
    
    # Test init_weight_update_communicator
    print("Testing init_weight_update_communicator...")
    start_time = time.time()
    
    try:
        success, message = await engine.init_weight_update_communicator(
            master_addr="127.0.0.1",
            master_port=29501,
            rank_offset=1,
            world_size=2,
            group_name="test_group",
            backend="nccl"
        )
        print(f"init_weight_update_communicator: success={success}, message={message}")
        print(f"Time taken: {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"init_weight_update_communicator failed: {e}")
        print(f"Time taken: {time.time() - start_time:.2f}s")
        
    # Test update_named_weight  
    print("Testing update_named_weight...")
    start_time = time.time()
    
    try:
        request = NamedWeightUpdateRequest({
            "name": "test_param",
            "dtype": torch.bfloat16,
            "shape": [128, 64]
        })
        
        success, message = await engine.update_named_weight(request)
        print(f"update_named_weight: success={success}, message={message}")
        print(f"Time taken: {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"update_named_weight failed: {e}")
        print(f"Time taken: {time.time() - start_time:.2f}s")
    
    ray.shutdown()
    print("Test completed")

if __name__ == "__main__":
    asyncio.run(test_sglang_weight_update()) 