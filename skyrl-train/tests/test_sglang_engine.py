#!/usr/bin/env python3
"""
Simple test script to verify SGLang engine implementation.

Usage:
  uv run --isolated --extra sglang python tests/test_sglang_engine.py
"""

import asyncio
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine
from skyrl_train.inference_engines.base import InferenceEngineInput


async def test_sglang_engine():
    """Test basic SGLang engine functionality."""
    print("Testing SGLang engine implementation...")
    
    try:
        # Test engine creation
        print("1. Creating SGLang engine...")
        engine = SGLangInferenceEngine(
            model="facebook/opt-125m",  # Small model for testing
            tensor_parallel_size=1,
            dtype="float16",
            trust_remote_code=True,
            seed=42,
            sampling_params={"temperature": 1.0, "max_new_tokens": 50}
        )
        print("✓ SGLang engine created successfully")
        
        # Test tp_size method
        print("2. Testing tp_size method...")
        tp_size = engine.tp_size()
        assert tp_size == 1, f"Expected tp_size=1, got {tp_size}"
        print(f"✓ tp_size method works: {tp_size}")
        
        # Test generate method with text prompts
        print("3. Testing generate method with text prompts...")
        input_batch: InferenceEngineInput = {
            "prompts": [
                [{"role": "user", "content": "Hello, how are you?"}],
                [{"role": "user", "content": "What is 2+2?"}]
            ],
            "prompt_token_ids": None,
            "sampling_params": {"temperature": 0.1, "max_new_tokens": 20},
            "trajectory_ids": None
        }
        
        output = await engine.generate(input_batch)
        assert "responses" in output, "Output should contain 'responses' key"
        assert "stop_reasons" in output, "Output should contain 'stop_reasons' key"
        assert len(output["responses"]) == 2, f"Expected 2 responses, got {len(output['responses'])}"
        assert len(output["stop_reasons"]) == 2, f"Expected 2 stop_reasons, got {len(output['stop_reasons'])}"
        print(f"✓ Generate method works with text prompts")
        print(f"  Response 1: {output['responses'][0][:50]}...")
        print(f"  Response 2: {output['responses'][1][:50]}...")
        
        # Test wake_up and sleep methods (should be no-ops)
        print("4. Testing wake_up and sleep methods...")
        await engine.wake_up()
        await engine.sleep()
        print("✓ wake_up and sleep methods work (no-ops)")
        
        # Test reset_prefix_cache
        print("5. Testing reset_prefix_cache...")
        await engine.reset_prefix_cache()
        print("✓ reset_prefix_cache method works")
        
        # Clean up
        print("6. Cleaning up...")
        await engine.teardown()
        print("✓ Engine teardown successful")
        
        print("\n🎉 All SGLang engine tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_ray_actor():
    """Test SGLang Ray actor creation."""
    print("\nTesting SGLang Ray actor...")
    
    try:
        import ray
        from skyrl_train.inference_engines.sglang.sglang_engine import SGLangRayActor
        
        # Initialize Ray (if not already initialized)
        if not ray.is_initialized():
            ray.init(local_mode=True)
        
        print("1. Creating SGLang Ray actor...")
        actor = SGLangRayActor.remote(
            model="facebook/opt-125m",
            tensor_parallel_size=1,
            dtype="float16",
            trust_remote_code=True,
            seed=42
        )
        print("✓ SGLang Ray actor created successfully")
        
        # Test remote method call
        print("2. Testing remote tp_size call...")
        tp_size = await actor.tp_size.remote()
        assert tp_size == 1, f"Expected tp_size=1, got {tp_size}"
        print(f"✓ Remote tp_size call works: {tp_size}")
        
        # Test remote generate call
        print("3. Testing remote generate call...")
        input_batch: InferenceEngineInput = {
            "prompts": [[{"role": "user", "content": "Hello world!"}]],
            "prompt_token_ids": None,
            "sampling_params": {"temperature": 0.1, "max_new_tokens": 10},
            "trajectory_ids": None
        }
        
        output = await actor.generate.remote(input_batch)
        assert "responses" in output, "Output should contain 'responses' key"
        assert len(output["responses"]) == 1, f"Expected 1 response, got {len(output['responses'])}"
        print(f"✓ Remote generate call works")
        print(f"  Response: {output['responses'][0][:50]}...")
        
        # Clean up
        print("4. Cleaning up Ray actor...")
        await actor.teardown.remote()
        ray.shutdown()
        print("✓ Ray actor cleanup successful")
        
        print("\n🎉 All SGLang Ray actor tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Ray actor test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    async def main():
        print("=" * 60)
        print("SGLang Engine Test Suite")
        print("=" * 60)
        
        # Test basic engine
        success1 = await test_sglang_engine()
        
        # Test Ray actor
        success2 = await test_ray_actor()
        
        if success1 and success2:
            print("\n✅ All tests passed! SGLang backend is working correctly.")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed. Please check the implementation.")
            sys.exit(1)
    
    asyncio.run(main()) 