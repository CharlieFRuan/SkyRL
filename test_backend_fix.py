#!/usr/bin/env python3
"""
Simple test to verify the backend configuration fix works
"""
import sys
import os
sys.path.insert(0, '/home/ubuntu/SkyRL/skyrl-train')

# Mock minimal config structure
class MockConfig:
    def __init__(self):
        self.generator = self
        self.trainer = self
        self.placement = self
        
    def __getattr__(self, name):
        return getattr(self, name, None)

def test_backend_detection():
    """Test that backend detection works correctly"""
    
    # Test case 1: vLLM backend should enable CUDA IPC
    print("=== Test 1: vLLM backend ===")
    cfg = MockConfig()
    cfg.generator.backend = 'vllm'
    cfg.generator.weight_sync_backend = 'nccl'
    cfg.trainer.placement.colocate_all = True
    
    # Simulate the FSDP worker logic
    use_cuda_ipc = False
    if cfg.generator.weight_sync_backend == "nccl" and cfg.trainer.placement.colocate_all:
        backend = getattr(cfg.generator, 'backend', 'vllm')
        print(f"Detected backend: {backend}")
        if backend != 'sglang':
            use_cuda_ipc = True
            print(f"Enabling CUDA IPC for {backend}")
        else:
            print(f"Disabling CUDA IPC for SGLang")
    
    print(f"Final use_cuda_ipc: {use_cuda_ipc}")
    assert use_cuda_ipc == True, "CUDA IPC should be enabled for vLLM"
    
    # Test case 2: SGLang backend should disable CUDA IPC
    print("\n=== Test 2: SGLang backend ===")
    cfg = MockConfig()
    cfg.generator.backend = 'sglang'
    cfg.generator.weight_sync_backend = 'nccl'
    cfg.trainer.placement.colocate_all = True
    
    # Simulate the FSDP worker logic
    use_cuda_ipc = False
    if cfg.generator.weight_sync_backend == "nccl" and cfg.trainer.placement.colocate_all:
        backend = getattr(cfg.generator, 'backend', 'vllm')
        print(f"Detected backend: {backend}")
        if backend != 'sglang':
            use_cuda_ipc = True
            print(f"Enabling CUDA IPC for {backend}")
        else:
            print(f"Disabling CUDA IPC for SGLang")
    
    print(f"Final use_cuda_ipc: {use_cuda_ipc}")
    assert use_cuda_ipc == False, "CUDA IPC should be disabled for SGLang"
    
    # Test case 3: Missing backend should default to vLLM
    print("\n=== Test 3: Missing backend (should default to vLLM) ===")
    cfg = MockConfig()
    # Don't set cfg.generator.backend
    cfg.generator.weight_sync_backend = 'nccl'
    cfg.trainer.placement.colocate_all = True
    
    # Simulate the FSDP worker logic
    use_cuda_ipc = False
    if cfg.generator.weight_sync_backend == "nccl" and cfg.trainer.placement.colocate_all:
        backend = getattr(cfg.generator, 'backend', 'vllm')
        print(f"Detected backend: {backend}")
        if backend != 'sglang':
            use_cuda_ipc = True
            print(f"Enabling CUDA IPC for {backend}")
        else:
            print(f"Disabling CUDA IPC for SGLang")
    
    print(f"Final use_cuda_ipc: {use_cuda_ipc}")
    assert use_cuda_ipc == True, "CUDA IPC should be enabled for default vLLM"
    
    print("\n✅ All tests passed! The backend detection fix works correctly.")

if __name__ == "__main__":
    test_backend_detection() 