BACKEND_CONFIG='{"trainer.placement.colocate_all": true, "trainer.placement.policy_num_gpus_per_node": 8, "trainer.micro_forward_batch_size_per_gpu": 64, "trainer.micro_train_batch_size_per_gpu": 64, "generator.inference_engine.num_engines": 2, "generator.inference_engine.tensor_parallel_size": 4, "generator.inference_engine.backend": "vllm", "generator.inference_engine.run_engines_locally": true, "generator.inference_engine.weight_sync_backend": "nccl", "generator.inference_engine.async_engine": true, "generator.inference_engine.gpu_memory_utilization": 0.8, "generator.batched": true}'

uv run --extra tinker --extra megatron -m skyrl.tinker.api \
    --base-model "Qwen/Qwen3-4B-Instruct-2507" \
    --backend megatron \
    --port 8000 \
    --backend-config "$BACKEND_CONFIG"
