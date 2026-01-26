#!/usr/bin/env python3
"""
Test script to demonstrate vLLM's different logging modes:
1. LoggingStatLogger - Prints throughput stats to stdout
2. PrometheusStatLogger - Exports metrics to /metrics endpoint
3. Using AsyncLLM directly with periodic do_log_stats calls

Run with: CUDA_VISIBLE_DEVICES=7 python test_vllm_logging.py
"""
import asyncio
import time
import os

# Set up environment for single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# ============================================================
# Option 1: Using LLM (Sync Engine) - logs are automatic
# ============================================================
def test_sync_llm():
    """
    The sync LLM class has disable_log_stats=True by default.
    Set disable_log_stats=False to enable throughput logging.
    """
    from vllm import LLM, SamplingParams
    
    print("\n" + "="*60)
    print("OPTION 1: Sync LLM with disable_log_stats=False")
    print("="*60)
    
    # Note: By default, LLM sets disable_log_stats=True
    # We explicitly enable it here
    llm = LLM(
        model="facebook/opt-125m",
        disable_log_stats=False,  # Enable stat logging
    )
    
    prompts = [
        "The capital of France is",
        "Hello, my name is",
        "The meaning of life is",
    ]
    sampling_params = SamplingParams(max_tokens=50)
    
    print("\nGenerating... (you should see throughput stats every 10s)")
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        print(f"Prompt: {output.prompt[:30]}...")
        print(f"Output: {output.outputs[0].text[:50]}...")
        print()
    
    del llm


# ============================================================
# Option 2: Using AsyncLLM directly - need to call do_log_stats()
# ============================================================
async def test_async_llm():
    """
    AsyncLLM needs periodic calls to do_log_stats() to trigger logging.
    The OpenAI server does this in a background task.
    """
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    
    print("\n" + "="*60)
    print("OPTION 2: AsyncLLM with manual do_log_stats() calls")
    print("="*60)
    
    engine_args = AsyncEngineArgs(
        model="facebook/opt-125m",
        disable_log_stats=False,  # Enable stat logging
    )
    
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    prompts = [
        "The capital of France is",
        "Hello, my name is",
        "The meaning of life is",
    ]
    sampling_params = SamplingParams(max_tokens=50)
    
    print("\nGenerating with async engine...")
    
    # Start a background task to periodically call do_log_stats
    async def log_stats_task():
        while True:
            await asyncio.sleep(5)  # Log every 5 seconds
            await engine.do_log_stats()
            print("[Manual log_stats call made]")
    
    log_task = asyncio.create_task(log_stats_task())
    
    try:
        for i, prompt in enumerate(prompts):
            request_id = f"request-{i}"
            results_generator = engine.generate(prompt, sampling_params, request_id)
            
            final_output = None
            async for output in results_generator:
                final_output = output
            
            if final_output:
                print(f"Prompt: {prompt[:30]}...")
                print(f"Output: {final_output.outputs[0].text[:50]}...")
                print()
        
        # Force a final log
        await engine.do_log_stats()
    finally:
        log_task.cancel()
    
    del engine


# ============================================================
# Option 3: Check available CLI options
# ============================================================
def show_cli_options():
    """Show available CLI options for logging."""
    print("\n" + "="*60)
    print("VLLM LOGGING OPTIONS SUMMARY")
    print("="*60)
    
    print("""
### CLI Arguments (vllm serve / OpenAI Server)

--disable-log-stats
    Disable logging statistics (throughput, queue size, etc.)
    Default: False (stats are logged)

--enable-log-requests  
    Log incoming requests with their parameters
    Default: False
    
--enable-log-outputs
    Log model outputs (generations). Requires --enable-log-requests
    Default: False

--aggregate-engine-logging
    Aggregate stats across multiple DP engines into single log line
    Default: False

### Environment Variables

VLLM_LOG_STATS_INTERVAL=10.0
    Interval in seconds between stat logs (default: 10 seconds)

### Python API

# Sync LLM (offline inference)
llm = LLM(model="...", disable_log_stats=False)

# Async LLM
engine = AsyncLLMEngine.from_engine_args(
    AsyncEngineArgs(model="...", disable_log_stats=False)
)

# IMPORTANT for AsyncLLM:
# - The OpenAI server has a background task that calls do_log_stats() periodically
# - When using AsyncLLM directly, YOU must call do_log_stats() to trigger logging
# - Without periodic do_log_stats() calls, LoggingStatLogger won't print anything

### What Logging Looks Like

# LoggingStatLogger output (printed to stdout):
# INFO: Avg prompt throughput: 123.4 tokens/s, Avg generation throughput: 456.7 tokens/s, 
#       Running: 3 reqs, Waiting: 0 reqs, GPU KV cache usage: 12.3%, Prefix cache hit rate: 45.6%

### Prometheus Metrics (via /metrics endpoint on OpenAI server)

When running `vllm serve`, metrics are available at:
    http://localhost:8000/metrics

Example metrics:
    vllm:num_requests_running
    vllm:num_requests_waiting
    vllm:kv_cache_usage_perc
    vllm:prompt_tokens
    vllm:generation_tokens
    vllm:time_to_first_token_seconds
    vllm:e2e_request_latency_seconds
""")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import sys
    
    show_cli_options()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("\n" + "="*60)
        print("Running actual tests (requires GPU)...")
        print("="*60)
        
        test_sync_llm()
        asyncio.run(test_async_llm())


