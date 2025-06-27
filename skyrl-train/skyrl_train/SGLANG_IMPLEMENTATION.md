# SGLang Backend Implementation

This document summarizes the SGLang backend implementation for the SkyRL training framework.

## Overview

I have successfully implemented SGLang support as an alternative inference backend to vLLM. The implementation follows the same patterns and interfaces as the existing vLLM backend, ensuring seamless integration with the existing codebase.

## What Was Implemented

### 1. Core SGLang Engine (`skyrl-train/skyrl_train/inference_engines/sglang/sglang_engine.py`)

- **`SGLangInferenceEngine`** class that implements `InferenceEngineInterface`
- Uses `sglang.srt.entrypoints.engine.Engine` as the underlying inference engine
- Provides the same async interface as vLLM engines
- **`SGLangRayActor`** for Ray-based distributed inference
- Proper parameter mapping from vLLM-style kwargs to SGLang ServerArgs

### 2. Updated Ray Engine Factory (`skyrl-train/skyrl_train/inference_engines/ray_wrapped_inference_engine.py`)

- Modified `create_ray_wrapped_inference_engines` to support backend selection
- Added `backend` parameter to choose between "vllm" and "sglang"
- Conditional engine creation based on backend type
- Proper handling of SGLang-specific limitations (no async engine)

### 3. Updated Configuration Integration (`skyrl-train/skyrl_train/entrypoints/main_base.py`)

- Modified `create_ray_wrapped_inference_engines_from_config` to pass backend parameter
- Enables backend selection through `cfg.generator.backend` configuration

### 4. Example Usage Scripts

- **`examples/gsm8k/run_gsm8k_sglang.sh`**: Complete example showing how to use SGLang for GSM8K training
- **`tests/test_sglang_engine.py`**: Comprehensive test suite for SGLang engine functionality

### 5. Documentation

- **`docs/examples/sglang-backend.rst`**: Complete guide on using SGLang backend
- **`SGLANG_IMPLEMENTATION.md`**: This summary document

## Key Features

### ✅ Implemented
- [x] Basic SGLang engine with async interface
- [x] Ray actor support for distributed inference  
- [x] Weight update functionality using SGLang's distributed weight update API
- [x] Chat template support and prompt preprocessing
- [x] Sampling parameter mapping between vLLM and SGLang formats
- [x] Configuration-based backend selection
- [x] Comprehensive test suite
- [x] Documentation and examples

### ⚠️ Limitations
- **No Async Engine**: SGLang doesn't support async engines, so `async_engine` must be `false`
- **No Sleep/Wake**: SGLang doesn't have sleep/wake functionality (implemented as no-ops)
- **Different Sampling Params**: Some vLLM-specific parameters are not supported by SGLang

## Configuration Changes

To use SGLang backend, simply change the configuration:

```yaml
generator:
  backend: "sglang"        # Use SGLang instead of vLLM
  async_engine: false      # Required for SGLang
  # ... rest of configuration remains the same
```

## Usage Examples

### Command Line
```bash
# Use SGLang backend
uv run --isolated --extra sglang -m skyrl_train.entrypoints.main_base \
  generator.backend=sglang \
  generator.async_engine=false \
  # ... other parameters
```

### Configuration File
```yaml
generator:
  backend: "sglang"
  async_engine: false
  num_inference_engines: 4
  inference_engine_tensor_parallel_size: 1
  sampling_params:
    max_new_tokens: 1024  # Note: different from vLLM's max_generate_length
    temperature: 1.0
    top_p: 1.0
```

## Testing

Run the test suite to verify SGLang backend works:

```bash
uv run --isolated --extra sglang python tests/test_sglang_engine.py
```

## Architecture

The implementation follows the same architecture as vLLM:

```
InferenceEngineInterface (abstract)
├── VLLMInferenceEngine (vLLM backend)
├── AsyncVLLMInferenceEngine (vLLM async backend)  
└── SGLangInferenceEngine (SGLang backend) ← NEW

Ray Actors:
├── VLLMRayActor = ray.remote(VLLMInferenceEngine)
├── AsyncVLLMRayActor = ray.remote(AsyncVLLMInferenceEngine)
└── SGLangRayActor = ray.remote(SGLangInferenceEngine) ← NEW

Factory Function:
create_ray_wrapped_inference_engines(backend="vllm" | "sglang") ← UPDATED
```

## Implementation Details

### Parameter Mapping

The implementation maps vLLM-style parameters to SGLang ServerArgs:

| vLLM Parameter | SGLang Parameter |
|----------------|------------------|
| `model` | `model_path` |
| `tensor_parallel_size` | `tp_size` |
| `max_model_len` | `context_length` |
| `seed` | `random_seed` |

### Sampling Parameters

SGLang uses different sampling parameter names:

| vLLM | SGLang |
|------|--------|
| `max_generate_length` | `max_new_tokens` |
| `min_tokens` | Not supported |
| `include_stop_str_in_output` | Not supported |

### Weight Updates

SGLang weight updates use the engine's built-in distributed weight update functionality:
- `init_weights_update_group()` for initialization
- `update_weights_from_distributed()` for weight broadcasting

## Dependencies

SGLang is included as an optional dependency in `pyproject.toml`:

```toml
[project.optional-dependencies]
sglang = [
    "sglang[srt,openai]==0.4.6.post5",
    "torch-memory-saver>=0.0.5",  
    "flashinfer-python@...",
]
```

## Future Improvements

Potential areas for enhancement:

1. **Async Engine Support**: When SGLang adds async engine support, implement `AsyncSGLangInferenceEngine`
2. **Additional Sampling Parameters**: Map more vLLM sampling parameters as SGLang adds support
3. **Performance Optimizations**: Tune SGLang-specific performance settings
4. **Error Handling**: Add more robust error handling for SGLang-specific issues

## Conclusion

The SGLang backend implementation provides a complete alternative to vLLM while maintaining full compatibility with the existing SkyRL training framework. Users can now choose between vLLM and SGLang based on their specific performance requirements and preferences. 