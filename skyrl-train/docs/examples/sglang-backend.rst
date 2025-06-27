Using SGLang Backend
==================

This guide explains how to use SGLang as an inference backend instead of vLLM for your reinforcement learning training.

Overview
--------

SGLang is an alternative inference engine that can be used as a backend for generating responses during RL training. It provides similar functionality to vLLM but with different performance characteristics and features.

Key differences from vLLM:
- No asynchronous engine support currently
- Different API for weight updates
- No explicit sleep/wake functionality
- Different sampling parameter names

Installation
------------

To use SGLang backend, you need to install the SGLang dependencies:

.. code-block:: bash

    uv run --isolated --extra sglang your_training_script

Configuration
-------------

To use SGLang backend, set the ``generator.backend`` configuration to ``"sglang"``:

.. code-block:: yaml

    generator:
        backend: "sglang"  # Use SGLang instead of vLLM
        async_engine: false  # SGLang doesn't support async engines yet
        # ... other configuration options

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

The following configuration options are available when using SGLang backend:

- ``generator.backend``: Set to ``"sglang"`` to use SGLang backend
- ``generator.async_engine``: Must be ``false`` for SGLang (async not supported)
- ``generator.tensor_parallel_size``: Tensor parallelism size (same as vLLM)
- ``generator.model_dtype``: Model data type (same as vLLM)
- ``generator.sampling_params``: Sampling parameters (see below for differences)

Sampling Parameters
~~~~~~~~~~~~~~~~~~~

SGLang uses slightly different sampling parameter names compared to vLLM:

.. code-block:: yaml

    generator:
        sampling_params:
            max_new_tokens: 1024    # Instead of max_generate_length
            temperature: 1.0
            top_p: 1.0
            top_k: -1
            # min_tokens and include_stop_str_in_output are not supported in SGLang

Example Usage
-------------

Complete GSM8K Training Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a complete example of running GSM8K training with SGLang backend:

.. code-block:: bash

    #!/bin/bash
    # run_gsm8k_sglang.sh
    
    uv run --isolated --extra sglang -m skyrl_train.entrypoints.main_base \\
      trainer.algorithm.advantage_estimator="grpo" \\
      data.train_data="['$HOME/data/gsm8k/train.parquet']" \\
      data.val_data="['$HOME/data/gsm8k/validation.parquet']" \\
      trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \\
      trainer.placement.colocate_all=true \\
      trainer.strategy=fsdp \\
      generator.backend=sglang \\
      generator.run_engines_locally=true \\
      generator.async_engine=false \\
      generator.batched=true \\
      environment.env_class=gsm8k \\
      trainer.logger="wandb" \\
      trainer.project_name="gsm8k_sglang"

Configuration File Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also create a configuration file for SGLang backend:

.. code-block:: yaml

    # config/sglang_gsm8k.yaml
    
    defaults:
      - ppo_base_config
      
    generator:
      backend: "sglang"
      async_engine: false
      num_inference_engines: 4
      inference_engine_tensor_parallel_size: 1
      model_dtype: "bfloat16"
      run_engines_locally: true
      weight_sync_backend: "nccl"
      batched: true
      n_samples_per_prompt: 5
      
      sampling_params:
        max_new_tokens: 1024
        temperature: 1.0
        top_p: 1.0
        top_k: -1
        
    trainer:
      policy:
        model:
          path: "Qwen/Qwen2.5-1.5B-Instruct"
      strategy: "fsdp"
      
    environment:
      env_class: "gsm8k"

Limitations
-----------

Current limitations when using SGLang backend:

1. **No Async Engine**: SGLang doesn't support asynchronous engines, so ``generator.async_engine`` must be ``false``
2. **No Sleep/Wake**: SGLang doesn't have explicit sleep/wake functionality like vLLM
3. **Different Sampling Parameters**: Some vLLM-specific sampling parameters are not supported

Performance Considerations
--------------------------

- SGLang may have different memory usage patterns compared to vLLM
- Performance characteristics can vary depending on the model and hardware
- Test both backends to determine which works best for your specific use case

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import Error**: If you get import errors for SGLang modules, make sure you installed with the correct extra:

.. code-block:: bash

    uv run --isolated --extra sglang your_command

**Async Engine Error**: If you see errors about async engines, make sure ``generator.async_engine=false`` in your configuration.

**Sampling Parameter Error**: If you get errors about unsupported sampling parameters, check the SGLang documentation for supported parameters.

Getting Help
~~~~~~~~~~~~

For SGLang-specific issues:
- Check the `SGLang documentation <https://sgl-project.github.io/>`_
- Report issues to the `SGLang GitHub repository <https://github.com/sgl-project/sglang>`_

For integration issues with this framework:
- Check the existing tests in ``tests/sglang/``
- Report issues to this project's GitHub repository 