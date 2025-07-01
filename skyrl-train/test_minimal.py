import sglang as sgl
from omegaconf.dictconfig import DictConfig
import ray


def initialize_ray():
    env_vars = {
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_P2P_DISABLE": "0",
        "CUDA_LAUNCH_BLOCKING": "1",
    }
    ray.init(runtime_env={"env_vars": env_vars})

def main():
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params object.
    sampling_params = {"temperature": 0, "max_new_tokens": 30}

    # Create an LLM.
    llm = sgl.Engine(
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
    )

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()
