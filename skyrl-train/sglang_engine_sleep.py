import sglang as sgl
import torch
import time

if __name__ == "__main__":
    engine = sgl.Engine(
        model_path="Qwen/Qwen3-0.6B",
        enable_memory_saver=True,
    )
    print(f"Free GPU memory BEFORE SLEEP: {torch.cuda.mem_get_info()[0] / 1024**2:.1f} MB")
    engine.release_memory_occupation()
    print(f"Free GPU memory AFTER SLEEP: {torch.cuda.mem_get_info()[0] / 1024**2:.1f} MB")
    time.sleep(10)
    print(f"Free GPU memory AFTER WAITING 10 seconds: {torch.cuda.mem_get_info()[0] / 1024**2:.1f} MB")
