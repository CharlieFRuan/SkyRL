import os
import subprocess
import time

base_env = os.environ.copy()
script = "/home/ray/default/SkyRL/skyrl-train/examples/algorithms/dapo/run_dapo_qwen3_1.7b_aime_fullyAsync.sh"

for mini_batch_size in [512, 128, 64]:
    for max_staleness in [1, 4, 8]:
        for max_concurrency_raio in [1, 2, 4, 8]:
            if max_concurrency_raio > max_staleness + 1:
                continue
            max_concurrency = mini_batch_size * (max_staleness + 1) // max_concurrency_raio
            if max_concurrency < mini_batch_size:
                continue
            
            eval_ckpt_interval = 512 // mini_batch_size * 5
            my_env = {
                "MAX_STALENESS_STEPS": str(max_staleness),
                "NUM_PARALLEL_GENERATION_WORKERS": str(max_concurrency),
                "EVAL_CKPT_INTERVAL": str(eval_ckpt_interval),
                "MINI_BATCH_SIZE": str(mini_batch_size),
            }
            env = {
                **base_env,
                **my_env,
            }

            print("Running with", my_env)
            try:
                subprocess.run(
                    ["bash", script],
                    env=env,
                    cwd="/home/ray/default/SkyRL/skyrl-train",
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                # Keep going even if a run fails.
                print(f"Run failed (returncode={exc.returncode}) for {my_env}")
            time.sleep(60)
