import os
import subprocess
import time

mini_batch_size = 256
base_env = os.environ.copy()
script = "/home/ray/default/SkyRL/skyrl-train/examples/fully_async/async_run_gsm8k.sh"

for ratio in ["4-4", "6-2"]:
    for max_staleness in [1, 4, 8]:
        for max_concurrency_raio in [1, 2, 4, 8]:
            if ratio == "4-4" and max_staleness == 1 or max_staleness == 4:
                # already ran
                continue

            if ratio == "6-2":
                num_inf_gpus = 6
                num_train_gpus = 2
            else:
                num_inf_gpus = 4
                num_train_gpus = 4
            if max_concurrency_raio > max_staleness + 1:
                continue
            max_concurrency = mini_batch_size * (max_staleness + 1) // max_concurrency_raio
            if max_concurrency < mini_batch_size:
                continue
            my_env = {
                "NUM_INFERENCE_GPUS": str(num_inf_gpus),
                "NUM_POLICY_GPUS": str(num_train_gpus),
                "MAX_STALENESS_STEPS": str(max_staleness),
                "NUM_PARALLEL_GENERATION_WORKERS": str(max_concurrency),
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
