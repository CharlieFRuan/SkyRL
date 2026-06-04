# Running SkyRL Multi-Node Jobs E2E (Together cluster, direct/non-TCLI)

This is a hands-on runbook for running SkyRL across **2+ Together GPU nodes directly over SSH**
(bypassing TCLI/k8s), validated on 2× H200 nodes (`gpu-dp-q9wbz-5q745` + `gpu-dp-q9wbz-77pcc`).
It covers environment setup, the NCCL/InfiniBand all-reduce sanity check, bringing up a Ray
cluster, and launching a 2-node `run_gsm8k.sh`.

> TL;DR of the hard-won lessons (see Troubleshooting for details):
> - **`/home` and `/scratch` are node-local** — there is no shared FS. Replicate everything per node.
> - Use a **persistent shared `.venv` at the same absolute path on every node**, and launch with the
>   venv's Python directly. Do **not** use `uv run --isolated` + the Ray uv runtime-env hook for the
>   driver in a multi-node cluster — it ships a base env without `ray` to remote workers.
> - The node FQDN resolves to a **k8s ClusterIP that is not bindable** → always pass **real IPs**
>   (`--node-ip-address`, `--local_addr`, `--master_addr`).
> - Interconnect is **InfiniBand 400G NDR**; pin NCCL to the 400G HCAs and use the routed NIC for OOB.
> - A node whose **NVSwitch fabric manager is not `active`** cannot run CUDA (error 802) — check first.

---

## 0. Variables used below

```bash
# Pick your nodes. HEAD is where you launch the driver.
HEAD_IP=10.40.16.194           # gpu-dp-q9wbz-5q745 (this node)
WORKER_IP=10.40.58.131         # gpu-dp-q9wbz-77pcc  (ssh alias: n2)
RAY_PORT=6379
GPUS_PER_NODE=8
NUM_NODES=2
PROJECT=/home/charlie_key/SkyRL
VENV=$PROJECT/.venv
# Same absolute paths must exist on EVERY node.
```

SSH config (`~/.ssh/config`) — we are already inside the cluster network, so connect node-to-node
directly by internal IP (no bastion needed):

```
Host n2 gpu-dp-q9wbz-77pcc
  HostName 10.40.58.131
  User charlie_key
  IdentityFile ~/.ssh/id_ed25519
  StrictHostKeyChecking accept-new
  ServerAliveInterval 60
```

`chmod 600 ~/.ssh/id_ed25519 ~/.ssh/config`. The key is authorized cluster-wide, so the same key
reaches any node by its internal IP.

---

## 1. Pre-flight: verify each candidate node is healthy

CUDA on H200 needs the **NVSwitch fabric manager** running. A node can pass `nvidia-smi` and even
`torch.cuda.is_available()` yet fail on the first real allocation with `CUDA error 802: system not
yet initialized` if the fabric is broken.

```bash
# fabric manager must be "active" (compare across nodes)
ssh n2 'systemctl is-active nvidia-fabricmanager'        # -> active   (failed = unusable)
ssh n2 "nvidia-smi -q | grep -A2 '^    Fabric'"          # State: Completed / Success
# definitive: a tiny allocation must succeed (no Error 802)
ssh n2 "$VENV/bin/python -c 'import torch; torch.empty(1, device=\"cuda\"); print(\"cuda ok\")'"
```

If `nvidia-fabricmanager` is `failed`, that node is out (fixing it needs root:
`sudo systemctl restart nvidia-fabricmanager`). Pick another node whose fabric is `active`.

Confirm the interconnect (expect several `InfiniBand ... 400 Gb/sec (4X NDR)` ports, all `ACTIVE`):

```bash
for d in /sys/class/infiniband/mlx5_*; do
  echo "$(basename $d): $(cat $d/ports/1/state) $(cat $d/ports/1/link_layer) $(cat $d/ports/1/rate)"
done
```

On these nodes the 400G NDR HCAs common to both are **`mlx5_4..9`** (the `ibp*` IPoIB interfaces show
`DOWN` — that's fine, NCCL uses IB verbs directly). The routed node IP rides **`enp2s0`**.

---

## 2. Environment setup (per node — no shared FS!)

`/home` and `/scratch` are local ext4 on each node, so the repo, dataset, and venv must exist on
**every** node at the **same absolute path**.

From the head node:

```bash
# (a) uv on the worker (same arch, just copy the binary)
ssh n2 'mkdir -p ~/.local/bin ~/data'
scp ~/.local/bin/uv n2:~/.local/bin/uv && ssh n2 'chmod +x ~/.local/bin/uv && ~/.local/bin/uv --version'

# (b) replicate the repo (skip venv/git/caches) + dataset
rsync -a --exclude='.venv' --exclude='.git' --exclude='*.log' --exclude='ckpts' \
      --exclude='__pycache__' --exclude='*.egg-info' \
      -e ssh $PROJECT/ n2:$PROJECT/
rsync -a -e ssh ~/data/gsm8k/ n2:~/data/gsm8k/     # if running gsm8k
```

Build a **persistent** venv on **both** nodes (this is the key to multi-node Ray working — every Ray
worker uses this exact interpreter):

```bash
# head
cd $PROJECT && uv sync --extra fsdp
# worker  (use --directory; ssh starts in $HOME, not the project)
ssh n2 'PATH=$HOME/.local/bin:$PATH uv sync --directory '"$PROJECT"' --extra fsdp'

# sanity (both must import cleanly)
$VENV/bin/python -c 'import ray, skyrl, torch, vllm; print(ray.__version__, torch.__version__)'
ssh n2 "$VENV/bin/python -c 'import ray, skyrl, torch, vllm; print(ray.__version__, torch.__version__)'"
```

> Backend extras conflict — use `--extra fsdp` **or** `--extra megatron`, never both.

---

## 3. NCCL / network environment

Export these wherever NCCL runs (the `ray start` shells on every node, and the driver). Putting them
on the `ray start` command means the raylet and all spawned workers inherit them.

```bash
export NCCL_IB_HCA=mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9   # 400G NDR ports (skip slow/RoCE)
export NCCL_SOCKET_IFNAME=enp2s0      # OOB/bootstrap over the routed node IP
export GLOO_SOCKET_IFNAME=enp2s0
# export NCCL_DEBUG=INFO              # uncomment to confirm transport selection
```

Confirm NCCL picks IB in logs: `NET/IB : Using [0]mlx5_4:1/IB ... ; OOB enp2s0:<ip>`.

---

## 4. Sanity check: internode all-reduce bandwidth

Uses `all_reduce_bench.py` (standard `torch.distributed.run`). Two upstream gotchas were fixed in our
copy: it hardcoded `FI_PROVIDER=efa` (AWS-only — removed) and was **missing the `run(local_rank)`
call** in `__main__` (added). If you grab a fresh copy, re-apply both.

Launcher (`run_bench.sh`) — note **real IPs** and `--local_addr` per node:

```bash
#!/bin/bash
NODE_RANK=${1:?need node_rank}; shift
if [ "$NODE_RANK" = "0" ]; then LOCAL_ADDR=10.40.16.194; else LOCAL_ADDR=10.40.58.131; fi
export PATH=$HOME/.local/bin:$PATH PYTHONUNBUFFERED=1
export NCCL_IB_HCA=mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 NCCL_SOCKET_IFNAME=enp2s0 GLOO_SOCKET_IFNAME=enp2s0
exec /home/charlie_key/SkyRL/.venv/bin/python -u -m torch.distributed.run \
  --nnodes=2 --node_rank="$NODE_RANK" --nproc_per_node=8 \
  --rdzv_backend=static --master_addr=10.40.16.194 --master_port=6000 --local_addr="$LOCAL_ADDR" \
  /home/charlie_key/all_reduce_bench.py "$@"
```

Run **both** ranks as **persistent background processes** (rank0 on head, rank1 on worker) so neither
is killed when a shell returns:

```bash
# rank0 (head) — background task
bash run_bench.sh 0 --payload_size_in_gib 4 --num_iterations 15 > bench_rank0.log 2>&1 &
# rank1 (worker) — keep the ssh open for the whole run
ssh n2 'bash /home/charlie_key/run_bench.sh 1 --payload_size_in_gib 4 --num_iterations 15' > bench_rank1.log 2>&1 &
wait
grep -E 'GBps|average bandwidth' bench_rank0.log    # result table prints on rank0
```

Expected on this fabric (2×8 H200, IB 400G NDR): **busbw ≈ 430 GB/s** at 4 GiB. Omit
`--payload_size_in_gib` to scan the full 32 KB–16 GB curve.

---

## 5. Bring up the Ray cluster (from the venv)

Start Ray with the **venv's `ray`** on every node so workers spawned by the raylet inherit a Python
that already has `ray`/`skyrl`/`torch`/`vllm`. Use **real IPs**; the worker must stay alive
independent of the ssh session, so use `--block` inside a held-open session (or `nohup setsid`).

```bash
# HEAD (this node)
cd $PROJECT
NCCL_IB_HCA=mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 NCCL_SOCKET_IFNAME=enp2s0 GLOO_SOCKET_IFNAME=enp2s0 \
  $VENV/bin/ray start --head --node-ip-address=$HEAD_IP --port=$RAY_PORT --num-gpus=$GPUS_PER_NODE --dashboard-host=0.0.0.0

# WORKER — run in a PERSISTENT background session (--block keeps it foreground under the ssh)
ssh n2 'NCCL_IB_HCA=mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 NCCL_SOCKET_IFNAME=enp2s0 GLOO_SOCKET_IFNAME=enp2s0 \
  /home/charlie_key/SkyRL/.venv/bin/ray start --address=10.40.16.194:6379 \
  --node-ip-address=10.40.58.131 --num-gpus=8 --block' > ray_worker.log 2>&1 &

# verify: should report ALIVE=2, GPU=16
$VENV/bin/python - <<'PY'
import ray; ray.init(address='10.40.16.194:6379', logging_level='ERROR')
print('ALIVE', sum(n['Alive'] for n in ray.nodes()),
      'GPU', sum(n['Resources'].get('GPU',0) for n in ray.nodes() if n['Alive']))
PY
```

> If the worker shows up briefly then drops (CPU counts but GPU stays 8): it was SIGHUP'd when the ssh
> closed — keep the session open with `--block`, or detach with `nohup setsid`. If the worker errors
> `Failed to spawn: ray` / `--extra fsdp has no effect`, you launched outside the project dir.

---

## 6. Launch the multi-node SkyRL job

Launch the driver with the **venv Python directly** (`run_gsm8k_venv.sh` is `run_gsm8k.sh` with
`uv run --isolated --extra fsdp -m` replaced by `$VENV/bin/python -m`), connect to the existing
cluster via `RAY_ADDRESS`, and **do not set `RAY_RUNTIME_ENV_HOOK`**.

```bash
sed 's#uv run --isolated --extra fsdp -m#'"$VENV"'/bin/python -m#' \
    examples/train/gsm8k/run_gsm8k.sh > ~/run_gsm8k_venv.sh

cd $PROJECT
RAY_ADDRESS=$HEAD_IP:$RAY_PORT NUM_GPUS=$GPUS_PER_NODE LOGGER=console \
NCCL_IB_HCA=mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 NCCL_SOCKET_IFNAME=enp2s0 GLOO_SOCKET_IFNAME=enp2s0 \
  bash ~/run_gsm8k_venv.sh \
    trainer.placement.policy_num_nodes=$NUM_NODES \
    trainer.placement.critic_num_nodes=$NUM_NODES \
    trainer.placement.ref_num_nodes=$NUM_NODES \
    generator.inference_engine.num_engines=16 \
    trainer.micro_train_batch_size_per_gpu=40 \
    trainer.micro_forward_batch_size_per_gpu=40 \
  &> gsm8k_2node.log
```

Why these overrides:
- `*_num_nodes=2` + `NUM_GPUS=8` ⇒ `*_num_gpus_per_node=8` ⇒ a single 16-GPU data-parallel group.
- `num_engines=16` — colocated vLLM with `tensor_parallel_size=1` needs one engine per GPU (16).
- **Batch-size divisibility:** `policy_mini_batch_size_per_gpu = policy_mini_batch_size *
  n_samples_per_prompt / (num_nodes*gpus_per_node)`. Here `256*5/16 = 80`, which must be divisible by
  `micro_train_batch_size_per_gpu`. The default `64` fails (80 % 64 ≠ 0); `40` works (and divides the
  `320` train-batch/GPU too). Recompute this whenever you change GPU count or batch sizes.

Healthy progress markers in `gsm8k_2node.log`:
`Mesh Ranks: [... world_size=16 ...]` → `init policy/ref/critic models done` →
`Initialized weight sync state` → `Finished: 'sync_weights'` → `Step 0:` (baseline eval, ~pass@1 0.08)
→ `Finished: 'generate'` → `Finished: 'step'` → `Step 1:`.

---

## 7. Teardown

```bash
pkill -f skyrl.train.entrypoints.main_base    # stop the driver
$VENV/bin/ray stop                            # stop head
ssh n2 '/home/charlie_key/SkyRL/.venv/bin/ray stop'   # stop worker
# kill the held-open worker ssh background job too (jobs -l / kill %N)
```

If you reserved nodes with a taint, release them:
`kubectl taint nodes <node> reserved-by=<you>:NoSchedule-`.

---

## 8. Troubleshooting (symptom → cause → fix)

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA error 802: system not yet initialized` (often on first alloc; `nvidia-smi` looks fine) | NVSwitch **fabric manager not active** on that node | `systemctl is-active nvidia-fabricmanager`; restart needs root, else pick a healthy node |
| torchrun hangs at `c10d ... server socket ... has timed out` | node **FQDN resolves to a k8s ClusterIP** that can't be bound | pass **real IPs**: `--master_addr`, `--local_addr`, `--node-ip-address` |
| Ray worker: `setup_worker.py ... ModuleNotFoundError: No module named 'ray'` (or `skyrl`) | uv hook reproduced `uv run` on the worker **outside the project dir** (or a cold/racy build) → env missing `skyrl`/`ray` | run uv with `--directory /home/charlie_key/SkyRL` (or `cd`) everywhere, and **warm the cache on each node** first (Section 10). Or use the `.venv` route (Section 5–6). |
| Worker `Failed to spawn: ray` / `--extra fsdp has no effect when used outside of a project` | ran `uv` from `$HOME` (ssh default cwd), not the project | `cd $PROJECT` or `uv --directory $PROJECT ...` |
| Worker joins (CPU counts) then drops; GPUs never reach 16 | `ray start` daemons **SIGHUP'd** when the ssh session closed | run with `--block` in a held-open session, or `nohup setsid` |
| `AssertionError: ... policy_mini_batch_size_per_gpu N should be divisible by micro_train_batch_size_per_gpu M` | batch math doesn't divide at the new GPU count | set `micro_{train,forward}_batch_size_per_gpu` to a divisor of `mini_batch*n_samples/world_size` |
| all-reduce bench "succeeds" instantly with **no result table** | upstream bug: `__main__` never calls `run()`; or block-buffered stdout | ensure `run(local_rank)` is called; set `PYTHONUNBUFFERED=1` |
| NCCL falls back to slow path / low busbw | wrong/extra HCAs selected (10G SDR or RoCE) | pin `NCCL_IB_HCA` to the 400G NDR ports common to all nodes; `NCCL_SOCKET_IFNAME=enp2s0` |
| Repeated `Failed to establish connection to the metrics exporter agent` | Ray prometheus/metrics agent only | benign — ignore |

---

## 9. Notes specific to this cluster

- `5q745` ↔ `77pcc` internal IPs: `10.40.16.194` / `10.40.58.131`; both reachable by `~/.ssh/id_ed25519`.
- `m8htz` (`10.40.62.120`) has a **broken fabric manager** (failed since 2026-05-27) — do not use until an
  admin restarts `nvidia-fabricmanager`.
- Reserve a node from TCLI scheduling with a `NoSchedule` taint (TCLI's `_is_valid_gpu_node` skips tainted
  nodes): `kubectl taint nodes <node> reserved-by=<you>:NoSchedule`.

---

## 10. Launching jobs with `uv run --isolated` (preferred over a persistent `.venv`)

Sections 5–6 use a persistent `.venv` because it's the most deterministic. If you prefer the SkyRL
convention of `uv run --isolated` (no `.venv` to manage), it works too — but two things are
**mandatory**, and they were the actual cause of the original `ModuleNotFoundError: No module named 'ray'`
on remote workers:

1. **uv must run from the project on *every* node, including workers.** `ssh` lands in `$HOME`, not the
   project, so `uv run --isolated --extra fsdp ...` there builds an env *outside the project* (you'll see
   `--extra fsdp has no effect when used outside of a project`) with **no `skyrl`/`ray`**. Always use
   `uv run --directory /home/charlie_key/SkyRL --isolated --extra fsdp ...` (or `cd` first) on remote
   commands. The Ray uv runtime-env hook reproduces the driver's `uv run` on each worker, so if the
   driver's working dir / project context is wrong, every worker fails the same way.

2. **Warm the uv cache on each node first**, so workers reuse a fully-built env instead of each racing to
   build one (a cold/racy build is what surfaces as a half-populated env missing `ray`):
   ```bash
   # head
   cd /home/charlie_key/SkyRL && uv run --isolated --extra fsdp python -c "import skyrl, skyrl.train, ray, torch, vllm"
   # worker (note --directory)
   ssh n2 'uv run --directory /home/charlie_key/SkyRL --isolated --extra fsdp python -c "import skyrl, skyrl.train, ray, torch, vllm"'
   ```
   (Module is `skyrl` / `skyrl.train` — there is no top-level `skyrl_train`.)

Then enable the uv hook and launch the driver from the project dir (this is the SkyRL-documented path
for an existing cluster on Ray ≥ 2.48):
```bash
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
cd /home/charlie_key/SkyRL
RAY_ADDRESS=$HEAD_IP:$RAY_PORT NUM_GPUS=8 LOGGER=console NCCL_IB_HCA=mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 \
NCCL_SOCKET_IFNAME=enp2s0 GLOO_SOCKET_IFNAME=enp2s0 \
  bash examples/train/gsm8k/run_gsm8k.sh \
    trainer.placement.policy_num_nodes=2 trainer.placement.critic_num_nodes=2 trainer.placement.ref_num_nodes=2 \
    generator.inference_engine.num_engines=16 \
    trainer.micro_train_batch_size_per_gpu=40 trainer.micro_forward_batch_size_per_gpu=40 \
  &> gsm8k_2node.log
```
`run_gsm8k.sh` already uses `uv run --isolated --extra fsdp -m ...`, so no edit is needed (unlike the
`.venv` path which needs the `sed` swap).

**Caveat observed on this cluster:** starting the *Ray daemons* with `uv run --isolated ... ray start`
on the worker sometimes registered the node as `Active` but with **0 GPU/0 CPU** (resources never
attached), so the head saw only 8 GPUs. Starting the daemons from the synced env
(`.venv/bin/ray start`, Section 5) registered all 16 reliably. If you hit the 0-resource symptom with
`uv run`, bring up the *cluster* via the `.venv/bin/ray` route (one-time `uv sync`) but still **launch
the job with `uv run --isolated`** — the two are independent. (`uv sync` for the daemons is not the same
as managing a `.venv` for the job env.)
