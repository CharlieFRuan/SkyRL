# Running SkyRL Multi-Node Jobs E2E (Together cluster, direct/non-TCLI)

This is a hands-on runbook for running SkyRL across **2+ Together GPU nodes directly over SSH**
(bypassing TCLI/k8s). First validated on 2× H200 (`gpu-dp-q9wbz-5q745` + `gpu-dp-q9wbz-77pcc`) and
since run at **4× H200** for Qwen3.6-35B-A3B DAPO/LoRA. It covers environment setup, the
NCCL/InfiniBand all-reduce sanity check, bringing up a Ray cluster, and launching a multi-node job.

> **Starting a cluster on different/new nodes:** the node IPs, IB HCA names, and routed NIC are
> **per-cluster/per-node — discover them (§1), don't blindly copy the literals below.** The values
> shown (`mlx5_2..9`, `enp2s0`) are just what these particular nodes use; the IB HCA set and especially
> the socket interface vary by node.

> TL;DR of the hard-won lessons (see Troubleshooting for details):
> - **`/home` and `/scratch` are node-local** — there is no shared FS. Replicate everything per node.
> - Use a **persistent shared `.venv` at the same absolute path on every node**, and launch with the
>   venv's Python directly. Do **not** use `uv run --isolated` + the Ray uv runtime-env hook for the
>   driver in a multi-node cluster — it ships a base env without `ray` to remote workers.
> - The node FQDN resolves to a **k8s ClusterIP that is not bindable** → always pass **real IPs**
>   (`--node-ip-address`, `--local_addr`, `--master_addr`).
> - Interconnect is **InfiniBand 400G NDR**; pin NCCL to the 400G HCAs (discover per `ibstat`/§1 — here
>   `mlx5_2..9`, skipping the 100G `mlx5_0/1`). This is the **data** path.
> - **`NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME` are PER-NODE** (the TCP control/bootstrap path). The
>   routed 10.40 NIC differs across nodes (`enp2s0` on most, `enp1s0` on at least one) — **auto-detect
>   on each node**; a single hardcoded value silently breaks the odd node. Set them at `ray start` (the
>   raylet hands them to the workers it spawns).
> - A node whose **NVSwitch fabric manager is not `active`** cannot run CUDA (error 802) — check first.
> - **Long-running (multi-day) Ray sessions get their `/tmp` session dir reaped** (the `logs/` +
>   `node_ip_address.json` disappear): housekeeping tasks fail with `FileNotFoundError`, or new drivers
>   can't connect (`a ray instance hasn't started`). Fix = **restart Ray into a fresh session** (§8).

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
# Keep Ray's session/spill dirs OFF the 93G root (/tmp) — put them on the big
# /home (or /scratch) array. See §11. Must be exported on every `ray start`.
RAY_TMP=/home/charlie_key/ray_tmp
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

Pick the **400G NDR HCAs that are `ACTIVE` on every node** for `NCCL_IB_HCA`. On these nodes the full
healthy set is **`mlx5_2..9`** (8 ports); **`mlx5_0/1` are the 100G rails and are skipped**. (History:
while 5q745's `mlx5_2/mlx5_3` were transiently down we pinned the 6-port subset `mlx5_4..9`; once the
HCAs were repaired we went back to all 8. Rule: **exclude any port not `ACTIVE`/400G on *all* nodes**,
because a dead HCA shows up as `IBV_WC_RETRY_EXC_ERR(12)` → `ncclRemoteError` mid-run, not at startup.)
The `ibp*` IPoIB interfaces showing `DOWN` is fine — NCCL uses IB verbs directly.

Find the **routed node IP and its NIC, per node** — this is what NCCL/Gloo OOB and `--node-ip-address`
use, and **it is not the same interface name on every node**:
```bash
ip -o -4 addr show | awk '$4 ~ /^10\.40\./ {print $2, $4}'   # e.g. enp2s0 on most nodes, enp1s0 on n4
```

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
# IB HCA (DATA path) = the 400G ports ACTIVE on ALL nodes (from §1). These nodes: mlx5_2..9 (skip 100G mlx5_0/1).
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
# SOCKET_IFNAME (TCP CONTROL/bootstrap path) is PER-NODE — auto-detect the routed 10.40 NIC on THIS node:
export IFACE=$(ip -o -4 addr show | awk '$4 ~ /^10\.40\./ {print $2; exit}')   # enp2s0 / enp1s0 / ...
export NCCL_SOCKET_IFNAME=$IFACE
export GLOO_SOCKET_IFNAME=$IFACE      # Gloo CPU collectives (init_process_group/barrier) are all-TCP too
# export NCCL_DEBUG=INFO              # uncomment to confirm transport selection
```

`NCCL_IB_HCA` selects the **data** transport (collectives ride IB); the `*_SOCKET_IFNAME` vars only pin
the **TCP control/bootstrap + out-of-band** path, so the NIC being a modest Ethernet link is fine. The
socket vars are **per-node**: auto-detect them in every `ray start` shell — do **not** hardcode one
value cluster-wide, or it will be wrong on a node whose routed NIC differs (e.g. `enp1s0`) and the
multi-node process-group init will hang. Confirm NCCL picks IB in logs:
`NET/IB : Using [0]mlx5_2:1/IB ... ; OOB <iface>:<ip>`.

---

## 4. Sanity check: internode all-reduce bandwidth

Uses `all_reduce_bench.py` (standard `torch.distributed.run`), now at
**`/home/charlie_key/SkyRL/all_reduce_bench.py`** (moved under the repo; replicate per node — `/home`
is node-local). Two upstream gotchas were fixed in our copy: it hardcoded `FI_PROVIDER=efa`
(AWS-only — removed) and was **missing the `run(local_rank)` call** in `__main__` (added). If you
grab a fresh copy, re-apply both.

> **Focused per-HCA / per-pair tester:** to check *which* IB rail is bad between *which* two nodes,
> use `SkyRL/ib_hca_repro/ib_pair_test.sh` (auto-detects each node's routed NIC, so it works for n4's
> `enp1s0` too). It runs a 2-node NCCL all_reduce over a chosen `NCCL_IB_HCA` set and prints PASS/FAIL:
> ```bash
> cd /home/charlie_key/SkyRL/ib_hca_repro
> bash ib_pair_test.sh mlx5_3 head:10.40.16.194 n4:10.40.40.19            # one rail (FAIL if dead)
> bash ib_pair_test.sh mlx5_2,mlx5_3,...,mlx5_9 head:10.40.16.194 n4:10.40.40.19 6061 1 10 8  # 8 GPUs/node
> ```
> **Use `nproc_per_node=8` (the last arg) to catch a degraded rail.** With 1 GPU/node NCCL only uses
> GPU0's local rail, so a single-bad-rail mixed into an 8-rail set *passes* a 1-GPU test but *fails* the
> 8-GPU test (the GPU affinitized to the bad rail hits it) — which is what kills real 8-GPU/node training.

Launcher (`run_bench.sh`) — note **real IPs** and `--local_addr` per node:

```bash
#!/bin/bash
NODE_RANK=${1:?need node_rank}; shift
if [ "$NODE_RANK" = "0" ]; then LOCAL_ADDR=10.40.16.194; else LOCAL_ADDR=10.40.58.131; fi
export PATH=$HOME/.local/bin:$PATH PYTHONUNBUFFERED=1
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 NCCL_SOCKET_IFNAME=enp2s0 GLOO_SOCKET_IFNAME=enp2s0
exec /home/charlie_key/SkyRL/.venv/bin/python -u -m torch.distributed.run \
  --nnodes=2 --node_rank="$NODE_RANK" --nproc_per_node=8 \
  --rdzv_backend=static --master_addr=10.40.16.194 --master_port=6000 --local_addr="$LOCAL_ADDR" \
  /home/charlie_key/SkyRL/all_reduce_bench.py "$@"
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
# IB HCA set (same on all healthy nodes); IFACE is auto-detected PER node (see below).
HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9

# HEAD (this node)
cd $PROJECT
IFACE=$(ip -o -4 addr show | awk '$4 ~ /^10\.40\./ {print $2; exit}')   # this node's routed NIC
NCCL_IB_HCA=$HCA NCCL_SOCKET_IFNAME=$IFACE GLOO_SOCKET_IFNAME=$IFACE TMPDIR=$RAY_TMP RAY_TMPDIR=$RAY_TMP \
  $VENV/bin/ray start --head --node-ip-address=$HEAD_IP --port=$RAY_PORT --num-gpus=$GPUS_PER_NODE --dashboard-host=0.0.0.0

# WORKER — IFACE is auto-detected ON THE WORKER (may be enp1s0, not enp2s0!). PERSISTENT session (--block).
# TMPDIR/RAY_TMPDIR keep the worker's Ray session dir off its root disk too (§11).
ssh n2 'IFACE=$(ip -o -4 addr show|grep " 10.40"|awk "{print \$2}"|head -1); \
  NCCL_IB_HCA='"$HCA"' NCCL_SOCKET_IFNAME=$IFACE GLOO_SOCKET_IFNAME=$IFACE \
  TMPDIR=/home/charlie_key/ray_tmp RAY_TMPDIR=/home/charlie_key/ray_tmp \
  /home/charlie_key/SkyRL/.venv/bin/ray start --address=10.40.16.194:6379 \
  --node-ip-address=10.40.58.131 --num-gpus=8 --block' > ray_worker.log 2>&1 &
# (verify each node's raylet got the right NIC:  tr '\0' '\n' </proc/$(pgrep -x raylet)/environ | grep SOCKET_IFNAME )

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
NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 NCCL_SOCKET_IFNAME=enp2s0 GLOO_SOCKET_IFNAME=enp2s0 \
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
| NCCL falls back to slow path / low busbw | wrong/extra HCAs selected (100G or RoCE) | pin `NCCL_IB_HCA` to the 400G NDR ports common to all nodes (skip `mlx5_0/1`) |
| Multi-node process-group init **hangs** at `init_worker_process_group` / Gloo `connectFullMesh ... Connection refused remote=[127.0.0.1]` | `*_SOCKET_IFNAME` unset or hardcoded to a NIC name that **doesn't exist on that node** (NIC differs per node) → auto-detect picks a non-routable iface / loopback | set `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME` to the node's own routed 10.40 NIC **at `ray start`** (§3 auto-detect). NOT a single cluster-wide value |
| `uv run --extra megatron --isolated ...` fails building `transformer-engine-torch` with `fatal error: cudnn.h: No such file or directory` (then, after fixing that, `nccl.h`) | Transformer Engine has no prebuilt wheel for this exact torch/CUDA tag, so it builds from source. `CUDA_HOME=/usr/local/cuda` only adds `/usr/local/cuda/include`; on these nodes cuDNN/NCCL headers come from Python NVIDIA wheels under `.venv/lib/python3.12/site-packages/nvidia/.../include` | export `CPATH`, `LIBRARY_PATH`, and `LD_LIBRARY_PATH` for the wheel-provided cuDNN/NCCL dirs before warming/building the megatron env (see §10). Once built, uv reuses the cached wheel |
| Mid-run crash: `cleanup_old_checkpoints ... FileNotFoundError: .../session_*/logs/worker-*.out`, or relaunch fails `Can't find node_ip_address.json ... a ray instance hasn't started` | a `/tmp` reaper deleted the **stale (multi-day) Ray session dir** out from under the live cluster (logs + `node_ip_address.json` gone) → zombie cluster | **restart Ray into a fresh session**: `ray stop` on all nodes (+ `pkill -9 -x raylet gcs_server`), then `ray start --head` again + rejoin workers. A fresh session resets the reap clock. Not disk-full and not a training bug |
| `IBV_WC_RETRY_EXC_ERR(12)` / `ncclRemoteError` appearing **mid-training** (not at startup), often at the backward all-reduce | a specific IB HCA is physically down/degraded on one node (e.g. `Polling`/`10 Gb` instead of `LinkUp`/`400`) and NCCL auto-selected it | `ibstat`/§1 across all nodes; **drop the bad port from `NCCL_IB_HCA`** (or repair it). Don't fall back to TCP (`NCCL_IB_DISABLE=1`) unless you must — it's ~slow |
| Repeated `Failed to establish connection to the metrics exporter agent` | Ray prometheus/metrics agent only | benign — ignore |

---

## 9. Notes specific to this cluster

- Node roster used for the 4-node runs (IP / routed NIC — **note n4 differs**), all reachable by `~/.ssh/id_ed25519`:
  | alias | node | IP | routed NIC |
  |-------|------|----|-----------|
  | (head) | gpu-dp-q9wbz-5q745 | 10.40.16.194 | enp2s0 |
  | n2 | gpu-dp-q9wbz-77pcc | 10.40.58.131 | enp2s0 |
  | n4 | gpu-dp-q9wbz-8vnx2 | 10.40.40.19 | **enp1s0** |
  | m8htz | gpu-dp-q9wbz-m8htz | 10.40.62.120 | enp2s0 |
  (the head also has non-routable ifaces — `enp7s0`/`enp8s0`/`vxlan.calico` — which is why auto-detect must
  filter for the `10.40.` subnet.)
- `m8htz`'s fabric manager was **broken (failed 2026-05-27) but has since been repaired** — it ran the full
  4-node job fine. Always re-check `nvidia-fabricmanager`/`ibstat` per §1 before trusting any node; health
  changes over time (5q745's `mlx5_2/mlx5_3` were also down for a while, then fixed).
- **(2026-06-16) 5q745's `mlx5_3` is FLAPPING** — observed at **10 Gb/s SDR** (~17:00) then back to **400
  NDR** (~18:35); it oscillates over minutes and intermittently fails NCCL QP setup. `mlx5_2` recovered.
  While it's down, the full `mlx5_2..9` set FAILS at 8 GPUs/node (`IBV_WC_RETRY_EXC_ERR(12) hca mlx5_3`);
  while it's up the same test PASSES — so the live NCCL repro is intermittent (failure is at QP setup, not
  per-transfer). This caused the silent `SYSTEM_ERROR` worker kills at the inter-step weight-sync in the
  Qwen3.6 35B runs (they landed in a down-window). **Catch the flap deterministically with
  `ib_hca_repro/ib_watch.sh`** (polls link rate; no GPUs needed); reproduce the NCCL impact with
  `ib_hca_repro/ib_check.sh` (run in a loop). **Workaround applied:** `NCCL_IB_HCA` in
  `skyrl/train/utils/utils.py` drops `mlx5_3` (`mlx5_2,mlx5_4..9`, 7 rails). Real fix: service 5q745's
  `mlx5_3` (cable/transceiver/switch port), then restore the 8-rail set once `ib_watch.sh` stays 400 NDR.
- Reserve a node from TCLI scheduling with a `NoSchedule` taint (TCLI's `_is_valid_gpu_node` skips tainted
  nodes): `kubectl taint nodes <node> reserved-by=<you>:NoSchedule`; release with the trailing `-`.
- Found a fresh node? Run §1 (fabric + `ibstat` 400G ports + `ip addr` routed NIC) **before** adding it —
  that's the whole point of the per-node discovery. SkyRL backend env (`NCCL_IB_HCA`, debug) for the actual
  training runs is set in `skyrl/train/utils/utils.py` and propagated via Ray runtime_env; the `*_SOCKET_IFNAME`
  vars are NOT propagated (they ride the per-node raylet env from `ray start`).

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

3. **If warming `--extra megatron`, include the Python-wheel cuDNN/NCCL headers.** On a cold cache,
   `transformer-engine-torch==2.11.0` first guesses a prebuilt wheel URL for the exact torch/CUDA tag
   (`torch2.11.0+cu128` here). That URL 404s, so it builds from source. The source build includes
   `ATen/cudnn/cudnn-wrapper.h`, which needs plain `<cudnn.h>`, and later Transformer Engine headers
   include `nccl.h`. Those headers are present, but in the Python NVIDIA wheels, not in
   `/usr/local/cuda/include`; setting only `CUDA_HOME=/usr/local/cuda` is not enough.

   Build/warm once per node with the include and library paths exported:
   ```bash
   cd /home/charlie_key/SkyRL
   export NVIDIA_SITE=$PWD/.venv/lib/python3.12/site-packages/nvidia
   export CPATH=$NVIDIA_SITE/cudnn/include:$NVIDIA_SITE/nccl/include${CPATH:+:$CPATH}
   export LIBRARY_PATH=$NVIDIA_SITE/cudnn/lib:$NVIDIA_SITE/nccl/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
   export LD_LIBRARY_PATH=$NVIDIA_SITE/cudnn/lib:$NVIDIA_SITE/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
   uv run --extra megatron --isolated python -c "import transformer_engine; import transformer_engine_torch"
   ```

   After this succeeds, the exact command without those exports should reuse uv's cached
   `transformer-engine-torch` wheel:
   ```bash
   uv run --extra megatron --isolated python -c "import transformer_engine"
   ```

Then enable the uv hook and launch the driver from the project dir (this is the SkyRL-documented path
for an existing cluster on Ray ≥ 2.48):
```bash
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
cd /home/charlie_key/SkyRL
RAY_ADDRESS=$HEAD_IP:$RAY_PORT NUM_GPUS=8 LOGGER=console NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 \
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

---

## 11. Keep Ray + SkyRL logs OFF the 93G root (`/tmp`)

The root FS `/` (`/dev/vda2`) is **only 93G** on these nodes; `/home` and `/scratch` are the big
~7TB `/dev/md0` array. Two things default to `/tmp` (root) and will fill it / crash the run:

1. **Ray session + spill dir** → defaults to `/tmp/ray`. Each Ray session also spills the object
   store and (if the uv runtime-env hook is active) a copy of the working dir there. Ray derives this
   from `tempfile.gettempdir()`, so it honors **`TMPDIR`** (and **`RAY_TMPDIR`**). Export both to a
   `/home` path **on every `ray start`** (head + workers) — see §0 (`RAY_TMP`) and §5. The worker's
   value rides the per-node `ray start` env (like `*_SOCKET_IFNAME`); it is not propagated for you.
2. **SkyRL infra logs** (`vLLM`/worker stdout, `infra-*.log`) → controlled by the
   **`trainer.log_path`** config (default `/tmp/skyrl-logs`). Override it on the driver command line:
   `trainer.log_path=/home/charlie_key/skyrl-logs`. (The `shiyi_glm47_flash_fully_async.sh` script
   sets this via its `LOG_PATH` var.)

Quick check after launch — both should live under `/home`, and `/` should stay flat:
```bash
du -sh /home/charlie_key/ray_tmp /home/charlie_key/skyrl-logs ; df -h / | tail -1
ls -d /tmp/ray 2>/dev/null && echo "WARNING: Ray still using /tmp/ray — TMPDIR not set at ray start"
```

> Bonus (single-node `uv run` tinker/runs): `export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` stops Ray from
> shipping the 13G `.venv` working-dir into the session dir on every launch. Not needed for the
> `.venv/bin/python` multi-node path (§5–6) since that doesn't trigger the uv hook at all.
