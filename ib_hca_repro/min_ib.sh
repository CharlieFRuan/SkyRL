#!/bin/bash
HCA=$1; RANK=$2; LADDR=$3
export NCCL_IB_HCA=$HCA NCCL_SOCKET_IFNAME=enp2s0 GLOO_SOCKET_IFNAME=enp2s0
export NCCL_DEBUG=WARN PYTHONUNBUFFERED=1
NV=/home/charlie_key/SkyRL/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$(echo $NV/*/lib|tr ' ' :):/usr/local/cuda/lib64
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH
exec /home/charlie_key/SkyRL/.venv/bin/python -m torch.distributed.run \
  --nnodes=2 --node_rank=$RANK --nproc_per_node=1 \
  --rdzv_backend=static --master_addr=10.40.16.194 --master_port=6002 --local_addr=$LADDR \
  /home/charlie_key/nccl_ib_minrepro.py
