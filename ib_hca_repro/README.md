# Minimal repro: mlx5_2 / mlx5_3 are bad InfiniBand HCAs on this cluster

2-node, 1-GPU-per-node NCCL all_reduce; only NCCL_IB_HCA changes between runs.
Stage nccl_ib_minrepro.py + min_ib.sh to BOTH nodes (/home is node-local), then:
  bash min_ib.sh mlx5_2,mlx5_3 0 10.40.16.194    # head (rank0)
  bash min_ib.sh mlx5_2,mlx5_3 1 10.40.58.131    # peer (rank1)   -> CRASHES
  bash min_ib.sh mlx5_4,mlx5_5 0 10.40.16.194
  bash min_ib.sh mlx5_4,mlx5_5 1 10.40.58.131    -> SUCCESS

Result (2026-06-04):
  mlx5_2,mlx5_3 -> NCCL WARN NET/IB IBV_WC_RETRY_EXC_ERR(12) hca mlx5_2 -> ncclRemoteError -> abort
  mlx5_4,mlx5_5 -> 5/5 all_reduce OK (256MB ~0.02s) -> SUCCESS
=> mlx5_2 / mlx5_3 do not carry RDMA traffic (transport retry exhausted); mlx5_4..mlx5_9 are healthy.
Production fix: NCCL_IB_HCA="mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9" (set in skyrl/train/utils/utils.py).

## Per-node, not cluster-wide: it's the HEAD node (5q745)
ibstat across nodes:
  5q745(head): mlx5_2 & mlx5_3 = Physical state "Polling", Rate 10  (links DOWN/negotiating)
               mlx5_4..mlx5_9 = LinkUp, Rate 400
  n2, n3, m8htz: mlx5_2..mlx5_9 ALL = LinkUp, Rate 400 (healthy)
Confirm: n2<->n3 all_reduce on mlx5_2,mlx5_3 (head excluded) = SUCCESS.
=> The fault is the HEAD node 5q745's mlx5_2/mlx5_3 ports (phys "Polling", degraded 10 Gb/s),
   NOT the mlx5_2/mlx5_3 rails cluster-wide. Every cross-node collective includes the head
   (rank 0 / colocate driver), so NCCL routing through the head's dead links broke any peer.
Real fix: service 5q745's mlx5_2/mlx5_3 (cable/transceiver/switch-port). Workaround in use:
   NCCL_IB_HCA=mlx5_4..mlx5_9 globally (costs 2/8 rails on all nodes, but avoids the head's dead ports).
