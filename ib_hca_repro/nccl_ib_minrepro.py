import os, time, torch, torch.distributed as dist
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
x = torch.ones(64 * 1024 * 1024, device="cuda")  # 256 MB tensor
if rank == 0:
    print(f"[repro] world={world} HCA={os.environ.get('NCCL_IB_HCA')} -- starting allreduce", flush=True)
for i in range(5):
    t = time.time(); dist.all_reduce(x); torch.cuda.synchronize()
    if rank == 0:
        print(f"[repro] iter {i}: all_reduce OK in {time.time()-t:.3f}s (x[0]={x[0].item()})", flush=True)
if rank == 0:
    print("[repro] SUCCESS — this HCA set works over IB", flush=True)
dist.destroy_process_group()
