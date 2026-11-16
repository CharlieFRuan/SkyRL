"""Back up one Megatron checkpoint to the cluster's artifact bucket, so it survives a cluster restart. Run from the HEAD
(only the head's role can write the bucket): each file streams from the worker holding its largest copy (shards are
spread over the training nodes) straight into S3, no local disk.

    python3 ops/upload_ckpt_s3.py <run> <global_step>
    -> $ANYSCALE_ARTIFACT_STORAGE/charlie/ckpts/<run>/global_step_<step>/...

Restore: `aws s3 cp --recursive <prefix> $D/runs/<run>/ckpts/global_step_<step>/` on the head, rsync to the training
nodes, and write latest_ckpt_global_step.txt (see ops/sync_ckpt.py).
"""

import concurrent.futures
import os
import subprocess
import sys

RUN, STEP = sys.argv[1], sys.argv[2]
SRC = f"/mnt/local_storage/charlie/runs/{RUN}/ckpts/global_step_{STEP}"
DST = f"{os.environ['ANYSCALE_ARTIFACT_STORAGE']}/charlie/ckpts/{RUN}/global_step_{STEP}"
SSH = ["ssh", "-p", "2222", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "-c", "aes128-gcm@openssh.com"]

nodes = subprocess.run(["bash", "/home/ray/hackskyrl/ops/workers.sh"], capture_output=True, text=True).stdout.split()
listing = {}
for n in nodes:
    out = subprocess.run([*SSH, n, f"cd {SRC} 2>/dev/null && find . -type f -printf '%s %P\\n'"], capture_output=True, text=True).stdout
    listing[n] = {p: int(s) for s, p in (line.split(" ", 1) for line in out.splitlines() if line)}
files = {}
for n in nodes:
    for p, size in listing[n].items():
        if size > files.get(p, (None, -1))[1]:
            files[p] = (n, size)
print(f"{len(files)} files, {sum(s for _, s in files.values()) / 1e9:.0f} GB -> {DST}", flush=True)


def upload(item):
    path, (node, size) = item
    for attempt in range(3):
        cat = subprocess.Popen([*SSH, node, f"cat {SRC}/{path}"], stdout=subprocess.PIPE)
        up = subprocess.run(["aws", "s3", "cp", "-", f"{DST}/{path}", "--expected-size", str(size), "--only-show-errors"], stdin=cat.stdout)
        cat.stdout.close()
        if cat.wait() == 0 and up.returncode == 0:
            return path, size
    raise RuntimeError(f"upload failed: {path}")


with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    for done, (path, size) in enumerate(pool.map(upload, sorted(files.items(), key=lambda kv: -kv[1][1])), 1):
        print(f"  {done}/{len(files)} {path} {size / 1e9:.1f} GB", flush=True)

# Verify: every object present with the right size.
out = subprocess.run(["aws", "s3", "ls", "--recursive", DST + "/"], capture_output=True, text=True, check=True).stdout
remote = {line.split(None, 3)[3].split(f"global_step_{STEP}/", 1)[1]: int(line.split()[2]) for line in out.splitlines() if line.strip()}
bad = [p for p, (_, s) in files.items() if remote.get(p) != s]
print("VERIFIED" if not bad else f"MISMATCH: {bad}", f"{len(remote)} objects")
