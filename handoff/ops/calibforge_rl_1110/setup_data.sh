#!/usr/bin/env bash
# Runs ON a worker: datasets for the 2026-11-10 RL run. Idempotent.
#   $D/data/calibforge_v2_train48 : CalibForge v2 minus the 4 tasks whose verifier times out
#                                   (ops/calibforge_rl_1106/dropped_tasks.txt), every sandbox 4 vCPU / 8 GB / 20 GB
#                                   disk (user, 11-10: the sandbox should not be the limit; Dune allows 8 / 16 / 64).
#   $D/data/tbench21_own          : Terminal-Bench 2.1 (commit 7131e43) with each task's own sandbox size. The agent
#                                   timeout comes from the run (override_timeout_sec=3600).
set -euo pipefail
D=/mnt/local_storage/charlie
cd $D/data
if [ ! -f calibforge_v2_train48/.done ]; then
  rm -rf calibforge_v2_train48 && mkdir calibforge_v2_train48
  for t in cf_v2_x/tasks/*/; do
    n=$(basename "$t"); grep -qx "$n" $D/ops/calibforge_rl_1106/dropped_tasks.txt || cp -a "$t" calibforge_v2_train48/
  done
  python3 - <<'PY'
import glob, re
for f in glob.glob("/mnt/local_storage/charlie/data/calibforge_v2_train48/*/task.toml"):
    s = open(f).read()
    s, a = re.subn(r"(?m)^cpus\s*=.*$", "cpus = 4", s)
    s, b = re.subn(r"(?m)^memory\s*=.*$", 'memory = "8G"', s)
    assert a >= 1 and b >= 1, (f, a, b)   # one task repeats the keys under [agent]
    # 20 GB: six trials of the 3,600 s inference pass filled a 10 GB disk (model-training tasks installing torch).
    s, c = re.subn(r"(?m)^storage\s*=.*$", 'storage = "20G"', s)
    if c == 0:  # one task names no disk size
        s = re.sub(r"(?m)^(memory = \"8G\")$", '\\1\nstorage = "20G"', s, count=1)
    open(f, "w").write(s)
PY
  touch calibforge_v2_train48/.done
fi
if [ ! -f tbench21_own/.done ]; then
  rm -rf tbench21_own
  [ -d tb21src ] || { git clone -q https://github.com/harbor-framework/terminal-bench-2-1.git tb21src && (cd tb21src && git checkout -q 7131e43); }
  (cd tb21src && git checkout -q 7131e43 -- tasks) && cp -r tb21src/tasks tbench21_own && touch tbench21_own/.done
fi
# TB2.1 at 4 vCPU / 8 GB / 20 GB (user, 11-10 evening: "I didn't know we are doing 1 CPU. Let's do 4 vCPU"; 8 GB as the
# 11-04 reproduction of the published score, 20 GB because verifiers installing torch filled the tasks' own 10 GB).
if [ ! -f tbench21_4cpu/.done ]; then
  rm -rf tbench21_4cpu && cp -r tbench21_own tbench21_4cpu && rm -f tbench21_4cpu/.done
  python3 - <<'PY'
import glob, re
for f in glob.glob("/mnt/local_storage/charlie/data/tbench21_4cpu/*/task.toml"):
    s = open(f).read()
    s, a = re.subn(r"(?m)^cpus\s*=.*$", "cpus = 4", s)
    s, b = re.subn(r"(?m)^memory_mb\s*=.*$", "memory_mb = 8192", s)
    s, c = re.subn(r"(?m)^storage_mb\s*=.*$", "storage_mb = 20480", s)
    assert a == 1 and b == 1 and c == 1, (f, a, b, c)
    open(f, "w").write(s)
PY
  touch tbench21_4cpu/.done
fi
s() { grep -h "$1" $2/*/task.toml | sed 's/ //g' | sort | uniq -c | sort -rn | head -8 | awk '{printf "%s x%s, ", $2, $1}'; }
echo "TRAIN $(find calibforge_v2_train48 -mindepth 1 -maxdepth 1 -type d | wc -l): $(s '^cpus' calibforge_v2_train48) $(s '^memory' calibforge_v2_train48) $(s '^storage' calibforge_v2_train48)"
echo "EVAL $(find tbench21_own -mindepth 1 -maxdepth 1 -type d | wc -l): $(s '^cpus' tbench21_own) $(s '^memory' tbench21_own) $(s '^storage' tbench21_own)"
echo "EVAL4 $(find tbench21_4cpu -mindepth 1 -maxdepth 1 -type d | wc -l): $(s '^cpus' tbench21_4cpu) $(s '^memory_mb' tbench21_4cpu) $(s '^storage_mb' tbench21_4cpu)"
