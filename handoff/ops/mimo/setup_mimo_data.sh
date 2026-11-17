#!/usr/bin/env bash
# Runs ON a worker: the MiMo Harbor tasks (private CharlieFRuan/mimo-v2.6-harbor-tasks), resized for Dune;
# mimo-train-avail (ops/mimo_train_avail.txt minus format-code-task-001975); SWE-bench-100. Idempotent.
set -euo pipefail
D=/mnt/local_storage/charlie; mkdir -p $D/{logs,runs,data,tmp}
set -a; source $D/keys.env; set +a
export HF_TOKEN=$HF_API_KEY HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 XDG_CACHE_HOME=$D/.cache UV_CACHE_DIR=/home/ray/.cache/uv
HF="uvx --from huggingface_hub[cli]==0.36.0 hf"
if [ ! -f $D/data/mimo-data/.done ]; then
  rm -rf $D/data/mimo-src
  # hf's xet transfer stalled on the 14 MB tarball (2026-11-06): plain HTTPS.
  mkdir -p $D/data/mimo-data
  curl -sfL --retry 5 -m 600 -H "Authorization: Bearer $HF_TOKEN" -o $D/data/mimo-data/mimo-code.tar.gz \
    https://huggingface.co/datasets/CharlieFRuan/mimo-v2.6-harbor-tasks/resolve/main/mimo-code.tar.gz
  tar -xzf $D/data/mimo-data/mimo-code.tar.gz -C $D/data/mimo-data
  python3 - <<'PY'
import glob, re
n = 0
for f in glob.glob("/mnt/local_storage/charlie/data/mimo-data/mimo-code/*/task.toml"):
    s = open(f).read()
    t = s
    m = re.search(r"^memory_mb = (\d+)", t, re.M)
    if m and int(m.group(1)) > 16384:
        t = t.replace(m.group(0), "memory_mb = 16384")
    # Dune: large MiMo images failed to build on small disks (SWE-bench matplotlib at 10 GB); 30 GB floor, max 64.
    m = re.search(r"^storage_mb = (\d+)", t, re.M)
    if m and int(m.group(1)) < 30720:
        t = t.replace(m.group(0), "storage_mb = 30720")
    if t != s:
        open(f + ".orig", "w").write(s); open(f, "w").write(t); n += 1
print("task.toml resized for Dune (memory <= 16 GiB, disk >= 30 GB):", n)
PY
  touch $D/data/mimo-data/.done
fi
[ -f $D/data/swebench100/.done ] || { $HF download DCAgent2/swebench-verified-random-100-folders --repo-type dataset --revision 0c553bd6 --local-dir $D/data/swebench100 && touch $D/data/swebench100/.done; }
OUT=$D/data/mimo-train-avail; rm -rf $OUT; mkdir -p $OUT
for t in $(cat $D/ops/mimo_train_avail.txt); do
  [ "$t" = format-code-task-001975 ] && continue  # verifier always times out; stalls the async run (RUN_NOTES)
  [ -d $D/data/mimo-data/mimo-code/$t ] && ln -s $D/data/mimo-data/mimo-code/$t $OUT/$t
done
echo "MIMO_DATA_DONE train=$(ls $OUT | wc -l) swe100=$(ls -d $D/data/swebench100/*/ | wc -l)"
