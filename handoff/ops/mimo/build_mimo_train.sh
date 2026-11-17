#!/usr/bin/env bash
# The MiMo training set: every task in mimo-data/mimo-code minus every exclusion list in mimo-data/stats
# (excluded_tasks.tsv, residue_excluded.tsv, leak_flagged.txt, excluded_infra.txt, broken_nop.txt; heavy_ids.txt is
# sizing only, not an exclusion). Writes $D/data/mimo-train/ as symlinks. Run on every worker (paths must match).
set -euo pipefail
D=/mnt/local_storage/charlie
SRC=$D/data/mimo-data/mimo-code
STATS=$D/data/mimo-data/stats
OUT=$D/data/mimo-train
EXCL=$(mktemp)
for f in excluded_tasks.tsv residue_excluded.tsv leak_flagged.txt excluded_infra.txt broken_nop.txt; do
  grep -v '^#' "$STATS/$f" | grep -oE '^format-code-task-[0-9]+' || true
done | sort -u > "$EXCL"
rm -rf "$OUT"; mkdir -p "$OUT"
n=0
for t in $(ls "$SRC"); do
  grep -qx "$t" "$EXCL" && continue
  ln -s "$SRC/$t" "$OUT/$t"; n=$((n + 1))
done
echo "MIMO_TRAIN_DONE tasks=$n excluded=$(wc -l < "$EXCL") total=$(ls "$SRC" | wc -l)"
rm -f "$EXCL"
