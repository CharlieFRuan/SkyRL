#!/usr/bin/env bash
# Runs ON the driver: TB2.1's two Debian-bullseye tasks (qemu-startup, qemu-alpine-ssh) can never pass since bullseye's
# end of life: deb.debian.org 404s the curl package their tests/test.sh installs (the files moved to
# archive.debian.org). One line in our eval copy's test.sh points the bullseye-security source at the archive.
# Harbor reads the task dir when a trial starts, so it applies from the next eval without a restart. Idempotent.
# User, 11-10: "Yes add the quick fix without restarting please."  Tested: test_qemu_apt_fix.py.
set -euo pipefail
D=/mnt/local_storage/charlie
for t in qemu-startup qemu-alpine-ssh; do
  f=$D/data/tbench21_4cpu/$t/tests/test.sh
  grep -q "archive.debian.org" "$f" && { echo "$t: already fixed"; continue; }
  cp -n "$f" "$f.orig"
  python3 - "$f" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
fix = ("# (our eval copy) Debian bullseye is past end of life: its security packages moved to archive.debian.org.\n"
       "sed -i 's|http://deb.debian.org/debian-security|http://archive.debian.org/debian-security|g' "
       "/etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null\n")
assert s.count("apt-get update\n") == 1, p
open(p, "w").write(s.replace("apt-get update\n", fix + "apt-get update\n"))
PY
  echo "$t: fixed"; sed -n 1,9p "$f"
done
