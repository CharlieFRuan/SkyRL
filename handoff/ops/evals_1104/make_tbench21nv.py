"""$D/data/tbench21nv: Terminal-Bench 2.1 with every agent timeout raised to max(7200 s, the task's own), as NVIDIA's
Lightning recipe (timeout_strategy max, run_timeout 7200) and Artificial Analysis run it. Everything else unchanged."""
import re, shutil, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst, symlinks=True)
n = 0
for toml in sorted(dst.glob("*/task.toml")):
    text = toml.read_text()
    m = re.search(r"(\[agent\]\s*\ntimeout_sec\s*=\s*)([0-9.]+)", text)
    assert m, toml
    new = max(7200.0, float(m.group(2)))
    toml.write_text(text[: m.start(2)] + str(new) + text[m.end(2) :])
    n += 1
print(f"{n} tasks -> {dst}")
