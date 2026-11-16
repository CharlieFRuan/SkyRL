"""Download one URL with N parallel HTTP range requests (egress here is throttled per connection).
    python3 rdl.py <url> <out> [connections=64] [auth header value]"""
import concurrent.futures as cf, os, subprocess, sys
url, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 64
auth = ["-H", f"Authorization: {os.environ['RDL_AUTH']}"] if os.environ.get("RDL_AUTH") else []
eff = subprocess.run(["curl", "-sIL", "-m", "60", *auth, "-o", "/dev/null", "-w", "%{url_effective}", url], capture_output=True, text=True).stdout
hdr = subprocess.run(["curl", "-sI", "-m", "60", eff], capture_output=True, text=True).stdout
size = [int(l.split(":")[1]) for l in hdr.lower().splitlines() if l.startswith("content-length:")][-1]
chunk = max(1 << 20, -(-size // (n * 4)))
parts = [(a, min(a + chunk, size) - 1) for a in range(0, size, chunk)]
def get(r):
    a, b = r; p = f"{out}.part{a}"
    for _ in range(10):
        subprocess.run(["curl", "-sL", "-m", "1200", "-r", f"{a}-{b}", "-o", p, eff])
        if os.path.exists(p) and os.path.getsize(p) == b - a + 1: return p
    raise RuntimeError(f"range {a}-{b} failed")
with cf.ThreadPoolExecutor(n) as ex: ps = list(ex.map(get, parts))
with open(out, "wb") as f:
    for p in ps: f.write(open(p, "rb").read()); os.remove(p)
print("RDL_DONE", out, size, os.path.getsize(out))
