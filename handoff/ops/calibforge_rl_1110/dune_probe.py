"""Basic probe of the Dune sandbox API (creates and deletes its own two sandboxes only; lists by label).
    set -a; . $D/dune_keys.env; set +a; python dune_probe.py <task dir> <task dir>"""
import asyncio, collections, re, sys, time
from daytona import AsyncDaytona, CreateSandboxFromImageParams, Image, ListSandboxesQuery, Resources

LABELS = {"owner": "charlieruan", "run": "dune-probe-1115"}


async def one(d, image, tag):
    t0 = time.time()
    try:
        sb = await d.create(
            CreateSandboxFromImageParams(image=Image.base(image), labels=LABELS, ephemeral=True, auto_stop_interval=0,
                                         resources=Resources(cpu=4, memory=8, disk=20)),
            timeout=600,
        )
    except Exception as e:  # noqa: BLE001
        print(f"{tag}: CREATE FAILED after {time.time() - t0:.1f} s: {type(e).__name__}: {str(e)[:300]}", flush=True)
        return
    t1 = time.time()
    try:
        r = await sb.process.exec("nproc; free -g | sed -n 2p | tr -s ' ' | cut -d' ' -f2; echo probe-ok", timeout=60)
        out = " / ".join((r.result or "").split())
    except Exception as e:  # noqa: BLE001
        out = f"EXEC FAILED {type(e).__name__}: {str(e)[:200]}"
    t2 = time.time()
    try:
        await d.delete(sb)
        dl = "deleted"
    except Exception as e:  # noqa: BLE001
        dl = f"DELETE FAILED {type(e).__name__}: {str(e)[:200]}"
    print(f"{tag}: create {t1 - t0:.1f} s | exec {t2 - t1:.1f} s -> {out} | {dl} in {time.time() - t2:.1f} s", flush=True)


async def census(d, when):
    t0 = time.time(); c = collections.Counter()
    async for s in d.list():
        l = s.labels or {}
        who = ("ours:" + (l.get("run") or "?")[-24:]) if l.get("owner") == "charlieruan" else "NOT OURS"
        c[(who, str(s.state).split(".")[-1])] += 1
    print(f"{when}: list ok in {time.time() - t0:.1f} s: {dict(c)}", flush=True)


async def main():
    images = []
    for t in sys.argv[1:]:
        m = re.search(r'docker_image\s*=\s*"([^"]+)"', open(t.rstrip("/") + "/task.toml").read())
        images.append(m.group(1))
    async with AsyncDaytona() as d:
        try:
            await census(d, "before")
        except Exception as e:  # noqa: BLE001
            print(f"LIST FAILED: {type(e).__name__}: {str(e)[:300]}"); return
        await asyncio.gather(*(one(d, im, f"image {i + 1}") for i, im in enumerate(images)))
        await asyncio.sleep(5)
        await census(d, "after")
        left = [s async for s in d.list(ListSandboxesQuery(labels=LABELS))]
        print("probe sandboxes left:", len(left))


asyncio.run(main())
