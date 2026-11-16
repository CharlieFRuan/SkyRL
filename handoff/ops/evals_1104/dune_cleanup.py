"""Delete our leaked sandboxes (label owner=charlieruan) of the given runs on the provider at DAYTONA_API_URL.

    set -a; . $D/dune_keys.env; set +a; python dune_cleanup.py <run>[@<UTC ISO cutoff>] ...

A run without a cutoff loses all its sandboxes (use only for runs that have finished or been stopped); with one, only
sandboxes created before it (a relaunched run reusing the name keeps its live sandboxes).
"""
import asyncio, collections, sys
from daytona import AsyncDaytona, ListSandboxesQuery

rules = dict((a.split("@") + [""])[:2] for a in sys.argv[1:])


async def main():
    async with AsyncDaytona() as d:
        sbs = [s async for s in d.list(ListSandboxesQuery(labels={"owner": "charlieruan"}))]
        print("ours:", len(sbs), dict(collections.Counter((s.labels or {}).get("run", "?") for s in sbs)), flush=True)
        doomed = [
            s for s in sbs
            if (s.labels or {}).get("run") in rules and str(s.created_at or "") < (rules[s.labels["run"]] or "9999")
        ]
        print("deleting", len(doomed), dict(collections.Counter(s.labels["run"] for s in doomed)), flush=True)
        sem, errs = asyncio.Semaphore(32), []

        async def rm(s):
            async with sem:
                try:
                    await d.delete(s)
                except Exception as e:  # noqa: BLE001
                    errs.append(e)

        await asyncio.gather(*(rm(s) for s in doomed))
        print("errors", len(errs))


asyncio.run(main())
