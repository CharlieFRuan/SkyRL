"""Count the sandboxes visible to the Dune key by whose they are (labels only; deletes nothing).
    set -a; . $D/dune_keys.env; set +a; python key_census.py"""
import asyncio, collections
from daytona import AsyncDaytona


async def main():
    async with AsyncDaytona() as d:
        c = collections.Counter()
        async for s in d.list():
            l = s.labels or {}
            who = ("ours:" + (l.get("run") or "?")[-28:]) if l.get("owner") == "charlieruan" else "NOT OURS"
            c[(who, str(s.state).split(".")[-1])] += 1
        print("sandboxes on the key:", dict(c) or "none")


asyncio.run(main())
