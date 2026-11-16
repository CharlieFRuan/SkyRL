"""_ShortCreate caps the creation wait and passes everything else through (no network).
    cd $D/SkyRL && python ops/.../test_create_timeout.py   (needs harbor importable)"""
import asyncio, os, sys
os.environ["SKYCAP_CREATE_TIMEOUT_SEC"] = "120"
sys.path.insert(0, os.environ.get("SKYRL_DIR", "/mnt/local_storage/charlie/SkyRL_head_copy"))
from examples.train_integrations.harbor_skycap import daytona as mod

seen = []


class Fake:
    marker = 7

    async def create(self, params=None, timeout=60, **kw):
        seen.append((params, timeout, kw))
        return "sandbox"


async def main():
    assert mod.CREATE_TIMEOUT_SEC == 120
    c = mod._ShortCreate(Fake(), 120)
    assert await c.create(params="p", timeout=1800) == "sandbox" and seen[-1] == ("p", 120, {})
    assert await c.create(params="p", timeout=40) == "sandbox" and seen[-1][1] == 40
    assert await c.create(params="p", timeout=0) == "sandbox" and seen[-1][1] == 120  # 0 = no limit in the SDK
    assert c.marker == 7
    print("ok")


asyncio.run(main())
