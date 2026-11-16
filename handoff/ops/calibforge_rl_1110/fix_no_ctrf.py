"""Runs ON the driver. 17 CalibForge tasks run pytest without the ctrf plugin, so their verifier never writes
/logs/verifier/ctrf.json. SKYCAP_VERIFIER_RETRIES treats "reward 0 and no ctrf.json" as "the tests never ran": every
failed attempt on these tasks was re-verified 4 times, raised VerifierTestsDidNotRunError, retried and then masked its
group, so the tasks only cost compute and never trained (seen 11-11 02:35: 32 trials of one task in an hour).
This adds one line to each such test.sh, in the branch that runs after pytest failed, writing a small ctrf.json marker.
The reward logic is untouched. Harbor reads the test script at verification time, so it applies without a restart.
Idempotent; originals kept as test.sh.orig."""
import glob, os, re, subprocess

D = "/mnt/local_storage/charlie/data/calibforge_v2_train48"
MARK = "echo '{\"results\": {\"tool\": {\"name\": \"none\"}, \"note\": \"tests ran; this test.sh has no ctrf plugin\"}}' > /logs/verifier/ctrf.json"
done = skipped = 0
for f in sorted(glob.glob(D + "/*/tests/test.sh")):
    s = open(f).read()
    # Has the pytest run a --ctrf option (some scripts only pip-install the plugin), or our marker already?
    if "--ctrf" in s or "this test.sh has no ctrf plugin" in s:
        continue
    lines = s.split("\n")
    idx = [i for i, l in enumerate(lines) if re.fullmatch(r"\s*echo 0 > /logs/verifier/reward\.txt\s*", l)]
    one = [i for i, l in enumerate(lines) if re.fullmatch(r"\s*echo 1 > /logs/verifier/reward\.txt\s*", l)]
    # The post-test branch: the last "echo 0", which must come right after the (single) "echo 1" + else.
    if not idx or len(one) != 1 or not (one[0] < idx[-1] <= one[0] + 3):
        print("SKIP (unexpected shape):", f); skipped += 1; continue
    i = idx[-1]; indent = re.match(r"\s*", lines[i]).group(0)
    lines.insert(i + 1, indent + MARK)
    if not os.path.exists(f + ".orig"): open(f + ".orig", "w").write(s)
    open(f, "w").write("\n".join(lines))
    assert subprocess.run(["bash", "-n", f]).returncode == 0, f
    done += 1
print("patched", done, "skipped", skipped, "| still writing no ctrf.json:", sum(("--ctrf" not in open(f).read() and "no ctrf plugin" not in open(f).read()) for f in glob.glob(D + "/*/tests/test.sh")))
