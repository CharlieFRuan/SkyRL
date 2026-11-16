"""Score an eval job together with its fill-in jobs: per task, the first K trials (by start time) that did not fail on
the sandbox provider, with fill trials (<task>__f<i>) folded into <task>. Prints the mean over those trials, the
per-pass means (pass i = each task's i-th clean trial; a missing one scores 0), pass@1 / pass@K, and how many tasks
still lack K clean trials.

    python3 score_evals.py <trials dir> [<fill trials dir> ...] [--k 3] [--n-tasks 500]
"""

import argparse
import collections
import glob
import json
import os
import math

INFRA = {
    "DaytonaForbiddenError",
    "DaytonaError",
    "DaytonaAuthenticationError",
    "DaytonaServiceUnavailableError",
    "DaytonaBadRequestError",
    "DaytonaProcessNotFoundError",
    "DaytonaTimeoutError",
    "DaytonaInternalServerError",
    "SandboxBuildFailedError",
    "AgentSetupTimeoutError",
}


def infra_failure(result_path: str, exc) -> bool:
    """A failure of the sandbox provider, not the model: INFRA, an agent whose install failed (no agent log), or an
    agent that exited because skycap had already closed its trajectory (idle past skycap.ttl while its sandbox was
    being built)."""
    if exc in INFRA:
        return True
    if exc == "NonZeroAgentExitCodeError":
        log = os.path.join(os.path.dirname(result_path), "agent", "mini-swe-agent.txt")
        if not os.path.exists(log):
            # The agent's install command failed (network in the sandbox): the agent never ran.
            return True
        if "trajectory_closed" in open(log, errors="replace").read()[-20000:]:
            return True
    return False
parser = argparse.ArgumentParser()
parser.add_argument("dirs", nargs="+")
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--n-tasks", type=int, default=None)
parser.add_argument("--rerun-timeouts", action="store_true",
                    help="exclude agent timeouts in every dir but the last (the re-run, whose results count as they are)")
args = parser.parse_args()

by_task = collections.defaultdict(list)
infra = collections.Counter()
other = collections.Counter()
for i, d in enumerate(args.dirs):
    for f in glob.glob(f"{d}/*/result.json"):
        r = json.load(open(f))
        task = os.path.basename(os.path.dirname(f)).rsplit("__", 1)[0].split("__f")[0]
        exc = (r.get("exception_info") or {}).get("exception_type")
        if infra_failure(f, exc) or (args.rerun_timeouts and i < len(args.dirs) - 1 and exc == "AgentTimeoutError"):
            infra[exc] += 1
            continue
        if exc:
            other[exc] += 1
        reward = float(((r.get("verifier_result") or {}).get("rewards") or {}).get("reward", 0) or 0)
        by_task[task].append((r.get("started_at") or "", reward))
k = args.k
n_tasks = args.n_tasks or len(by_task)
kept = {t: [rw for _, rw in sorted(v)[:k]] for t, v in by_task.items()}
short = {t: len(v) for t, v in kept.items() if len(v) < k}
missing_tasks = n_tasks - len(kept)
trials = [rw for v in kept.values() for rw in v]
passes = [sum(v[i] for v in kept.values() if len(v) > i) / n_tasks for i in range(k)]


def pass_at(n, c, kk):
    return 1.0 if n - c < kk else 1.0 - math.comb(n - c, kk) / math.comb(n, kk)


full = {t: v for t, v in kept.items() if len(v) == k}
pak = sum(pass_at(k, sum(1 for rw in v if rw > 0), k) for v in full.values()) / max(1, len(full))
print(f"{args.dirs[0]}")
print(
    f"  clean trials {len(trials)} / {n_tasks * k}; tasks with < {k} clean: {len(short) + missing_tasks}"
    f" (none: {missing_tasks})"
)
print(f"  mean over clean trials {sum(trials) / max(1, len(trials)):.4f}")
print(f"  per-pass (missing = 0) {[round(p, 4) for p in passes]}  avg {sum(passes) / k:.4f}")
print(f"  pass@{k} over tasks with {k} clean trials ({len(full)}): {pak:.4f}")
print(f"  infra failures excluded {dict(infra.most_common())}; other exceptions {dict(other.most_common(5))}")
