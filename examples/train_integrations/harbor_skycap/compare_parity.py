"""Compare stock-Harbor runs with skycap runs of the same agent on the same tasks (parity check).

usage: python compare_parity.py [--split K] --arm baseline <trials_dir> [...] --arm skycap <trials_dir> [...]

Each argument after an --arm name is one run: a directory of Harbor trial dirs (each with result.json and
agent/mini-swe-agent.trajectory.json). With --split K, a single directory holding K attempts per task (harbor -k K,
or eval_n_samples_per_prompt=K) is split into K runs by each task's attempt order. Prints, per arm: per-run mean reward, the mean and spread over runs, a
95% CI over task-level means (tasks resampled), exit statuses, exceptions, turn counts and per-call token sizes;
then the difference between the arms with a paired (per-task) bootstrap CI.
"""

import argparse
import collections
import json
import random
import statistics as st
from pathlib import Path


def load_run(trials_dir: Path) -> dict:
    """task -> list of trial records (a run normally has one per task)."""
    out = collections.defaultdict(list)
    for result in trials_dir.glob("*/result.json"):
        r = json.loads(result.read_text())
        exc = (r.get("exception_info") or {}).get("exception_type")
        reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        traj_path = result.parent / "agent" / "mini-swe-agent.trajectory.json"
        turns, exit_status, prompt_tokens, completion_tokens = None, None, [], []
        if traj_path.exists():
            t = json.loads(traj_path.read_text())
            exit_status = (t.get("info") or {}).get("exit_status")
            msgs = [m for m in t.get("messages", []) if m.get("role") == "assistant"]
            turns = len(msgs)
            for m in msgs:
                usage = ((m.get("extra") or {}).get("response") or {}).get("usage") or {}
                if usage.get("prompt_tokens"):
                    prompt_tokens.append(usage["prompt_tokens"])
                if usage.get("completion_tokens"):
                    completion_tokens.append(usage["completion_tokens"])
        task = result.parent.name.rsplit("__", 1)[0]
        out[task].append(
            dict(
                started=r.get("started_at") or "",
                reward=None if reward is None else float(reward),
                exception=exc,
                exit_status=exit_status,
                turns=turns,
                max_prompt=max(prompt_tokens) if prompt_tokens else None,
                completion=sum(completion_tokens) if completion_tokens else None,
            )
        )
    return out


def split_run(run: dict, k: int) -> list:
    """K runs out of one directory with K attempts per task: the i-th attempt (by start time) of each task."""
    runs = [collections.defaultdict(list) for _ in range(k)]
    for task, xs in run.items():
        for i, x in enumerate(sorted(xs, key=lambda x: x["started"])[:k]):
            runs[i][task].append(x)
    return runs


def run_mean(run: dict) -> float:
    # A trial with no verifier reward scores 0, as Harbor and the skycap eval both count it.
    vals = [(x["reward"] or 0.0) for xs in run.values() for x in xs]
    return sum(vals) / len(vals) if vals else float("nan")


def task_means(runs: list) -> dict:
    acc = collections.defaultdict(list)
    for run in runs:
        for task, xs in run.items():
            acc[task] += [(x["reward"] or 0.0) for x in xs]
    return {t: sum(v) / len(v) for t, v in acc.items()}


def boot_ci(values: list, n: int = 10000, seed: int = 0) -> tuple:
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def describe(name: str, runs: list) -> dict:
    print(f"\n== {name}: {len(runs)} run(s)")
    for i, run in enumerate(runs):
        n = sum(len(v) for v in run.values())
        print(f"  run {i}: {len(run)} tasks, {n} trials, mean reward {run_mean(run):.3f}")
    means = [run_mean(r) for r in runs]
    if len(means) > 1:
        print(f"  over runs: mean {st.mean(means):.3f}, sd {st.stdev(means):.3f}")
    tm = task_means(runs)
    lo, hi = boot_ci(list(tm.values()))
    print(f"  task-level mean {st.mean(tm.values()):.3f} (95% CI {lo:.3f}-{hi:.3f}, tasks resampled)")
    flat = [x for r in runs for xs in r.values() for x in xs]
    print(f"  exit statuses: {dict(collections.Counter(x['exit_status'] for x in flat).most_common())}")
    print(f"  exceptions:    {dict(collections.Counter(x['exception'] for x in flat).most_common())}")
    for k in ("turns", "max_prompt", "completion"):
        v = sorted(x[k] for x in flat if x[k] is not None)
        if v:
            print(f"  {k:11s} p50 {v[len(v) // 2]}  p90 {v[int(0.9 * (len(v) - 1))]}  max {v[-1]}  (n={len(v)})")
    return tm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", action="append", required=True, metavar=("NAME", "TRIALS_DIR"))
    ap.add_argument("--split", type=int, default=0, help="split each directory into this many runs (K attempts per task)")
    args = ap.parse_args()
    arms = {}
    for a in args.arm:
        runs = [load_run(Path(p)) for p in a[1:]]
        arms[a[0]] = [r for run in runs for r in split_run(run, args.split)] if args.split else runs
    tms = {name: describe(name, runs) for name, runs in arms.items()}
    if len(tms) == 2:
        (a, ta), (b, tb) = tms.items()
        common = sorted(set(ta) & set(tb))
        diffs = [tb[t] - ta[t] for t in common]
        lo, hi = boot_ci(diffs)
        print(f"\n== {b} - {a} on {len(common)} common tasks: {st.mean(diffs):+.3f} (paired 95% CI {lo:+.3f} to {hi:+.3f})")
        print("   parity holds if the CI contains 0 and the gap is small relative to run-to-run spread")


if __name__ == "__main__":
    main()
