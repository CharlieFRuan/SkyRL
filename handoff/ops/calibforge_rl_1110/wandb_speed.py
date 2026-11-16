"""Per-step timing and concurrency numbers of a run from W&B.  python wandb_speed.py <run id>"""
import sys, statistics as st
import wandb
run = wandb.Api(timeout=60).run(f"sky-posttraining-uc-berkeley/skycap-calibforge-fully-async/{sys.argv[1]}")
rows = [r for r in run.scan_history() if r.get("reward/avg_raw_reward") is not None]
keys = sorted({k for r in rows for k in r})
if "--keys" in sys.argv:
    print([k for k in keys if k.startswith(("timing/", "async/", "generate/group", "generate/intra"))]); raise SystemExit
want = [k for k in keys if k.startswith(("timing/", "async/")) or "group_completion_time" in k or "intra_group" in k
        or k in ("generate/harbor/agent_execution_time_mean", "generate/harbor/agent_execution_time_p90", "generate/harbor/agent_execution_time_max",
                 "generate/harbor/num_attempts", "generate/harbor/num_retried_attempts", "generate/trajectory_time_completion_mean")]
skip = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 2
print(f"steps {len(rows)} (stats over steps after the first {skip})")
for k in want:
    v = [r[k] for r in rows[skip:] if isinstance(r.get(k), (int, float))]
    if v:
        v.sort()
        print(f"{k:58s} mean {st.mean(v):9.1f}  median {st.median(v):9.1f}  min {v[0]:9.1f}  max {v[-1]:9.1f}  n {len(v)}")
