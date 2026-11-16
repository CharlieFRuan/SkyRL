"""Tasks consumed so far in the current epoch, from W&B (32 kept + dropped per step).  python wandb_consumed.py <run id> [<run id> ...]"""
import sys, wandb
api = wandb.Api(timeout=60); total = 0; steps = 0
for rid in sys.argv[1:]:
    run = api.run(f"sky-posttraining-uc-berkeley/skycap-calibforge-fully-async/{rid}")
    rows = [r for r in run.scan_history(keys=["async/num_groups_dropped", "_step"]) if r.get("async/num_groups_dropped") is not None]
    d = sum(r["async/num_groups_dropped"] for r in rows)
    print(rid, "steps", len(rows), "first/last", rows[0]["_step"], rows[-1]["_step"], "dropped", int(d), "consumed", int(d) + 32 * len(rows))
    total += int(d) + 32 * len(rows); steps += len(rows)
print("total consumed", total, "over", steps, "steps =", round(total / steps, 1), "per step")
