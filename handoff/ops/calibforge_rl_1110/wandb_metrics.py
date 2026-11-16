"""Print a run's key per-step metrics from W&B.  python wandb_metrics.py <run id> [project]"""
import sys
import wandb
rid = sys.argv[1]; proj = sys.argv[2] if len(sys.argv) > 2 else "skycap-calibforge-fully-async"
run = wandb.Api(timeout=60).run(f"sky-posttraining-uc-berkeley/{proj}/{rid}")
rows = list(run.scan_history())
keys = sorted({k for r in rows for k in r})
want = [k for k in keys if any(s in k for s in ("reward/avg_raw_reward", "reward/avg_pass_at", "policy_entropy", "logprobs_diff_mean",
        "generate/avg_num_tokens", "avg_response_length", "eval/all/avg_score", "eval/all/pass_at", "grad_norm", "timing/step",
        "staleness_mean", "staleness_max", "num_groups_dropped", "keep_rate", "zero_var"))][:16]
print("keys:", want)
for r in rows:
    if r.get("_step") is None: continue
    vals = {k.split("/")[-1][:22]: (round(r[k], 4) if isinstance(r.get(k), float) else r.get(k)) for k in want if r.get(k) is not None}
    # Off-policy drift: fraction of tokens DPPO masks, and the mean |rollout - trainer| logprob gap (unrounded).
    for k, short in (("policy/loss_metrics/clip_ratio", "clip_ratio"), ("policy/minibatch_rollout_logprobs_abs_diff_mean", "logprob_diff")):
        if isinstance(r.get(k), float): vals[short] = float("%.3g" % r[k])
    if vals: print(r["_step"], vals)
