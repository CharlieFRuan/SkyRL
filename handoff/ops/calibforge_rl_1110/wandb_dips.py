"""Per-step reward next to the error / timeout / length counters of the same step.  python wandb_dips.py <run id>"""
import sys, statistics as st
import wandb
rid = sys.argv[1]
run = wandb.Api(timeout=60).run(f"sky-posttraining-uc-berkeley/skycap-calibforge-fully-async/{rid}")
rows = [r for r in run.scan_history() if r.get("reward/avg_raw_reward") is not None]
keys = sorted({k for r in rows for k in r})
if "--keys" in sys.argv:
    print([k for k in keys if k.startswith("generate/") or k.startswith("reward/")]); raise SystemExit
cols = ["reward/avg_raw_reward", "reward/avg_pass_at_16", "generate/skycap/num_error_trajectories", "generate/skycap/num_timeout_trajectories",
        "generate/skycap/num_masked_instances", "generate/skycap/trajectories_context_length_exceeded", "generate/harbor/num_failed_attempts",
        "generate/harbor/num_failed_attempts/AgentTimeoutError", "generate/harbor/num_failed_attempts/VerifierTimeoutError",
        "generate/harbor/num_failed_attempts/NonZeroAgentExitCodeError", "generate/avg_num_tokens", "generate/skycap/avg_num_turns",
        "generate/harbor/agent_execution_time_mean", "generate/harbor/environment_setup_time_mean", "generate/harbor/verifier_time_mean",
        "generate/num_groups_dropped", "generate/batch_num_seq"]
cols = [c for c in cols if c in keys]
print(" | ".join(c.split("/")[-1][:20] for c in ["step"] + cols))
for r in rows:
    print(r["_step"], " | ".join((("%.3g" % r[c]) if isinstance(r.get(c), (int, float)) else "-") for c in cols))
def corr(a, b):
    ma, mb = st.mean(a), st.mean(b); sa, sb = st.pstdev(a), st.pstdev(b)
    return 0 if not sa or not sb else sum((x - ma) * (y - mb) for x, y in zip(a, b)) / len(a) / sa / sb
rw = [r["reward/avg_raw_reward"] for r in rows]
print("reward mean %.3f sd %.3f" % (st.mean(rw), st.pstdev(rw)))
for c in cols[1:]:
    v = [r.get(c) for r in rows]
    v = [x if isinstance(x, (int, float)) else 0 for x in v]
    if True: print("corr(reward, %s) = %.2f" % (c.split("/")[-1], corr(rw, v)))
