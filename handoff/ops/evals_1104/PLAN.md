# Evals plan (started 2026-11-04 ~21:50 PST; user away ~12 h, autonomous)

## Ask (user, 2026-11-04)
- Port SkyCap + the sandbox relay onto the latest mainline SkyRL (new branch; **push it to the fork
  CharlieFRuan/SkyRL**: user OK'd).
- Evals: {Terminal-Bench 2.1, SWE-bench Verified} x {Qwen3.5-9B, Nemotron-3.5-Lightning}, **through SkyCap**, 3 runs
  each. TB with **Terminus 2**, SWE with **mini-swe-agent 2** (2.4.5).
- Also **no-SkyCap baselines** (stock `harbor run`) x 3, as time allows, to tell a SkyCap issue from a setup issue.
- Reproduce published numbers.
- **Daytona first** (1500 vCPU / 6000 GiB / 15000 GiB total; the user is the only one using it). **In parallel, try to
  make Dune Cloud work.**

## Targets
| Model | SWE-V (mini-swe-agent) | TB 2.1 (Terminus 2) |
|---|---|---|
| Nemotron-3.5-Lightning | 50.7 (card chart, mini 2.4.5) | 24.58 (NVIDIA card table, NeMo Evaluator, Terminus 2, 8 reps); AA 24 |
| Qwen3.5-9B | 60.0 (MiMo paper sec. 7) | 27.0 (MiMo paper), 29.2 (AA) |

## Settings decided
- **TB:** NVIDIA's recipe (user's choice).
  - Terminus 2, every sandbox 4 vCPU / 8 GB.
  - Timeout max(7200 s, the task's own).
  - 262k context, thinking kept across turns, temp 1.0 / top_p 0.95 (Nemotron).
- **Qwen3.5-9B:** follow the MiMo paper's timeouts; the harness can differ.
  - Sampling: the card's thinking + precise-coding preset (temp 0.6, top_p 0.95, top_k 20), unless MiMo says otherwise.
- **SWE (Nemotron):** mini-swe-agent's swebench config + card sampling. In October this gave 50.3 vs 50.7.
- **Dune:** no preview URLs, so the relay can't poll stubs there. The idea is an exec-based relay transport, in
  parallel, at low priority.
  - Per-sandbox max is 2 vCPU / 4 GiB, so TB under NVIDIA's recipe can't run there.
  - Probe report: `ops/dune/DUNE_EVAL.md`.
  - Rotate the Dune key: the probe agent's tool output showed temporary credentials.

## Per-task requirements
- **SWE-V:** all 500 tasks are 1 vCPU / 4G / 10G, agent and verifier 3000 s, image built from a Dockerfile.
- **TB 2.1 (89 tasks):**
  - cpus: 1 (83), 2 (3), 4 (3).
  - memory: 2G (68), 4G (13), 8G (8).
  - disk: 10G for all.
  - Prebuilt images; agent timeouts 600-12000 s.

## Settings research (ops/evals_1104/SETTINGS_RESEARCH.md), decisions 2026-11-04 ~23:30
- **Qwen3.5-9B targets** come from the MiMo-V2.6 report (arXiv 2610.11959), §7.2 Table 6: SWE-V avg@3 60.0, TB2.1
  avg@1 27.0.
  - It gives no harness, timeouts or sampling. Table 7 has mini-swe-agent at 60.6 on SWE-V.
  - SWE: each task's own timeouts (3000 s).
  - TB: Artificial Analysis's setup (Terminus 2, 250 episodes, timeout max(7200, task)).
- **TB, Nemotron (NVIDIA recipe):** Terminus 2, no turn cap, summarization on, temp 1.0 / top_p 0.95, 262k
  model_info, agent timeout max(7200, task), 4 vCPU / 8 GB, thinking replayed as `<think>` tags.
- **TB agent:** `harbor_skycap/terminus_think.py:TerminusThink`. It parses the reply after the last `</think>` and
  replays the whole raw reply. Served raw (SkyCap use_raw_content; the baseline's vLLM runs no reasoning parser).
  - NVIDIA's other Terminus patches are not applied: context-reset fallback, trimming, token counting.
- **Thinking history:**
  - Nemotron: all, in both arms (SkyCap thinking_retention=all; baseline truncate_history_thinking=false).
  - Qwen3.5: what the template keeps, in both arms (SkyCap thinking_retention=null).
- **TB data:** `$D/data/tbench21nv` = TB2.1 with agent timeout raised to max(7200, own); `ops/evals_1104/make_tbench21nv.py`.

## Port notes
- **Branch** `skycap-evals-1104` in `~/hackskyrl/src/SkyRL-upstream`, from mainline f854d6d9. Changes:
  - `sandbox_relay/{stub,manager,environment}.py`, plus relay managers in servers.py.
  - `skycap.sandbox_relay`, `skycap.tokenizer` and `skycap.thinking_retention` config.
  - `run_trials`, `SKYCAP_TRIAL_ATTEMPTS` and `SKYCAP_TRIAL_TIMEOUT_SEC` in the generator.
  - `entrypoints/main_harbor_skycap_eval.py`, `run_eval_skycap.sh`, `terminus_think.py`, mini-swe-agent SWE configs.
  - SKYCAP_* env forwarding.
- **Gotcha: mainline needs `skycap.tokenizer=<HF id>`.** `renderers` picks a renderer by exact HF id. With a local
  path Nemotron got the DefaultRenderer, which refuses thinking_retention=all.
- **Gotcha: never edit a running bash launcher.** It killed the first baseline. The wrappers now run from a snapshot
  in $RUN_DIR.
- **Cluster wrappers:** `ops/evals_1104/skycap_eval.sh` (SkyCap) and `ops/evals_1104/baseline_eval.sh` (stock
  harbor run + vLLM + router + relay, via a JobConfig from `make_job_config.py`).

## Log
- **2026-11-04 ~23:00: the Daytona key in charlie_keys.env is rejected ("Invalid credentials").**
  - The GitHub token is invalid too (401), so the branch is not pushed; it is committed locally (f72a1be7). HF and
    W&B keys work. The user was notified by terminal notification.
  - Pivot: everything runs on **Dune** (`SANDBOX=dune`):
    - SWE-bench uses the exec relay: `skycap.sandbox_relay.transport=exec` and `ExecRelayDaytonaEnvironment`,
      built by a helper agent (report in `ops/dune/DUNE_RELAY.md`).
    - TB is capped at 2 vCPU / 4 GiB (Dune's maximum).
    - Disks are 30 GB: matplotlib images fail to build on Dune's default disk (37 of 41 matplotlib trials failed in
      the first baseline).
- **Smoke tests:**
  - SkyCap + exec relay, SWE on Dune: 4 of 4 trials complete (1 solved), 0 unbridged calls.
  - TB on Dune: Terminus 2 runs fine (fix-git completed).
- **Running (2026-11-05 00:30):**
  - node4 `base-nem-swe-dune` (no SkyCap; 10 GB disks, so the failed matplotlib trials need a rerun with 30 GB).
  - node2 `skycap-nem-swe-dune`, node3 `skycap-qwen-swe-dune` (30 GB).
  - node1: TB smoke finishing.
- **Queue:**
  - node1: `skycap-nem-tb-dune`, then `skycap-qwen-tb-dune`.
  - Then the baselines: base-qwen-swe, base-nem-tb, base-qwen-tb, plus the matplotlib rerun for base-nem-swe.
- **2026-11-05 ~00:30-10:40: stalled for 10 h.** A command of mine waited on a permission prompt, so no check-ins
  ran. The jobs kept running.
  - Dune answered "RBAC: access denied" from ~01:36 PST (create + exec). Every trial created after that failed. By
    10:43 the key worked again.
  - The SWE jobs "finished", with clean-trial means as follows (infra failures excluded: Dune
    Forbidden/Error, build failed, agent setup timeout):
    - base-nem-swe-dune: 1140 clean, 0.506 (card 50.7).
    - skycap-nem-swe-dune: 693 clean, 0.456.
    - skycap-qwen-swe-dune: 580 clean, 0.531 (MiMo 60.0).
  - Fill-in runs re-run only the infra-failed trials so every task gets 3 clean ones (`make_fill.py`, SAMPLES=1,
    trials named `<task>__f<i>`):
    - node2 fill-skycap-nem-swe: 807 trials.
    - node3 fill-skycap-qwen-swe: 920 trials.
    - node4 fill-base-nem-swe: 363 trials.
- Terminus 2 got NVIDIA's patches (context reset etc.; terminus_think.py). Without them, Nemotron's long thinking
  fills the context and Terminus loops in summarization until timeout.
- node1 skycap-nem-tb-dune actually started at 10:43; its launch was the command stuck on the permission prompt.
- **Commits (local only; the GitHub token is invalid):** f72a1be7, ae2ee30d.
- **2026-11-05 ~13:00 check-in.**
  - **The SkyCap Nemotron TB run at 150 concurrent was wrecked by Dune.** 150 creates hit the 600 s timeout, then
    45 "tmux not found" and 21 setup timeouts, with Dune reporting "fork/exec too many open files". Only 34 clean
    trials; moved to runs_dead.
    - All 89 TB images pre-pulled into Dune's cache (`prewarm_dune_images.py`: 89 ok, 5-30 s each).
    - Relaunched skycap-nem-tb-dune (node1) and skycap-qwen-tb-dune (node2) at N_CONCURRENT=60 with
      `generator.rate_limit.trajectories_per_second=0.5`.
  - **Infra classification** (make_fill.py / score_evals.py) also counts as infra:
    - an agent that exited on skycap's `trajectory_closed`. 127 SkyCap trials, all started during the slow-Dune
      hour: skycap.ttl=3600 expired while the sandbox built, before the first call. The launcher now uses
      skycap.ttl=14400.
    - a NonZeroAgentExitCode with no agent log: the install failed (uv/apt download inside the sandbox).
  - **SWE Nemotron (clean trials, original + fill):**
    - SkyCap: 0.474 (1275 clean; 63 tasks with none, needs fill2).
    - Baseline: 0.481 (1473 clean); pass@3 0.615.
    - Parity holds. Card 50.7.
  - **SWE Qwen SkyCap:** 0.527 over 1286 clean trials, but 224 AgentTimeoutErrors (3000 s).
    - The cause is throughput: at 300 concurrent each engine has 30-40 running plus up to 39 waiting, ~1000 tok/s
      per engine, so ~30 tok/s per request and ~130 s per step. Prefix caching works (62-97%).
    - Decision: run Qwen at N_CONCURRENT=100. base-qwen-swe-dune relaunched at 100 (the c300 attempt is in
      runs_dead). The SkyCap Qwen SWE run gets a full rerun at 100 when a node frees.
  - **Queue:**
    1. node3 after the Qwen fill: fill2 for skycap-nem-swe (with the new classification).
    2. Full skycap-qwen-swe rerun at N=100.
    3. base-nem-tb.
    4. base-qwen-tb.
- **2026-11-05 13:30.**
  - **Dune raised per-sandbox limits to 8 vCPU / 16 GiB / 64 GiB.** Both SkyCap TB runs restarted at NVIDIA's
    4 vCPU / 8 GB (TB_CPUS=4 TB_MEMORY_MB=8192); the 2-vCPU attempts are in runs_dead.
  - **The Daytona key works again (user updated it).** The remaining queue stays on Dune so each comparison pair uses
    one provider; the GPU nodes, not sandboxes, are the bottleneck.
  - **Dune leaks.** 1618 of our sandboxes were alive with ~220 in use. Leaked by killed runs, and by finished runs
    (fill-skycap-qwen-swe left 227, about its agent-timeout count). Deleted with `ops/evals_1104/dune_cleanup.py`
    (run[@cutoff]). Run it at every check-in for finished or stopped runs.
  - **node3:** fill2-skycap-nem-swe, 228 trials over 89 tasks (built on node2: worker-to-worker ssh node2->node3
    fails host-key verification; copy via the head).
- **2026-11-05 13:50: SWE Nemotron parity holds.**
  - SkyCap (original + fill + fill2): **0.480** over 1495/1500 clean trials, pass@3 0.606.
  - Baseline: **0.481** over 1473 clean, pass@3 0.615.
  - Card: 50.7.
  - node3: `after_fill2.sh` waits for fill2, then reruns skycap-qwen-swe-dune at N=100. The c300 original and its
    fill-in move to runs_dead as `*-c300`. Log: `logs/skycap-qwen-swe-dune-c100.log`.
- **2026-11-05 14:30. Caveat for the Qwen SWE numbers:** throughput-bound even at 100 concurrent.
  - Baseline vLLM: ~1000 gen tok/s per engine over 21-28 running, so ~40 tok/s per request. One engine hit 99% KV
    with 12 waiting: the prefix-affinity router is imbalanced.
  - 73/472 baseline trials hit the 3000 s agent timeout. The SkyCap rerun has none so far, with 6-13 running per
    engine.
  - Report Qwen with its timeout counts. Its 60.0 (MiMo) was likely measured with faster serving.
- **2026-11-05 14:50.** skycap-nem-tb-dune: 9 DaytonaInternalServerError "fork/exec /bin/bash: too many open files"
  on long trials. Most likely Dune's in-sandbox toolbox leaks file descriptors over a long Terminus session. Counted
  as infra; fill these after the run (make_fill.py with K=3 on tbench21nv).
- **2026-11-05 17:30: skycap-nem-tb-dune done** (4 vCPU / 8 GB, Terminus 2 + NVIDIA patches, N=60).
  - 234/267 clean trials, mean **0.325**; per-pass with missing = 0: 0.315 / 0.292 / 0.247; pass@3 0.508.
  - NVIDIA: 24.58 (8 reps); AA: 24.
  - 33 infra (Dune "too many open files"), 32 agent timeouts (7200+ s).
  - node1: fill-skycap-nem-tb (33 trials). make_fill.py now keys by the task dir (TB's task_name is
    "terminal-bench/<x>") and knows DaytonaTimeoutError / DaytonaInternalServerError.
- **Queue after this fill on node1:** base-nem-tb-dune (4 vCPU / 8 GB, N=60).
- **2026-11-05 18:30: skycap-qwen-tb-dune done** (4 vCPU / 8 GB, Terminus 2, 250 episodes, N=60).
  - 261/267 clean trials, mean **0.253**; per-pass 0.214 / 0.258 / 0.270; pass@3 0.410.
  - MiMo: 27.0; AA: 29.2.
  - 60 agent timeouts: likely the same slow Qwen serving.
  - Missing: 6 infra trials (fill later).
  - node2: **base-nem-tb-dune** launched (no SkyCap, 4/8, N=60).
- **2026-11-05 19:15: base-qwen-swe-dune done** (no SkyCap, N=100).
  - 1500/1500 clean trials, mean **0.507**; per-pass 0.536 / 0.478 / 0.508; pass@3 0.664.
  - **443 agent timeouts** (slow serving: TP1 + sticky hashing + no max_tokens). MiMo: 60.0.
  - Killed 9 orphaned processes from the c300 attempt (8 vllm serve + router; no GPU memory).
  - node4: **base-qwen-tb-dune** launched (4/8, N=60).
- **fill-skycap-nem-tb:** the long tasks fail on Dune again with "too many open files" (12 of 24). Dune's toolbox fd
  leak hits long sessions, so these tasks are unfinishable on Dune. Report as infra-excluded, or rerun on Daytona.
- **2026-11-05 20:15: final.**
  - **skycap-qwen-swe-dune (N=100): 0.511** over 1500/1500 clean trials (per-pass 0.520 / 0.512 / 0.502; pass@3
    0.668; 395 agent timeouts). Baseline 0.507, so parity holds.
  - **skycap-nem-tb-dune + fill: 0.304** over 253/267 clean (pass@3 0.449; 47 Dune fd failures over both, 14 tasks
    unfinishable on Dune).
- **Extra pair (no user answer on the restart, so run in addition, nothing discarded): Qwen SWE "v2"** with better
  serving, both arms.
  - Changes: TP2 x 4 replicas; vllm-router cache_aware (prefix affinity, rebalanced at abs 8 / rel 1.3) in place of
    consistent_hash / rendezvous hash; per-turn max_tokens 81920 (`mini_swe_agent_swebench_qwen35_mt.yaml`).
  - node3 **skycap-qwen-swe-v2**, node1 **base-qwen-swe-v2** (baseline_eval.sh TP_SIZE=2 ROUTER=vllm-router).
- **2026-11-05 21:15. Correction: timeouts do NOT explain the Qwen SWE gap.**
  - Paired on the same 436 tasks: v1 baseline 0.530 (346/1308 timeouts) vs v2 baseline 0.502 (7/438 timeouts).
  - v2 failures: 155 submitted a wrong patch, 31 hit the 250-step limit, 30 other.
  - So Qwen3.5-9B is ~0.50-0.53 with our config (mini-swe-agent swebench.yaml, temp 0.6 precise-coding preset,
    thinking on).
  - The gap to MiMo's 60.0 is most likely harness config and sampling: MiMo gives neither. Xiaomi's uni-agent docs
    list 53.8 (100 turns, 64K) vs 63.8-65.6 (200 turns, 128K) for this model.
  - Candidate follow-ups: the card's general thinking preset (temp 1.0, presence 1.5), a larger step budget.
- **2026-11-05 21:40: the user wants fair scores (no infra-caused failures): re-run the trials that timed out.**
  - Re-runs use v2 serving (TP2 x 4, vllm-router cache_aware, abs 8 / rel 1.3). TB re-runs run on **Daytona**
    (key valid again; avoids Dune's fd failures). SWE re-runs run on Dune, like their runs.
  - **Qwen SWE fair score:** the v2 pair (skycap-qwen-swe-v2, base-qwen-swe-v2) plus re-runs of their timeouts and
    infra failures (rt-*-qwen-swe-v2).
  - **TB:** each of the 4 runs gets rt-<run>, re-running its timeouts and infra failures once. A re-run trial counts
    as it is, even if it times out again.
  - Queues per node (`ops/evals_1104/node_queue.sh`, files in `ops/evals_1104/queues/`, logs `logs/queue_nodeN.log`):
    - node1: rt-base-qwen-swe-v2, then rt-skycap-qwen-tb.
    - node2: rt-skycap-nem-tb, then rt-base-nem-tb.
    - node3: rt-skycap-qwen-swe-v2.
    - node4: rt-base-qwen-tb.
  - Score: `score_evals.py <run> [<fills>...] <rt dir> --rerun-timeouts` (timeouts excluded in every dir but the
    last). Bias note: only failures get a second try. Report both the as-run and the re-run scores.
- **2026-11-05 21:40: vllm-router cache_aware is wrong for the SkyCap arm.**
  - skycap sends token-id `/generate` requests, so the text-prefix tree can't match. Prefix hit rate was ~26%,
    ~70k tok/s of re-prefill and ~156 running per engine (duplicate/abandoned requests); 58/278 timeouts.
  - The baseline arm (chat text) is fine with cache_aware: KV 10-36%, 15/565 timeouts.
  - Stopped skycap-qwen-swe-v2 (runs_dead/skycap-qwen-swe-v2-cacheaware).
  - node3 now runs **skycap-qwen-swe-v3**: TP2 x 4, max_tokens 81920, SkyRL's default consistent_hash (sticky).
    Its queue re-runs v3's timeouts.
  - Removed cache_aware from the SkyCap TB re-runs (node1, node2 queues restarted). Both arms now use sticky,
    TP2 x 4 routing. The baseline is cache_aware, which is also sticky on prefix.
- **2026-11-05 22:30: base-nem-tb-dune done** (no SkyCap, 4/8, Dune, TP1 affinity).
  - 249/267 clean trials, mean **0.297**; per-pass 0.281 / 0.281 / 0.270; pass@3 0.460.
  - SkyCap: 0.304. Parity holds on TB too.
  - 18 Dune fd failures, 33 agent timeouts. Re-run queued on node2 after rt-skycap-nem-tb.
  - skycap-qwen-swe-v3 is healthy: no waiting, KV ~20%, 97-98% prefix hits. Its 23 timeouts so far are the long
    first-batch tasks (~6%; baseline v2 3.6%).
- **2026-11-05 22:50: node_queue.sh bug.** Its busy() grep matched itself (only the first alternative was
  bracketed), so no queue would ever start. Fixed and re-armed all 4 queues; node2's rt-skycap-nem-tb started
  22:50 (Daytona). Verify that the Daytona sandboxes come up.
- **2026-11-05 23:10: Daytona re-runs failed with DaytonaAuthenticationError.**
  - The workers' $D/keys.env still had the old key; the user updated only ~/hackskyrl/charlie_keys.env.
  - Re-copied the keys to all 4 workers (Daytona list OK from node2).
  - Moved the failed rt-skycap-nem-tb / rt-base-nem-tb to runs_dead (*-badkey) and re-armed the node2 queue.
  - Lesson: after a key update, re-copy charlie_keys.env to every worker's $D/keys.env.
- **2026-11-05 23:50: SkyCap tail latency (finding to investigate).**
  - skycap-qwen-swe-v3: sec/step p50 3.4 / p90 **26.1**, timeouts 110/813 (13.5%).
  - base-qwen-swe-v2: p50 3.5 / p90 **8.4**, timeouts 60/1494 (4%). Same serving otherwise (TP2 x 4, sticky,
    max_tokens 81920); the engines are healthy (KV ~20%, 97-98% prefix hits).
  - Suspects: skycap rendering long histories each call (renderer pool 8 per server, 2 CPUs per server); the Dune
    exec relay on large bodies. The rt- re-run of timeouts evens out the score; profile the root cause separately.
  - Added DaytonaServiceUnavailableError, DaytonaBadRequestError (sandbox lost its IP) and
    DaytonaProcessNotFoundError to INFRA (make_fill / score_evals).
- **2026-11-06 00:08: base-qwen-swe-v2 done (as run): see score below; rt-base-qwen-swe-v2 (66 trials) running on node1.**
  - base-qwen-swe-v2 as run: 1499/1500 clean, mean **0.476**, per-pass 0.472 / 0.478 / 0.478, pass@3 0.601, 62 agent timeouts.
- **2026-11-06 00:48: base-qwen-tb-dune done (as run).**
  - 256/267 clean trials, mean **0.238**; per-pass 0.258 / 0.202 / 0.225; pass@3 0.333. 57 timeouts.
  - SkyCap (as run): 0.253.
  - rt-base-qwen-tb (68 trials, Daytona, TP2 vllm-router) running on node4.
- **2026-11-06 01:08: FAIR Qwen SWE baseline (base-qwen-swe-v2 + timeout re-runs): 0.480.**
  - 1500/1500 trials; pass@3 0.602; only 2 timeouts left. As run: 0.476.
  - Re-running the timeouts barely moves it (+0.4 pt): Qwen3.5-9B is ~0.48 in this harness/config, not 60.
  - node1 now on rt-skycap-qwen-tb (66, Daytona).
- **2026-11-06 01:30: FAIR SkyCap Nemotron TB (run + fill + Daytona re-run of timeouts/infra): 0.302.**
  - 265/267 trials, pass@3 0.437. As run: 0.304.
  - 31 of the 38 re-run timeouts timed out again with good serving and no Dune fd issue, so they are genuine.
  - NVIDIA: 24.58; AA: 24.
- **2026-11-06 02:35: skycap-qwen-swe-v3 done (as run): 0.526** (1499 clean, pass@3 0.687, 256 timeouts).
  rt-skycap-qwen-swe-v3 (260 trials, 178 tasks) running on node3.
  - **Open question: the v2/v3 pair is not at parity.**
    - Paired bootstrap over 500 tasks: v3 SkyCap - v2 base = +0.050 [+0.027, +0.075]; v1 base - v2 base = +0.031
      [+0.005, +0.058]. So base-qwen-swe-v2 is low.
    - Ruled out:
      - vllm-router dropping params: top_k and chat_template_kwargs both pass, probed on node4.
      - context-length rejections from the 81920 cap: 0 in v2 base.
      - timeouts: the re-run gives 0.480.
    - The remaining baseline-only difference is the vllm-router + plain `vllm serve` path vs SkyRL engines. Not
      yet explained.
    - The clean SkyCap-vs-stock comparison is the v1 pair (same routing style): **0.511 vs 0.507**.
- **2026-11-06 02:50: FINAL fair SkyCap Nemotron TB: 0.301** (266/267, pass@3 0.432; build-pov-ray re-run finished). node2 now on rt-base-nem-tb (53 trials, Daytona).
- **2026-11-06 03:10: FAIR Qwen TB 2.1** (runs plus the Daytona re-run of timeouts/infra, TP2 x 4):
  - base: **0.252** (266/267, pass@3 0.375). As run: 0.238.
  - SkyCap: **0.271** (266/267 with 65/66 re-runs in, pass@3 0.443). As run: 0.253.
  - MiMo: 27.0; AA: 29.2. SkyCap matches MiMo; the base is 2 pts under. Parity within noise (89 tasks x 3).
  - node4 is free (its queue is done).
- **2026-11-06 03:50: FINAL fair SkyCap Qwen TB** (66/66 re-runs in): see score_evals; node1 queue done (free).
  - SkyCap Qwen TB final fair: **0.270** (267/267, pass@3 0.438).
- **2026-11-06 04:10: FINAL fair SkyCap Qwen SWE v3: 0.551** (1500/1500, pass@3 0.702; as run 0.526).
  - 78 of the 256 re-run timeouts timed out again.
  - **Caveat: the re-run bias favors the arm with more timeouts.** SkyCap re-ran 256 trials, the v2 base 62. The
    fair gap (0.551 vs 0.480, +7 pts) overstates it; as run it was +5 pts.
  - The SkyCap tail latency (p90 26 s/step vs 8) gave SkyCap more timeouts. Those were then retried on the same
    serving and passed at ~30%.
  - node3 queue done (free). Remaining: node2 rt-base-nem-tb.
- **2026-11-06 06:10: ALL DONE. Fair base Nemotron TB: 0.297** (266/267, pass@3 0.432) vs SkyCap 0.301. Final table: RESULTS.md.
