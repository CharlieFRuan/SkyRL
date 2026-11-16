# SkyCap evals on mainline SkyRL: SWE-bench Verified and Terminal-Bench 2.1 (2026-11-04 to 11-06)

Branch `skycap-evals-1104` in `~/hackskyrl/src/SkyRL-upstream`, from NovaSky-AI/SkyRL main f854d6d9:

- The sandbox relay, with `preview` and `exec` transports.
- An eval entrypoint and launcher.
- `skycap.tokenizer` and `skycap.thinking_retention`.
- Terminus-2 with NeMo Evaluator's patches.

Committed locally only: the GitHub token was invalid. Run log and every decision: `PLAN.md` in this folder.

## Results (3 passes per task; mean over trials)

"As run" excludes only provider failures (sandbox create/build/internal errors, failed agent installs). "Fair" also
re-runs the agent timeouts once, on fast serving: TP2 x 4 replicas, sticky routing, Daytona for TB. A re-run that
times out again counts as a failure.

| Benchmark | Model | Arm | As run | Fair | Published |
|---|---|---|---|---|---|
| SWE-bench Verified (500) | Nemotron-3.5-Lightning | SkyCap | **0.480** | (no timeouts) | 50.7 (card, mini-swe-agent 2.4.5) |
| | | stock Harbor | **0.481** | (no timeouts) | |
| SWE-bench Verified (500) | Qwen3.5-9B | SkyCap, run 1 | 0.511 | | 60.0 (MiMo-V2.6 report) |
| | | stock Harbor, run 1 | 0.507 | | |
| | | SkyCap, fast serving (v3) | 0.526 | 0.551* | |
| | | stock Harbor, fast serving (v2) | 0.476 | 0.480 | |
| Terminal-Bench 2.1 (89) | Nemotron-3.5-Lightning | SkyCap | 0.304 | **0.301** | 24.58 (NVIDIA, Terminus 2), 24 (AA) |
| | | stock Harbor | 0.297 | **0.297** | |
| Terminal-Bench 2.1 (89) | Qwen3.5-9B | SkyCap | 0.253 | **0.270** | 27.0 (MiMo), 29.2 (AA) |
| | | stock Harbor | 0.238 | **0.252** | |

\* The re-run bias favors the arm with more timeouts: SkyCap v3 re-ran 256 trials, stock v2 re-ran 62.

## Takeaways

- **SkyCap vs stock Harbor:**
  - Parity on Nemotron SWE-V (0.480 vs 0.481), Nemotron TB (0.301 vs 0.297 fair), and Qwen SWE-V run 1
    (0.511 vs 0.507).
  - Qwen TB is within noise (0.270 vs 0.252 fair).
  - The second Qwen SWE pair differs by 5 pts (paired CI +2.7 to +7.5), because the stock v2 run is low against
    stock run 1 (-3.1 pts, CI +0.5 to +5.8). Ruled out: router param dropping, context-length rejections and
    timeouts. Unexplained; see PLAN.md.
- **Reproduction:**
  - Nemotron SWE-V reproduces the card (48-50 vs 50.7).
  - TB comes out above NVIDIA's 24.58: 30 for both arms. NVIDIA uses NeMo Evaluator's own runner (ECS Fargate,
    8 reps), and we apply only 3 of its 5 Terminus patches.
  - Qwen TB matches MiMo's 27.0.
  - Qwen SWE-V is ~48-55 vs 60.0. MiMo states no harness config. Xiaomi's uni-agent docs show 53.8-65.6 for this
    model depending on turns and context. Our gap is not infra: re-running timeouts barely moves stock (0.476 to
    0.480), and v2's failures are mostly wrong submitted patches.
- **SkyCap performance finding:** on long Qwen trajectories, SkyCap's per-step p90 is 26 s vs 8 s for stock on the
  same serving (median equal, 3.4 s). That gives more agent timeouts. Worth profiling (renderer on long histories?
  exec relay on large bodies?).
- **vllm-router `cache_aware` must not be used for SkyCap:** skycap sends token-id `/generate` requests, so the
  text-prefix tree can't match. It caused a ~26% prefix hit rate and a pile-up. Use SkyRL's default
  `consistent_hash` (97-98% hits).

## Settings

- **SWE:**
  - mini-swe-agent 2.4.5 with its own SWE-bench config (250 steps, 60 s command timeout, tool calls); each task's
    own timeouts (3000 s).
  - Nemotron: temp 1.0 / top_p 0.95, all thinking kept.
  - Qwen: temp 0.6 / top_p 0.95 / top_k 20, the template's thinking rule. Fast-serving runs cap each turn at 81920
    tokens.
- **TB:**
  - Terminus 2 (`terminus_think.py`: parses after `</think>` and replays thinking, as NVIDIA does), with NeMo
    Evaluator's context-reset, unwind and token-anchor patches.
  - 4 vCPU / 8 GB sandboxes, agent timeout max(7200 s, the task's own), 262k context.
  - Nemotron: no turn cap, temp 1.0 / top_p 0.95. Qwen: 250 episodes (AA).
- **Sandboxes:**
  - Dune Cloud: no preview URLs, so the relay uses the exec transport; 30 GB disks for SWE images.
  - Daytona for the TB re-runs; its key was invalid until 11-05 13:20.

## Infra incidents (all handled; details in PLAN.md)

- **Keys:** Daytona and GitHub keys were invalid at the start. Later, the workers had stale copies after the user
  updated the key on the head.
- **Dune outages and failures:**
  - RBAC "access denied" for about 9 h overnight. Every trial created then failed; those were filled in.
  - "too many open files" on long TB sessions.
  - matplotlib image builds fail on the default disk.
  - Sandbox leaks (cleaned with `dune_cleanup.py`).
- **My mistakes, all fixed:**
  - Edited a running launcher.
  - SkyCap's 3600 s TTL expired during slow sandbox builds.
  - TP1 + hash routing overloaded engines for Qwen.
  - cache_aware routing on the SkyCap arm.
  - A queue-runner grep bug.
- **A 10 h stall:** a permission prompt went unanswered.
