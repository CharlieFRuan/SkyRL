# Nemotron-3.5-Lightning RL on CalibForge v2: everything we tried (2026-11-16 to 2026-11-23)

Status: **all Nemotron runs stopped on 2026-11-23 (~21:15 PST) on the owner's instruction.** No further Nemotron runs
are planned. This file is the single entry point: why the runs were done, every experiment, rough results, knobs,
code changes, W&B links, where the artefacts are, and what is still open. It is written so that an agent with no
other context can pick the work up.

This file lives on its own branch (`charlie/nemotron35-rl-summary-1123`, an orphan branch with no shared history),
so the Qwen3.5-9B branch `charlie/calibforge-rl-1115` is untouched. Nothing Nemotron-specific was committed to the
Qwen branch; the Nemotron code changes exist only as local patches, reproduced in the appendix here.

Model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (Mamba-2 / attention / MoE hybrid, 30B, ~3B active).
Stack: SkyRL (fork `CharlieFRuan/SkyRL`, branch `charlie/calibforge-rl-1115`, commit `aada336d`) with the Megatron
trainer backend, vLLM rollout engines, SkyCap with in-sandbox Terminus 2, Harbor tasks. Training set: CalibForge v2
(terminal tasks, binary verifier reward). Eval: Terminal-Bench 2.1 (89 tasks).

Source of truth for every number: the dated log `ops/calibforge_rl_1110/PLAN.md` on the Lila head node
(`~/hackskyrl`), cited below as (Lnnnn) = line number, and `ops/calibforge_rl_1110/FINDINGS.md` sections 2d / 2e for
11-16 to 11-19. Numbers here are copied from those logs. "+-" is one standard error over tasks.

---

## 1. Bottom line

- With every recipe that stayed stable, Nemotron gained **about +0.02 to +0.06 reward on the tasks it trained on**
  after 35 to 60 steps (4 to 6 visits of a 512-task set; +0.02 to +0.04 for three of the four such runs), **about +0.03 on held-out CalibForge tasks**, and at best
  **about +0.03 on TB2.1** (0.266-0.270 at steps 75-100 against a base of 0.237, 2 SE, then back to 0.251 at step
  114). More training did not add more.
- Qwen3.5-9B on the same stack and task set went from about 0.25 to 0.33-0.37 on TB2.1 in about 100 steps, and
  that climb was reproduced on a second cluster (section 9).
- The recipes that were not stable failed the same way: **policy entropy ran away** (from about 0.055 to 0.5 and
  above) while replies got longer and trials ran into the context limit. Stabilising it needed three things
  together: freeze the MoE router and the Mamba mixers, replay rollout routing in the trainer (R3), and keep LR at
  4e-6 or below.
- What is not shown: that the small gain is inherent to the model. The published NVIDIA / vLLM run of this model
  (section 8) also moved its training score only from 0.56 to 0.59 with exact train / inference consistency, and
  collapsed without it, which is the same picture.

## 2. Why these runs were done (the chain of questions)

1. **Goal.** Repeat on Nemotron-3.5-Lightning what worked for Qwen3.5-9B: RL on CalibForge v2, show a TB2.1 gain.
2. **First full run did nothing (11-16).** Horizon `nemo35-cf-v2-hzn-b32-lr2e-6-1116` (2e-6, DPPO, nothing frozen):
   TB2.1 at step 25 was 0.2472 against a base of about 0.24. So: can this model learn at all in our stack?
3. **Overfits (11-16 / 11-17).** 16 tasks x 8, synchronous. With October's MiMo-overfit loss (GRPO with std
   normalisation, token_mean, PPO clip, token TIS) it learns on all three clusters (pass rate 0.40 -> 0.5-0.6);
   with the full-run loss (DPPO 0.15, prompt_mean, no std norm) the same tasks stayed flat. The chat-template
   rendering was verified token for token against the HF template (FINDINGS 2d).
4. **Why does entropy climb so high?** Switching the full run to the GLM-5 style loss (`rollout_is`) made reward
   move but entropy ran away: 0.054 -> 0.086 (step 10) -> 0.200 (14) -> 0.535 (16) with nothing frozen. This became
   the main question. Static module transplants put the rise in the **Mamba mixers (2.8% of parameters) and the
   router gate**; freezing both held entropy near base for a while (FINDINGS 2e).
5. **With the freeze it still crept.** Frozen runs without R3 reached entropy 0.15 by step 25 and 0.5-0.6 by step
   40; the added entropy was in the thinking tokens, replies got longer, context-exceeded endings rose. The
   trainer-vs-rollout logprob gap was 0.04-0.05 from step 1 (October's number was 0.02-0.03).
6. **Is it the train / inference mismatch?** R3 (rollout router replay) cut the gap to 0.011 and the out-of-band
   share from 1.5% to 0.15%; with R3 + freeze entropy stays at 0.055-0.09 for 50+ steps. But then reward barely
   moved in 30 steps.
7. **Is it a length bias in the loss?** prompt_mean weights long trajectories more; we measured a negative
   length-reward coefficient and tried `sequence_mean`. Also tried: score centering, LR 1e-5 / 3e-6 / 2e-6,
   training the mixers again under R3, sampling temperature 1.3 / 1.6.
8. **Does it learn anything with enough signal per task?** Repeat-visit runs on a fixed 512-task subset (6 visits),
   then a held-out LR-0 probe on 256 never-trained tasks, then a continuation on 1,829 new tasks.
9. **Decision (owner, 11-23).** "I think we can conclude this is a very hard model to RL on top of, compared to
   qwen3.5-9b", then "Let's kill all the Nemotron3.5 lightning runs."

## 3. Setup shared by all full runs

- Fully async, batch 32 task groups x 16 trials, max staleness 3 steps, one reply capped at 16,384 tokens, training
  context 131,072, agent timeout 3,600 s, no summarization, thinking kept across turns, sampling T 1.0.
- Timeouts and context-exceeded endings are trained as reward 0 (owner's choice); zero-variance groups filtered;
  no std normalisation; MoE load-balancer bias update off (`moe_router_bias_update_rate=0.0`).
- `rollout_is` = the GLM-5 style loss: importance ratio between trainer and rollout logprobs with a band
  `eps_clip_low=0.5`, `eps_clip_high=4` on Lila (Horizon reports its band as [0.5, 5]), tokens outside the band
  dropped, no TIS.
- "Frozen" = router and Mamba mixers not trained. Lila: `freeze_moe_router=true` + `freeze_mamba_mixer=true`
  (`requires_grad=False`, local patch, appendix A1). Horizon and JUPITER: `trainer.policy.lr_zero_modules=[router,mamba]`
  (an optimizer group at LR 0, so Adam state exists and resume works; Horizon diff `lr_zero_modules_1118.diff`).
- R3 = `moe_enable_routing_replay` (launcher `R3=1`): the trainer forward uses the expert indices the rollout
  engine recorded.
- Metrics: `policy_entropy` (section 8 for its definition), `policy/minibatch_rollout_logprobs_abs_diff_mean`
  ("gap"), `grad_norm`, `reward/avg_raw_reward`.
- Clusters: **Lila** (Anyscale Ray, 4 nodes x 8 B200; sandboxes on the shared Daytona RL org; 2,598-task cleaned set
  `calibforge_v2_daytona3`), **Horizon** (Slurm; sandboxes on Dune; 2,788-task set), **JUPITER** (Slurm, GH200;
  Dune). An epoch of the full set is about 57 trained steps (keep rate about 0.65); an epoch of a 512-task set is
  10 trained steps.
- Paired readout used from 11-21 on: per task group of 16 trials, run mean minus base-reference mean for the same
  task, SE over tasks, complete 16/16 groups only. Partial visits read too high (fast, easy groups finish first).
- Launch line of the last Lila run (the template for all R3 runs; `run_rl_nem.sh` is on the Lila head under
  `ops/calibforge_rl_1110/`):

```bash
R3=1 LR=4.0e-6 EPOCHS=6 NUM_ENGINES=10 POLICY_NUM_NODES=2 POLICY_GPUS_PER_NODE=4 \
SKYRL_POLICY_PG_STRATEGY=STRICT_SPREAD TASKS_DIR=$D/data/calibforge_v2_daytona3_sub512 \
bash ops/calibforge_rl_1110/run_rl_nem.sh nem35-cf-sub512-lila-b32-lr4e-6-ris-frz-r3-pmean-1123 \
  trainer.algorithm.policy_loss_type=rollout_is trainer.algorithm.eps_clip_low=0.5 trainer.algorithm.eps_clip_high=4 \
  trainer.algorithm.off_policy_correction.tis_ratio_type=null trainer.algorithm.loss_reduction=prompt_mean \
  trainer.algorithm.grpo_norm_by_std=false trainer.algorithm.zero_variance_filter=true \
  generator.apply_overlong_filtering=false \
  trainer.policy.megatron_config.transformer_config_kwargs.moe_router_bias_update_rate=0.0 \
  trainer.policy.megatron_config.freeze_moe_router=true trainer.policy.megatron_config.freeze_mamba_mixer=true
```

## 4. Knobs tried, and what each did

| Knob | Values tried | Result |
|---|---|---|
| Loss | DPPO 0.15; `rollout_is`; October recipe (GRPO std-norm, token_mean, PPO clip, token TIS) in overfits | DPPO + freeze: entropy and reward flat for 35 steps. `rollout_is`: reward moves, entropy creeps unless R3 is on. October recipe learns in 16-task overfits; full-run recipe did not. |
| Freezing | nothing; router only; router + Mamba mixers | Nothing frozen, no R3: entropy 0.054 -> 0.535 in 16 steps. Nothing frozen with R3 (JUPITER D2): entropy 0.055 -> 0.115 in 23 steps, no reward gain over the frozen arm, context share 0.17 -> 0.32. Mixers trained with R3: no gain (+0.005 +- 0.004 on Lila, +0.005 +- 0.006 on JUPITER), entropy a little higher. |
| R3 | off; on | Gap 0.045 -> 0.011, out-of-band 1.5% -> 0.15%. Only R3 runs stayed stable past 40 steps. |
| LR | 2e-6, 3e-6, 4e-6, 1e-5 | 1e-5 is worse (JUPITER: -0.035 +- 0.007 vs 4e-6 at visit 2, capped replies 0.32 -> 0.42). 2e-6 (no R3) stable for 80 steps but TB2.1 stayed at base. Lowering LR after a runaway started did not stop it. |
| `loss_reduction` | prompt_mean; sequence_mean | Without R3, sequence_mean delays the runaway by about 20 steps, does not prevent it. With R3: similar or slightly more reward on training tasks, always more context-exceeded endings; TB2.1 at base at step 45 (Horizon). |
| Score centering (+ sample-support replay, top_k 20) | Horizon 50 steps, JUPITER 5 steps | Entropy flat (0.058-0.068) without R3, but no training-task gain; TB2.1 0.2509 / 0.2584 at 25 / 50 with 27% context endings. |
| Sampling temperature (LR 0 passes) | 1.0, 1.3, 1.6 | T 1.3: no change (-0.001 +- 0.008). T 1.6: -0.222 +- 0.028, context-exceeded 39%. |
| Thinking | kept across turns; stock template flow (JUPITER k4) | Stock flow much worse: 0.348 vs 0.475 batch reward at steps 16-20. |
| Task set | full set (one visit per ~57 steps); 512-task subset x 6 visits; 1,829 new tasks | Gains only show clearly with repeat visits; a second visit of new tasks added nothing (+0.003 +- 0.007). |

## 5. Run inventory

W&B: entity `sky-posttraining-uc-berkeley`, project `skycap-calibforge-fully-async` (overfits:
`skycap-calibforge-overfit`). Lila run URLs are
`https://wandb.ai/sky-posttraining-uc-berkeley/skycap-calibforge-fully-async/runs/<id>`.

### 5a. Lila (Daytona), all at batch 32 x 16, `rollout_is` unless noted

| Run | W&B id | Recipe (LR / reduction / R3 / frozen / tasks) | Steps, end | Result |
|---|---|---|---|---|
| `nem35-cf-of16x8-lr5e-6-oct-1116` (+ `-noolf-1117`, other 16x8 overfits) | overfit project | October recipe, sync, 16 tasks x 8 | 20 | pass rate 0.40 -> 0.51-0.62; context endings 2 -> 24-36 of 128 (FINDINGS 2d) |
| `nem35-cf-v2-lila-b32-lr4e-6-ris-nofrz-1118-r2` | `vgc4f43h`, `xsmlplxc`, `ctlc8h9o` | 4e-6 / prompt_mean / off / nothing / full | 20, stopped | entropy 0.054 -> 0.535 at step 16; reward 0.40-0.56 no trend (L2199) |
| `nem35-cf-v2-lila-b32-lr3e-6-ris-frz-1119` | `22kws0i3` | 3e-6 / prompt_mean / off / router+mixers / full | 15, stopped | entropy 0.058 -> 0.069, gap 0.044-0.046; reward windows 0.486, 0.441 (L2214-L2230) |
| `nem35-cf-v2-lila-b32-lr4e-6-ris-frz-seqmean-1119` | `vsoq1ben` | 4e-6 / sequence_mean / off / router+mixers / full | 50, stopped by rule | entropy 0.058 -> 0.087 (36-40) -> 0.17 (50); reward windows 0.49 -> 0.43-0.46; gap 0.041 -> 0.06 (L2288-L2303) |
| `nem35-cf-v2-lila-b32-lr4e-6-ris-frz-r3-seqmean-1120` | `5gfjzpgc` | 4e-6 / sequence_mean / **on** / router+mixers / full | 30, stopped | entropy flat 0.054-0.057, gap 0.011-0.014; reward windows 0.44 0.43 0.53 0.46 0.45 0.42: stable, no learning visible in 30 steps (L2342) |
| `nem35-cf-v2-lila-b32-lr1e-5-ris-frz-r3-seqmean-1120` | `su79frph` | **1e-5**, else as above | 20, stopped | vs the 4e-6 run -0.006 +- 0.004; entropy 0.056 -> 0.082-0.089; gap up to 0.033 (L2373) |
| `nem35-cf-v2-lila-b32-lr0-baseref-r3-1120` | `owej8d4m` | LR 0, T 1.0 (base reference pass) | 5 no-op steps | base references for paired readouts (L2378) |
| `nem35-cf-v2-lila-b32-lr0-baseref-t13-r3-1120` | `su2vm0uv` | LR 0, sampling T 1.3 | 2 | -0.001 +- 0.008 vs T 1.0 (L2398) |
| `nem35-cf-v2-lila-b32-lr0-baseref-t16-r3-1121` | `35963cxa` | LR 0, T 1.6 | <1 | -0.222 +- 0.028, context-exceeded 39% (L2432) |
| `nem35-cf-v2-lila-b32-lr4e-6-ris-rfrz-r3-seqmean-1120` | `ft97dz9d` | 4e-6 / sequence_mean / on / **router only** / full | 20, stopped | vs frozen run +0.005 +- 0.004 (1,038 groups); entropy 0.053-0.065 (L2430) |
| `nem35-cf-sub512-lila-b32-lr4e-6-ris-frz-r3-seqmean-1121` | `47d926xj` | 4e-6 / sequence_mean / on / router+mixers / **sub512 x 6 visits** | 60, finished | vs base by visit: +0.001, +0.019, +0.036, +0.047, +0.059 +- 0.008, +0.057 +- 0.009 (plateau). Context-exceeded 0.18 -> 0.27; more turns per trial (33 -> 42); entropy 0.053-0.066 (L2447-L2509) |
| `nem35-cf-ho256-lila-lr0-base-r3-1122` | `puf2kah3` | LR 0, base, 256 held-out tasks | 1 pass | base 0.390-0.392 on held-out (L2521) |
| `nem35-cf-ho256-lila-lr0-sub512ckpt-r3-1122` / `b` / `c` | `9ixjpn5i`, `905nrmv9`, `vwkky8jt` | LR 0, checkpoint 61 of the run above, held-out | `c` is the valid one | **held-out +0.031 +- 0.012** (0.422 vs 0.391, 254 tasks); context-exceeded 0.313 vs 0.193 (L2538). `a` and `b` are failed attempts (bugs in section 7) |
| `nem35-cf-new1829-lila-b32-lr4e-6-ris-frz-r3-seqmean-from61-1122` | `u5htg08s`, `von2a9hq` | same recipe continued from checkpoint 61 on **1,829 new tasks** | 62-105, stopped by rule | second visit vs first on the same tasks +0.003 +- 0.007 (444 tasks); context share +0.068 +- 0.006 (0.317 -> 0.385) (L2592) |
| `nem35-cf-ho256-lila-lr0-new1829ckpt95-r3-1122` | `16eyw2sp` | LR 0, checkpoint 95, held-out | 1 pass | held-out +0.034 +- 0.014 vs base; vs checkpoint 61 **+0.000 +- 0.009**; context-exceeded 0.365 (L2576) |
| `nem35-cf-sub512-lila-b32-lr4e-6-ris-frz-r3-pmean-1123` | `91gluz6u` | 4e-6 / **prompt_mean** / on / router+mixers / sub512 | 48 (checkpoint 45), **killed 11-23 21:15** in the tail of visit 5 | vs base by visit: -0.003 +- 0.004, +0.013 +- 0.004, +0.010 +- 0.005, +0.035 +- 0.005, +0.036 +- 0.006 (502 of 512 tasks). Entropy 0.054 -> 0.087; gap 0.011-0.015; context share 0.16-0.24 (base ~0.19). No held-out probe was run. |

Mechanism seen in the held-out probes (base / checkpoint 61 / checkpoint 95, 250 tasks): turns per trial 31.5 /
44.8 / 42.4; end context 61k / 78k / 82k; context-exceeded 0.19 / 0.31 / 0.37; reward among clean endings 0.49 /
0.62 / 0.67 (L2573). The sequence_mean policy works longer per task: it solves more of the tasks it finishes and
runs out of context on more of the rest. Trials at >= 240 of 250 turns stayed at 0.1-1.0%: no looping.

### 5b. Horizon (Dune), full 2,788-task set

| Run | W&B | Recipe | Steps, end | Result |
|---|---|---|---|---|
| `nemo35-cf-v2-hzn-b32-lr2e-6-1116` (first full run) | `rm1l3c7w`, `m1z8g0s7` | 2e-6 / DPPO / prompt_mean / no R3 / nothing frozen | 89 (checkpoint 85), not resumed | reward flat (0.390 -> 0.411); entropy 0.051 -> 0.418 from about step 35; TB2.1 step 25 0.2472, step 50 0.2247 (base 0.2372) |
| 16x8 overfit arms (October vs full-run recipe, R3 on / off, arms A, C, M, B; ids in appendix B) | appendix B | see FINDINGS 2d / 2e | 5-30 | October recipe learns; `rollout_is` + freeze overfit 0.59 -> 0.83 |
| H `nemo35-cf-v2-hzn-b32-lr4e-6-ris-frz-1118` | `xfqock0w`, `djoi3k9a`, `exnyj3oq` | 4e-6 / rollout_is / prompt_mean / no R3 / frozen | 25, stopped | entropy 0.059 -> 0.148; TB2.1 step 25: 0.2734 and 0.2509 (pooled 0.2622) |
| `...-lr4to2e-6-ris-frz-from25-1119` | `86b728if` ... `62cq4bzm` (5 ids, appendix B) | H continued at 2e-6 | 26-35 | entropy kept climbing 0.23 -> 0.44 |
| `...-lr2e-6-ris-frz-seqmean-from35-1119` | `6jkz47g1` | the above switched to sequence_mean | 36-40 | entropy 0.37 -> 0.57: switching late does not stop it |
| `nemo35-cf-v2-hzn-b32-cand-from25-1118` ("2e-6 candidate") | `vmw1j8q5`, `xvfdrtud` | first run's step 25 + rollout_is + freeze, 2e-6, no R3 | to 80 | entropy ~0.06-0.07, stable; TB2.1 0.2285 (50), 0.2360 (80): base level |
| `nemo35-cf-v2-hzn-b32-lr4e-6-ris-frz-sc-1119` | `d8mygg6r`, `pv20y7eg` | H recipe + score centering + sample-support replay, top_k 20 | 50, stopped | entropy 0.058-0.068; no training-task gain vs the candidate; TB2.1 0.2509 (25), 0.2584 (50) |
| `nemo35-cf-v2-hzn-b32-lr4e-6-ris-frz-r3-1119` (**longest run**) | 12 ids, first `4y0fwmpi`, last `hwnic1cp` (appendix B) | 4e-6 / rollout_is / prompt_mean / **R3** / frozen | 114 (2 epochs), job ended | entropy 0.057 -> 0.097 (85) -> 0.12 (104-112); gap 0.011-0.016; task-matched edge over the 2e-6 lineage flat at +0.036-0.038 from step ~26; TB2.1: 25 0.2285, 35 0.2434, 55 0.2434, **75 0.2659, 100 0.2697**, 114 0.2509 |
| `nemo35-cf-v2-hzn-b32-lr4e-6-ris-frz-r3-seqmean-1122` (sequence_mean twin) | 8 ids, first `zquith8x`, last `sc63sko6` (appendix B) | the above with sequence_mean | 58, **cancelled 11-23** | no consistent edge over the prompt_mean run (-0.019 to +0.019 by window); context share 0.23-0.29 vs 0.21; **TB2.1 step 45: 0.2360 = base** |
| `...-rfrz-r3-1121`, `...-sub512-...-seqmean-1122` | - | prepared | never launched | - |

### 5c. JUPITER (Dune)

| Run / arm | W&B | Recipe | Steps, end | Result |
|---|---|---|---|---|
| `nemo-jup-of16x8-lr5e-6-1117` (overfit; three earlier 32-task overfits in appendix C2) | `5ciahmo7` | full-run recipe | 8 | 0.605 -> 0.661, ~1.4 SE |
| `nemo-jup-cf-v2-b32-lr2e-6-1117` (control, twin of Horizon's first full run) | `1h2s9nt6`, `lx4apacy`, `wuomk3ka` | 2e-6 / DPPO / prompt_mean / no R3 / nothing frozen | 37 (checkpoint 35) | reward flat (0.454 -> 0.469 over 24 steps); entropy 0.053-0.064 to step 30, then 0.068 -> 0.079 by 37 |
| A/B arms from the control's checkpoint 30 (`...-ab30-nolb`, `-frz`, `-rollis`, `-mzf-1118`) | appendix C3 | 2e-6; one change each: balancer off / router frozen / rollout_is / router + Mamba mixers at LR 0 | 37-50 | balancer off, router-only freeze and rollout_is did not slow the entropy rise (rollout_is: 0.128 at 48-50); **freezing mixers + router held it at 0.071 with reward 0.512 vs 0.457**. This is where the freeze recipe came from |
| J-r2 `nemo-jup-cf-v2-b32-lr4e-6-ris-frz-1118-r2` | `1nvvf22s`, `lce9kfit`, `uufisuec` | 4e-6 / rollout_is / prompt_mean / no R3 / frozen / full | 40, stopped on entropy rule | entropy 0.06 -> 0.17 (26-30) -> 0.63 (40); clean reward 0.53 -> 0.62 then falling; TB2.1 (8 engines): step 25 0.292, step 35 0.2472 (base 0.2472) |
| H-r2 `nemo-jup-cf-v2-b32-lr4e-6-dppo-frz-1118-r2` | `lq60kl2k`, `p0tanb8t` | same with **DPPO** | 35, walltime | entropy flat 0.052-0.060, clean reward flat; TB2.1 step 25: 0.258 |
| k4 `nemo-jup-of16x8-ris-nolb-trdefault-lr4e-6-1118-k4` | `riblvyjy`, `s19thxfo`, `1jnxn9ig` | 16x8 overfit, stock thinking flow, 4 turns trained per trajectory | 24 | reward 0.28-0.35 vs 0.43-0.47 with thinking kept |
| `nemo-jup-cf-v2-b32-lr4e-6-ris-frz-r3-1119` | 7 ids incl. `ai4sqcle` (appendix C4) | J-r2 + R3 | 26, parked | gap 0.011 (0.045 without R3), entropy 0.056-0.066, clean reward 0.54 -> 0.56 |
| `nemo-jup-cf-v2-b32-lr4e-6-ris-frz-r3sc-1120` | `53bp1o04` + 3 (appendix C4) | R3 + score centering | 5, retired | too short to read |
| `nemo-jup-cf-v2-b32-lr4e-6-ris-rfrz-r3-1121` | `njfw5njk`, `x1ywhr7t` | R3, mixers trained, full set | 15, stopped | +0.012 +- 0.014 vs frozen at 11-15; entropy 0.075 vs 0.060 |
| rv512 A `nemo-jup-cf-rv512-b32-lr1e-5-ris-frz-r3-1121` | `2k1cr4e8`, `95gylg9a`, `bp6lp84f` | **1e-5**, prompt_mean, R3, frozen, 512 tasks | 19, stopped | visit 2 vs base -0.033 +- 0.008; A - C -0.035 +- 0.007 |
| rv512 B `...-lr4e-6-ris-rfrz-r3-1121` | `xjyq09u0`, `wak64a1y` | 4e-6, router only frozen | 21, stopped | B - C at visit 2 +0.005 +- 0.006; entropy 0.067 -> 0.087 |
| rv512 C `...-lr4e-6-ris-frz-r3-1121` (reference arm) | 6 ids, first `1heg2a7d`, last `j0vajcij` | 4e-6, prompt_mean, R3, frozen | 51, **killed 11-23** | vs base by visit: -0.006, +0.003, +0.012 +- 0.006, **+0.022 +- 0.006**, +0.019 +- 0.007 (visit 5, final); context share 0.16-0.18; entropy 0.073-0.088 at steps 42-51 |
| rv512 D2 `...-lr4e-6-ris-nofrz-r3-1122b` | `wwsm00tr`, `qzugx9lt` | 4e-6, **nothing frozen**, R3 | 23, entropy guard | D2 - C: +0.001 +- 0.005, +0.001 +- 0.006, -0.025 +- 0.011 (partial); context share 0.165 -> 0.231 -> 0.317; entropy 0.055 -> 0.115; fewer, much longer replies |
| rv512 E `...-lr4e-6-ris-frz-r3-seqm-1122` | 5 ids, first `sakpeofz`, last `uht6nw1t` | 4e-6, **sequence_mean**, R3, frozen | 35, **killed 11-23** | vs base by visit: -0.001, +0.005, +0.024 +- 0.007, +0.035 +- 0.008 (visit 4, partial, 394 tasks); E - C +0.013 +- 0.006 at visit 3; context +0.022 +- 0.006; 7 fewer turns, +341 reply tokens per turn; entropy flat 0.061-0.065 |

Horizon's own final table (every run and eval with its W&B ids, checkpoint locations and code changes) is appendix
B. JUPITER's is appendix C (31 run directories, checkpoints, code and script changes, per-visit tables).

## 6. TB2.1 evals of Nemotron checkpoints (89 tasks x 3, T 1.0, no summarization, 16,384 tokens per reply)

| Checkpoint | avg | pass@3 | context endings | trials with a capped reply |
|---|---|---|---|---|
| Horizon base, three evals pooled (801 trials) | **0.2372** | - | 15-21% | 32-35% |
| JUPITER base (12 engines / 8 engines) | 0.2659 / 0.2472 | 0.3708 / 0.3596 | 14% / 19% | - |
| Published for this model | 0.2458 (model card), 0.2444 (report) | | | |
| first full run (2e-6 DPPO), step 25 | 0.2472 | 0.3371 | - | - |
| H (no R3, frozen, 4e-6), step 25, two evals | 0.2734, 0.2509 | 0.3708, 0.3596 | 27-29% | - |
| 2e-6 candidate, step 50 (two evals) / step 80 | 0.2285, 0.2285 / 0.2360 | 0.31-0.36 | ~19-22% | - |
| score centering, step 25 / 50 | 0.2509 / 0.2584 | 0.3708 / 0.3820 | - / 27.3% | - / 48.3% |
| R3 prompt_mean, step 25 / 35 / 55 | 0.2285 / 0.2434 / 0.2434 | 0.35-0.37 | 16-18% | 24% (55) |
| R3 prompt_mean, step 75 | **0.2659** | 0.3596 | 13.9% | 25.5% |
| R3 prompt_mean, step 100 | **0.2697** | 0.3933 | 18.8% | 31.3% |
| R3 prompt_mean, step 114 | 0.2509 | 0.3708 | 27.6% | 36.8% |
| R3 sequence_mean twin, step 45 | 0.2360 | 0.3146 | 28.3% | 42.8% |
| J-r2 (JUPITER, 8 engines), step 25 / 35 | 0.292 / 0.2472 | 0.416 / 0.3596 | 23% / 21% | - |
| H-r2 DPPO (JUPITER, 8 engines), step 25 | 0.258 | 0.371 | 18.7% | - |

- Per task against the 3-eval base: R3 75 + 100 pooled **+0.031 +- 0.016** (2 SE); R3 114 +0.014 +- 0.026; early
  R3 (25 / 35 / 55) +0.001 +- 0.016; sequence_mean twin 45 -0.001 +- 0.022 (L2545, L2550, L2681).
- One 89 x 3 eval has an SE of about 0.02 between repeats, so single evals cannot rank checkpoints.
- **The 16,384-token reply cap matters for Nemotron**: a third of base trials contain at least one capped reply and
  those trials pass at 0.084 against 0.309 for the rest (L2458). Policies that drift toward longer replies lose on
  TB2.1 through the cap and the context limit even when their reward among clean endings rises.
- Serving speed was the same across Horizon evals (209-231 tokens/s per call), so it does not explain differences.
- Never evaluated on TB2.1: any Lila checkpoint (Lila's sandboxes were on the Daytona RL org, where TB2.1 cannot
  run), JUPITER's frozen R3 step 25 (eval job held, never ran), the rv512 arms.

## 7. Bugs and traps found on the way (all affect a future run on this stack)

- **Megatron optimizer state cannot be loaded at a different LR** (param groups are keyed by max_lr / min_lr):
  resuming at another LR or at LR 0 raises KeyError. Patches: Horizon LR-change load patch; Lila weights-only load
  at LR 0 (appendix A3).
- **Weights-only load silently resets to base**: the optimizer keeps fp32 master copies built from the base model
  and writes them back at the first step, even at LR 0. Fix: `optimizer.reload_model_params()` after the load
  (appendix A4). This invalidated the first held-out probe attempt.
- **Step-loop bound**: resuming a checkpoint onto a dataset with fewer steps per epoch gives an empty loop
  (appendix A5).
- **R3 "received a mix" crash**: a fully masked group carries no routes while others do; padding fix (appendix A2).
- **Placement**: Ray `PACK` put both 4-GPU trainer bundles on one node and exhausted host memory (trainer node
  deaths on 11-19); env switch `SKYRL_POLICY_PG_STRATEGY=STRICT_SPREAD` (appendix A6).
- **Checkpoint traps**: `latest_ckpt_global_step.txt` is written before `fully_async_state.pt` (a stop in between
  breaks the resume); a non-numeric `global_step_*` name in `ckpts/` crashes the checkpoint scan; cleanup of old
  checkpoints runs on every node and the driver (concurrent deletes stalled saves on shared Lustre); consumed task
  ids in the state file are dataset indices, so a mid-epoch checkpoint misbehaves on another dataset.
- **Ray retries the entrypoint task** by default (Horizon patch `max_retries=0`).
- **An epoch cannot end while any trial is in flight**: one hung sandbox stalls the whole run (Daytona
  `create_session` has no request timeout; a 60-minute stall on Lila). JUWELS' `bound_session_calls` change fixes it.
- **Logged entropy lags**: in a fully-async run the logged value mixes stale batches (0.148 logged when batches from
  the current policy read 0.23).
- **Mamba state dtype** is not the cause of the 0.04 gap: production already keeps the SSM state in fp32; the
  residual is decode-vs-prefill and batching numerics (probe in `ops/calibforge_rl_1110/mamba_dtype_1119/`).
- **Staleness**: trained staleness reaches 4-5 although the limit is 3; prefix caches are not cleared on weight sync.
- Infra: Dune outages on 11-19, 11-20 and 11-23 (failed setups are masked, not trained as 0; probe gates added);
  Daytona SDK 2.x silently breaks label cleanup (pin `daytona<1`); JUPITER "CUDA OOM" was host RAM spilling into
  GPU memory on GH200 nodes; a Nemotron checkpoint with optimizer state is about 0.4 TB.

## 8. Entropy metric: ours vs verl vs the vLLM blog (checked 11-23)

The owner asked whether the "actor entropy" of about 0.05 in the vLLM blog post
(https://vllm.ai/blog/2026-10-14-train-inference-consistency, the Nemotron SWE-bench Pro figure with three panels:
train-inference KL, training score, actor entropy) is the same quantity as our `policy_entropy`.

**Yes, it is the same quantity by definition; one aggregation detail differs, and one part could not be read.**

| | SkyRL (our runs) | verl `main` |
|---|---|---|
| Per-token value | Shannon entropy of the trainer's next-token distribution over the **full vocabulary**, natural log | same |
| Megatron formula | `logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits` (`skyrl/backends/skyrl_train/distributed/megatron/model_utils.py:1430`; a fused LM-head kernel computes the same value) | identical expression in `verl/utils/megatron/tensor_parallel.py` (`_VocabParallelEntropy`); FSDP path: `logsumexp(logits) - sum(softmax(logits) * logits)` |
| Temperature | logits divided by the sampling temperature first (1.0 in our runs) | same (divided before entropy and logprob) |
| Tokens averaged | `loss_mask` = assistant-generated tokens (thinking included) | `response_mask` |
| Aggregation | masked token mean per micro-batch (`megatron_model_wrapper.py:1104-1131`), then a plain mean over micro-batches and ranks (`worker_utils.reduce_metrics`) | `agg_loss(..., loss_agg_mode)`, default `token-mean` over the global batch |
| Logged as | `policy_entropy` | `actor/entropy_loss` (older code and SkyRL's verl integration: `actor/entropy`) |

- Difference: our number is a mean of per-micro-batch means, verl's default is one token-weighted mean. With
  micro-batches of unequal token counts these differ slightly; this cannot produce a factor.
- Not verifiable from public text: the blog's Nemotron run used a Megatron "mLite" engine in a fork
  (`aoshen02/Megatron-LM` PR 7, `aoshen02/verl` PRs 11 and 12), whose recipe script did not render; the post does not
  say how entropy is aggregated or which tokens are masked in multi-turn trajectories. Its engine returns the
  `entropy` tensor to verl's standard loss, so the default path above is the likely one.
- Independent check on our side (11-18, `ops/calibforge_rl_1110/entropy_audit/`): base-model entropy on our
  transcripts is 0.0585 from the trainer, 0.0631 from HF full-vocabulary logits, 0.058 from vLLM. So our 0.055 at
  step 0 is real and agrees with the blog's 0.05.
- What the blog reports for this model (GRPO, 32 tasks x 8, ~57k tokens per trajectory, SWE-bench Pro, no TIS):
  **baseline (NVFP4 QAT + R3, default kernels): entropy 0.05 -> 2.6, training score 0.56 -> 0.22, train-inference
  KL 0.015 -> 0.002; bitwise-consistent trainer and rollout: entropy stays ~0.05, score 0.56 -> 0.59, KL exactly 0.**
  Their collapse without exact consistency matches our runaway; their stable run gains +0.03 on its training score,
  the same size as ours. Their fix (trainer forward calling vLLM's kernels, batch-invariant Mamba scan, fp32 router
  GEMM) is in forks and dev builds, not in a release we ran. Our substitute was R3 + frozen router and mixers,
  which holds the gap at 0.011-0.016 rather than 0.
- NVIDIA's own NeMo-RL recipe for this model (rlvr.yaml): 512 prompts x 16 per step, LR 4e-6, ratio clip 0.2 / 0.28,
  token-level loss, TIS weights in [0.2, 5], router frozen with fp32 router, trajectory age 1, thinking kept. Our
  batch is 32 x 16 at the same LR (16x fewer prompts per step) with staleness 3.

## 9. Qwen3.5-9B on the same stack, for comparison (not on this branch's subject; see the Qwen branch and FINDINGS 2b)

TB2.1 x5 on Dune, avg: Lila run (11-16): step 25 0.2787, 50 0.2989, 75 0.3281, final (104 steps) 0.3461; base
0.22-0.25. JUWELS reproduction (A100-40GB nodes): base 0.2472, 25 0.2629, 50 0.3236, 75 0.3236, 100 0.3303, 104
0.3236 / 0.3483 (5 engines) and 0.3663 (12 engines, Lila's serving speed), 110 0.3213, 115 0.3056, 120 0.3236. A
third epoch added nothing and produced a looping failure (early end-of-turn tokens, parse-error loops) peaking at
step 115. Slower serving costs about 0.03 on this eval (agent timeouts).

## 10. Artefacts

- **Logs**: Lila head `~/hackskyrl/ops/calibforge_rl_1110/PLAN.md` (dated log, 2,700 lines) and `FINDINGS.md`;
  per-run driver logs on the Lila driver under `$D/logs/<run>.log`, `D=/mnt/local_storage/charlie`.
- **Checkpoints, Lila** (node-local NVMe on the driver worker; they disappear if the Anyscale nodes are released):
  `$D/runs/<run>/ckpts/global_step_N`. Present on 11-23: prompt_mean twin 30, 40, 45; new1829 continuation 80, 90,
  100, 105 and `ckpts_probe/global_step_95`. In S3 (bucket and prefix in `~/hackskyrl/ops/upload_ckpt_s3.py`):
  prompt_mean twin 10, 11, 20, 21, 30, 31, 40, 41, 45; new1829 continuation 70, 80, 90, 98, 100, 105; the sub512
  sequence_mean run's 10-step and epoch-end checkpoints up to 61; earlier runs' 10 / 20 / 30 / 50 as logged. Each
  is about 0.4 TB with optimizer state.
- **Checkpoints, Horizon / JUPITER**: on their shared filesystems (Horizon R3 run: every 5th step 5-110, 57, 114;
  twin: up to 58; JUPITER rv512 arms C 51, E 35, D2 23, B 21, A 19; full lists in appendices B1 and C6). Public HF repo
  `CharlieFRuan/nemotron35-calibforge-ckpts` holds a few (Horizon H step 25, R3 step 75, early 2e-6 checkpoints);
  uploads there are paused (storage quota) and nothing is deleted without the owner.
- **Trial data**: `$D/runs/<run>/trials/` (Harbor result files, SkyCap records). TB traces and task data are
  private and are not on this branch.
- **Task subsets** (Lila, `$D/data/`): `calibforge_v2_daytona3` (2,598), `..._sub512`, `..._ho256` (held-out),
  new1829; base references for paired readouts in `calibforge_v2_daytona3_sub512.refs.json`. JUPITER:
  `data/calibforge_v2_rv512_1121`.
- **Analysis scripts** (Lila head, `ops/calibforge_rl_1110/`): `entropy_audit/`, `mamba_dtype_1119/`,
  `module_transplant.py`, `length_bias.py`, `surprisal_split.py`, `eval_agg.py`, `base_pool.py`, the `switch_*.sh`
  chain scripts, `ckpt_backup_watch.sh`.

## 11. State at the kill, and what is open

- **Lila**: run stopped by PID 21:14-21:17 PST 11-23; 30 labelled sandboxes deleted, 0 left; GPUs free; checkpoint
  45 verified in S3. The planned sixth visit and the held-out probe of the prompt_mean run were not done.
- **Horizon**: resume job and the step-75 eval job of the sequence_mean twin cancelled; watchers stopped; 0 labelled
  sandboxes; twin ends at checkpoint 58.
- **JUPITER**: confirmed 06:31 CET 11-24. Arms C and E stopped by PID at checkpoints 51 and 35 (C's visit 5 is its
  last complete visit); all 8 running, queued and held jobs cancelled; 726 labelled Dune sandboxes deleted, 0 left;
  proxy stopped; checkpoints, trial directories and logs kept. Its final table is appendix C.
- Never done: a TB2.1 eval of any repeat-visit checkpoint (Lila checkpoint 61 has the largest training-task gain,
  +0.057, and +0.031 held-out); a run with exact train / inference consistency (the vLLM blog's approach); a larger
  batch at the same LR (NVIDIA's recipe uses 512 prompts per step); raising the 16,384 reply cap in eval.
- Not established: why the no-R3 logprob gap (0.04-0.05) is twice October's; whether the +0.02-0.06 gains are
  learning of the tasks or a general shift toward working longer per task (held-out transfer and the context-share
  rise point to the second); whether the late TB2.1 gain of the R3 run (+0.031 +- 0.016) is real.
- Local patches are not pushed anywhere except as text in the appendix below. Decisions left with the owner: which
  of them to upstream; whether to keep the 0.4 TB checkpoints.


---

## Appendix A. Code changes made for the Nemotron runs (Lila), as text

Base: fork `CharlieFRuan/SkyRL`, branch `charlie/calibforge-rl-1115`, commit `aada336d`. None of these is committed
or pushed; they live in the working tree on the Lila head (`~/hackskyrl/src/SkyRL-upstream`) and / or in the
checkout on the Lila workers (`$D/SkyRL`, with `.bak1122` copies of the three files patched on 11-22).
Horizon and JUPITER carry their own equivalents (appendix B4, appendix C).

### A1. `freeze_mamba_mixer` (config flag + freeze before the optimizer is built). Applied with this script:

```python
"""Add megatron_config.freeze_mamba_mixer to a SkyRL checkout (local, uncommitted; 11-18). Idempotent.
    python add_freeze_mamba.py <SkyRL dir>
Freezes (requires_grad=False, before the optimizer / DDP are built; from a fresh start only) every parameter whose
runtime name matches .mixer.(in_proj.weight | out_proj.weight | conv1d_weight | conv1d_bias | A_log | D | dt_bias).
megatron-core's MambaMixer registers conv1d as flat params conv1d_weight / conv1d_bias (JUPITER, 11-18)."""
import sys
root = sys.argv[1]
def edit(path, marker, old, new):
    p = f"{root}/{path}"; s = open(p).read()
    if marker in s: print("already patched:", path); return
    assert s.count(old) == 1, (path, s.count(old))
    open(p, "w").write(s.replace(old, new)); print("patched:", path)

edit("skyrl/train/config/config.py", "freeze_mamba_mixer",
     "    freeze_moe_router: bool = False\n",
     "    freeze_mamba_mixer: bool = False\n"
     "    \"\"\"If True, freeze the Mamba mixer parameters (in/out proj, conv1d, A_log, D, dt_bias). Fresh starts only.\"\"\"\n"
     "    freeze_moe_router: bool = False\n")

edit("skyrl/backends/skyrl_train/distributed/megatron/megatron_utils.py", "def freeze_mamba_mixer",
     "def freeze_moe_router(model_or_models: Union[nn.Module, List[nn.Module]]):\n",
     "_MAMBA_MIXER_PARAM_RE = re.compile(r\"\\.mixer\\.(in_proj\\.weight|out_proj\\.weight|conv1d[._](weight|bias)|A_log|D|dt_bias)$\")\n\n\n"
     "def freeze_mamba_mixer(model_or_models: Union[nn.Module, List[nn.Module]]):\n"
     "    models = model_or_models if isinstance(model_or_models, list) else [model_or_models]\n"
     "    n_tensors, n_elems, kinds = 0, 0, {}\n"
     "    for model in models:\n"
     "        for name, param in model.named_parameters():\n"
     "            m = _MAMBA_MIXER_PARAM_RE.search(name)\n"
     "            if m is None:\n"
     "                continue\n"
     "            param.requires_grad = False\n"
     "            n_tensors += 1\n"
     "            n_elems += param.numel()\n"
     "            kinds[m.group(1)] = kinds.get(m.group(1), 0) + 1\n"
     "    logger.info(f\"freeze_mamba_mixer: froze {n_tensors} tensors / {n_elems} elements on this rank: {kinds}\")\n"
     "    if n_tensors == 0:\n"
     "        logger.warning(\"freeze_mamba_mixer: froze no parameters (no Mamba mixers on this rank?)\")\n"
     "    return model_or_models\n\n\n"
     "def freeze_moe_router(model_or_models: Union[nn.Module, List[nn.Module]]):\n")

edit("skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py", "freeze_mamba_mixer",
     "        # Freeze MoE router params before optimizer build.\n",
     "        # Freeze the Mamba mixers before optimizer build (local, 11-18).\n"
     "        if getattr(self.cfg.policy.megatron_config, \"freeze_mamba_mixer\", False):\n"
     "            from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import freeze_mamba_mixer\n\n"
     "            if self._rank == 0:\n"
     "                logger.info(\"freeze_mamba_mixer=True: freezing Mamba mixer params\")\n"
     "            self.provider.register_pre_wrap_hook(freeze_mamba_mixer)\n\n"
     "        # Freeze MoE router params before optimizer build.\n")
```

### A2. R3 "routes mix" fix (`skyrl/train/generators/utils.py`)

```diff
--- a/skyrl/train/generators/utils.py
+++ b/skyrl/train/generators/utils.py
@@ -319,6 +319,22 @@ def concatenate_generator_outputs(
         support_scores_expected: If True, that support must carry the sampler's member logprobs.
     """
     assert len(generator_outputs) > 0
+    # Local change (Horizon 2026-11-20, not pushed): with R3, a batch whose every row is masked (e.g. a group whose
+    # instance failed) carries no routes (harbor_skycap compose returns None when nothing trains), while the others do;
+    # that crashed the fully-async trainer ("received a mix"). Give such a batch's rows the same one-token dummy route
+    # compose gives placeholder rows (distinct experts, as the trainer's replay padding does); they train nothing.
+    routes_present = [output.get("rollout_expert_indices") is not None for output in generator_outputs]
+    if any(routes_present) and not all(routes_present):
+        ref = next(o["rollout_expert_indices"] for o in generator_outputs if o.get("rollout_expert_indices") is not None)
+        shaped = next((r for r in ref if r is not None), None)
+        if shaped is not None:
+            layers, topk = np.asarray(shaped).shape[1:]
+            dummy = np.broadcast_to(np.arange(topk, dtype=np.asarray(shaped).dtype), (1, layers, topk)).copy()
+            for output in generator_outputs:
+                if output.get("rollout_expert_indices") is None:
+                    if any(any(m) for m in output.get("loss_masks") or []):
+                        raise ValueError("a batch without routed experts has trainable tokens; refusing to pad it")
+                    output["rollout_expert_indices"] = [dummy for _ in output["response_ids"]]
     # Per-token side channels must be populated consistently across batches.
     for all_or_nothing_field in ("rollout_logprobs", "rollout_expert_indices", "rollout_sample_support"):
         present = [output.get(all_or_nothing_field) is not None for output in generator_outputs]
```

### A3. Weights-only checkpoint load when the policy LR is 0 (`skyrl/train/trainer.py`)

```diff
--- skyrl/train/trainer.py.bak1122
+++ skyrl/train/trainer.py
@@ -1919,11 +1919,17 @@
 
         # 3. Load policy checkpoint (dispatch handles offload/backload)
         logger.info(f"Loading policy checkpoint from {policy_ckpt_dir}")
+        # Local change (Lila 2026-11-22, not pushed): megatron pairs optimizer param groups by a tuple that includes
+        # max_lr / min_lr, so a checkpoint cannot be loaded into an optimizer built with another LR (KeyError in
+        # distrib_optimizer.load_state_dict). A measurement-only resume at lr 0 updates nothing: load the weights only.
+        _weights_only = float(self.cfg.trainer.policy.optimizer_config.lr) == 0.0
+        if _weights_only:
+            logger.warning("policy lr is 0: loading model weights only (optimizer / scheduler state skipped)")
         self.dispatch.load_checkpoint(
             "policy",
             policy_ckpt_dir,
-            load_optimizer_states=True,
-            load_lr_scheduler_states=True,
+            load_optimizer_states=not _weights_only,
+            load_lr_scheduler_states=not _weights_only,
         )
         logger.info("Successfully loaded policy checkpoint")
 
```

### A4. Refresh the optimizer master weights after a weights-only load (`skyrl/backends/skyrl_train/workers/worker.py`)

```diff
--- skyrl/backends/skyrl_train/workers/worker.py.bak1122
+++ skyrl/backends/skyrl_train/workers/worker.py
@@ -721,6 +721,15 @@
             load_optimizer_states=load_optimizer_states,
             load_lr_scheduler_states=load_lr_scheduler_states,
         )
+        # Local change (Lila 2026-11-22, not pushed): a Megatron optimizer keeps its own fp32 master copy of the
+        # weights (built from the base model) and writes it back into the model at every step. When the optimizer
+        # state is not loaded, refresh that copy from the loaded weights, as Megatron's own load_checkpoint does
+        # (optimizer.reload_model_params()); otherwise the first step, even at lr 0, resets the model to base.
+        if not load_optimizer_states and getattr(self, "optimizer", None) is not None:
+            _reload = getattr(self.optimizer, "reload_model_params", None)
+            if _reload is not None:
+                _reload()
+                print("[weights-only load] optimizer master weights reloaded from the loaded model weights", flush=True)
         return states
 
     def save_hf_model(self, export_dir: str, tokenizer):
```

### A5. Step-loop bound on resume into a smaller dataset (`skyrl/train/fully_async_trainer.py`)

```diff
--- skyrl/train/fully_async_trainer.py.bak1122
+++ skyrl/train/fully_async_trainer.py
@@ -540,7 +540,15 @@
                 # end an epoch early, drifting global_step out of epoch alignment). On resume the dataloader
                 # already reflects this epoch's trained steps. The range below is just an upper bound.
                 trained_steps_this_epoch = self.async_train_dataloader.num_trained() // self.mini_batch_size
-                for _step_idx in range(self.global_step, (1 + epoch) * self.num_steps_per_epoch + 1):
+                # Local change (Lila 2026-11-22, not pushed): a resume onto a dataset with fewer steps per epoch
+                # (held-out measurement pass) starts with global_step already above the nominal bound, and the
+                # epoch would end without a step. Allow this epoch's remaining steps from the resumed step.
+                # No-op for a normal run (there global_step + remaining <= the nominal bound).
+                _step_upper = max(
+                    (1 + epoch) * self.num_steps_per_epoch + 1,
+                    self.global_step + self.num_steps_per_epoch - trained_steps_this_epoch,
+                )
+                for _step_idx in range(self.global_step, _step_upper):
                     async with self._step_deadline(), Timer("step", self.all_timings):
                         self._loop_gauges.set(
                             "skyrl_gen_buffer_qsize",
```

### A6. Placement-group strategy switch (`skyrl/backends/skyrl_train/workers/worker.py`)

```diff
diff --git a/skyrl/backends/skyrl_train/workers/worker.py b/skyrl/backends/skyrl_train/workers/worker.py
index 94798abd..948e62e9 100644
--- a/skyrl/backends/skyrl_train/workers/worker.py
+++ b/skyrl/backends/skyrl_train/workers/worker.py
@@ -844,7 +844,9 @@ class PPORayActorGroup:
                 for i in range(len(bundles)):
                     bundles[i][resources_name] = self._num_resources_per_node
 
-            raw_pg = placement_group(bundles, strategy="PACK")
+            # SKYRL_POLICY_PG_STRATEGY=STRICT_SPREAD puts each per-node bundle on its own node (local, 11-19: with
+            # PACK both 4-GPU bundles of a "2-node" trainer landed on one node, whose host memory then ran out).
+            raw_pg = placement_group(bundles, strategy=os.environ.get("SKYRL_POLICY_PG_STRATEGY", "PACK"))
             get_ray_pg_ready_with_timeout(raw_pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)
 
         def _scheduling_strategy_for_rank(rank):
```

### A7. Launcher changes in the working tree (`run_fully_async_terminus.sh`; includes the `MODEL_FAMILY=nemotron-3.5` handling used by `run_rl_nem.sh`)

```diff
diff --git a/examples/train_integrations/harbor_skycap/run_fully_async_terminus.sh b/examples/train_integrations/harbor_skycap/run_fully_async_terminus.sh
index 21460e6d..b0c23275 100644
--- a/examples/train_integrations/harbor_skycap/run_fully_async_terminus.sh
+++ b/examples/train_integrations/harbor_skycap/run_fully_async_terminus.sh
@@ -26,6 +26,37 @@ HERE="$REPO/examples/train_integrations/harbor_skycap"
 cd "$REPO"
 
 MODEL="${MODEL:?set MODEL to the model checkout}"
+# MODEL_FAMILY=qwen3.5 (default) or nemotron-3.5 (Nemotron-3.5-Lightning: hybrid Mamba2 / attention / MoE, sigmoid
+# router with an expert bias; Megatron TP 4 / EP 8 with CPU optimizer offload; no R3 unless R3=1).
+MODEL_FAMILY="${MODEL_FAMILY:-qwen3.5}"
+case "$MODEL_FAMILY" in
+  qwen3.5)
+    FAMILY_ARGS=(
+      trainer.policy.language_model_only=true
+      trainer.ref.language_model_only=true
+      generator.inference_engine.language_model_only=true
+      generator.inference_engine.engine_init_kwargs.gdn_prefill_backend=triton
+    ) ;;
+  nemotron-3.5)
+    : "${SERVED_MODEL_NAME:=nemotron-3.5-lightning}" "${TOKENIZER:=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}"
+    : "${MEGATRON_TP:=4}" "${MEGATRON_EP:=8}"
+    R3_VALUE=false; [[ "${R3:-0}" == 1 ]] && R3_VALUE=true
+    FAMILY_ARGS=(
+      trainer.policy.megatron_config.moe_token_dispatcher_type=alltoall
+      trainer.policy.megatron_config.moe_router_load_balancing_type=none
+      trainer.policy.megatron_config.moe_grouped_gemm=true
+      trainer.policy.megatron_config.moe_router_score_function=sigmoid
+      trainer.policy.megatron_config.moe_router_enable_expert_bias=true
+      trainer.policy.megatron_config.moe_enable_routing_replay=$R3_VALUE
+      generator.inference_engine.enable_return_routed_experts=$R3_VALUE
+      trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=true
+      trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=1.0
+      trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=true
+      trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=true
+      trainer.mtp.enabled=false
+    ) ;;
+  *) echo "unknown MODEL_FAMILY=$MODEL_FAMILY (qwen3.5 or nemotron-3.5)" >&2; exit 1 ;;
+esac
 SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b}"
 TOKENIZER="${TOKENIZER:-Qwen/Qwen3.5-9B}"
 TASKS_DIR="${TASKS_DIR:?set TASKS_DIR to the directory of Harbor training tasks}"
@@ -39,6 +70,7 @@ POLICY_NUM_NODES="${POLICY_NUM_NODES:-1}"
 NUM_ENGINES="${NUM_ENGINES:-12}"
 TP_SIZE="${TP_SIZE:-2}"
 MEGATRON_TP="${MEGATRON_TP:-2}"
+MEGATRON_EP="${MEGATRON_EP:-1}"
 GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
 MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
 
@@ -119,13 +151,11 @@ echo "==> $RUN_NAME: $(find -L "$TASKS_DIR" -mindepth 1 -maxdepth 1 -type d | wc
   trainer.strategy=megatron \
   trainer.placement.colocate_all=false \
   trainer.placement.policy_num_nodes="$POLICY_NUM_NODES" \
-  trainer.placement.policy_num_gpus_per_node=8 \
-  trainer.policy.language_model_only=true \
-  trainer.ref.language_model_only=true \
+  trainer.placement.policy_num_gpus_per_node="${POLICY_GPUS_PER_NODE:-8}" \
   trainer.policy.megatron_config.tensor_model_parallel_size="$MEGATRON_TP" \
   trainer.policy.megatron_config.pipeline_model_parallel_size=1 \
   trainer.policy.megatron_config.context_parallel_size=1 \
-  trainer.policy.megatron_config.expert_model_parallel_size=1 \
+  trainer.policy.megatron_config.expert_model_parallel_size="$MEGATRON_EP" \
   trainer.policy.megatron_config.expert_tensor_parallel_size=1 \
   trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity=full \
   trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=uniform \
@@ -174,8 +204,7 @@ echo "==> $RUN_NAME: $(find -L "$TASKS_DIR" -mindepth 1 -maxdepth 1 -type d | wc
   generator.inference_engine.num_engines="$NUM_ENGINES" \
   generator.inference_engine.tensor_parallel_size="$TP_SIZE" \
   generator.inference_engine.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
-  generator.inference_engine.language_model_only=true \
-  generator.inference_engine.engine_init_kwargs.gdn_prefill_backend=triton \
+  "${FAMILY_ARGS[@]}" \
   generator.inference_engine.engine_init_kwargs.max_model_len="$MAX_MODEL_LEN" \
   generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
   generator.inference_engine.router_init_kwargs.policy=sticky_least_loaded \
```

### A8. The Nemotron full-run launcher `ops/calibforge_rl_1110/run_rl_nem.sh` (Lila)

```bash
#!/usr/bin/env bash
# CalibForge v2 RL, Nemotron-3.5-Lightning, FULLY ASYNC full run, 2026-11-18 (SkyRL run_fully_async_terminus.sh with
# MODEL_FAMILY=nemotron-3.5). Derived from run_rl.sh (the Qwen3.5-9B fully async launcher) with the Nemotron parts of
# run_overfit_nem.sh (model path, MODEL_FAMILY, Ray memory threshold, engine health timeout). Same agent and data
# handling as run_rl.sh: in-sandbox Terminus 2 through the sandbox relay, agent timeout 3600 s, training context 131072
# (engines serve 262144), one reply capped at 16,384 tokens, no summarization, thinking kept, final path, timeouts and
# out-of-context endings trained as reward 0, DPPO 0.15 / prompt_mean / no std norm / zero-variance filter (the repo
# launcher's recipe; change the loss with overrides). Defaults that differ from run_rl.sh:
#   - PROVIDER=daytona (the shared Daytona RL org: base snapshot + setup script, preview relay),
#     TASKS_DIR=$D/data/calibforge_v2_daytona; the Qwen run qwen35-cf-v2-daytona-b32-lr2e-6-1115 trained on the cleaned
#     set $D/data/calibforge_v2_daytona3 (2,598 tasks: 8 that cannot work there dropped, setup limit >= 1800 s);
#   - batch 32 x group 16, staleness 3, MAX_CONCURRENCY 900 and 3 trial starts / s (that Qwen run's choices for the
#     shared org);
#   - trainer on one node (8 GPUs, Megatron TP 4 / EP 8, CPU optimizer offload) + NUM_ENGINES=12 vLLM engines x TP 2
#     on the other three nodes; R3 off unless R3=1;
#   - no evals during training (EVAL_INTERVAL=0, EVAL_BEFORE_TRAIN=false; TB2.1 cannot run on the RL org);
#   - checkpoints every 5 steps, the last 4 kept (a Nemotron checkpoint with optimizer state is ~0.4 TB);
#   - LR 4.0e-6.
# Run ON a GPU worker (the driver node); the driver joins the Anyscale Ray cluster (never `ray stop` it).
#
#   cd $D && setsid nohup bash ops/calibforge_rl_1110/run_rl_nem.sh <run-name> [overrides...] &> logs/<run-name>.log < /dev/null &
set -euo pipefail
D=/mnt/local_storage/charlie
# Run from a snapshot: bash reads a script as it goes, so editing (or rsyncing) this file would break a running job.
if [[ -z "${RL_SNAPSHOT:-}" ]]; then
  mkdir -p $D/runs/$1; cp "$0" $D/runs/$1/launcher.sh
  cp $D/SkyRL/examples/train_integrations/harbor_skycap/run_fully_async_terminus.sh $D/runs/$1/run_fully_async_terminus.sh
  RL_SNAPSHOT=1 exec bash $D/runs/$1/launcher.sh "$@"
fi
export RUN_NAME=${1:?run name}
export RUN_DIR=$D/runs/$RUN_NAME
export RAY_ADDRESS=auto
export MODEL=${MODEL:-$D/hf/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16} MODEL_FAMILY=nemotron-3.5
# Host memory: the trainer with CPU optimizer offload takes ~1.4 TB of the node's 2 TB; Ray's default OOM threshold
# (0.95) killed drivers on JUPITER.
export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.98}
export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=${SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S:-1800}
# PROVIDER=dune: each task's own prebuilt image, exec relay. PROVIDER=daytona (default here): the shared Daytona org of
# 11-14 (no image builds, 5 snapshots): tasks converted to "base snapshot + setup script" (make_daytona_tasks.py,
# harbor_skycap/daytona_setup.py), relay over signed preview URLs; TB2.1 cannot run there, so set EVAL_INTERVAL=0.
export PROVIDER=${PROVIDER:-daytona}
if [[ "$PROVIDER" == daytona ]]; then
  export TASKS_DIR=${TASKS_DIR:-$D/data/calibforge_v2_daytona}
  ENV_CLASS=examples.train_integrations.harbor_skycap.daytona_setup:SetupRelayDaytonaEnvironment
  RELAY_TRANSPORT=preview
elif [[ "$PROVIDER" == daytona_eval ]]; then
  # The second Daytona org (11-14, $D/daytona_eval_keys.env): image builds allowed, 4 vCPU / 8 GiB / 20 GiB fine, so
  # tasks run from their own images as on Dune; relay over signed preview URLs. Used for TB2.1 evals.
  ENV_CLASS=examples.train_integrations.harbor_skycap.sandbox_relay.environment:RelayDaytonaEnvironment
  RELAY_TRANSPORT=preview
else
  ENV_CLASS=examples.train_integrations.harbor_skycap.sandbox_relay.exec_environment:ExecRelayDaytonaEnvironment
  RELAY_TRANSPORT=exec
fi
export TASKS_DIR=${TASKS_DIR:-$D/data/calibforge_v2_train48}
export EVAL_TASKS_DIR=${EVAL_TASKS_DIR:-$D/data/tbench21_4cpu}
export LABELS_OWNER=charlieruan
export XDG_CACHE_HOME=$D/.cache UV_CACHE_DIR=/home/ray/.cache/uv
export VENV=${VENV:-$D/venvs/upstream-megatron}
# torch 2.13 needs NCCL 2.29; the image's /usr/lib libnccl (2.28.9) shadows the wheel's. EFA libs for multi-node.
export LD_LIBRARY_PATH=$D/nccl/pkg/nvidia/nccl/lib:/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export SKYRL_LD_LIBRARY_PATH_EXPORT=1
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 NCCL_DEBUG=WARN
# Anyscale's raylet env points W&B at its hosted server; ours is wandb.ai.
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai} WANDB_ENTITY=${WANDB_ENTITY:-sky-posttraining-uc-berkeley}
export PROJECT=${PROJECT:-skycap-calibforge-fully-async}
export RAY_enable_ray_event=0 RAY_task_events_report_interval_ms=0
# Sandboxes on Dune (Daytona's API at DAYTONA_API_URL; forwarded to the Ray workers by skyrl/train/utils/utils.py).
umask 077
if [[ "$PROVIDER" == daytona ]]; then
  { grep -v '^export DAYTONA_API' $D/keys.env; cat $D/daytona_keys.env; } > "$RUN_DIR/keys.env"
elif [[ "$PROVIDER" == daytona_eval ]]; then
  { grep -v '^export DAYTONA_API' $D/keys.env; cat $D/daytona_eval_keys.env; } > "$RUN_DIR/keys.env"
else
  cat $D/keys.env $D/dune_keys.env > "$RUN_DIR/keys.env"
fi
umask 022
export KEYS_FILE="$RUN_DIR/keys.env"
set -a; . "$KEYS_FILE"; set +a
export HF_TOKEN=${HF_TOKEN:-$HF_API_KEY}
ulimit -n 524288 || true

# The recipe's knobs (run_fully_async_terminus.sh).
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32} GROUP_SIZE=${GROUP_SIZE:-16} MAX_STALENESS_STEPS=${MAX_STALENESS_STEPS:-3}
# Shared Daytona RL org: 900 trials in flight and 3 starts / s, as qwen35-cf-v2-daytona-b32-lr2e-6-1115 (11-15).
export MAX_CONCURRENCY=${MAX_CONCURRENCY:-900} TRAJECTORIES_PER_SECOND=${TRAJECTORIES_PER_SECOND:-3}
# 12 engines x TP 2 on three nodes; the trainer (Megatron TP 4 / EP 8, set by MODEL_FAMILY) takes the fourth.
export NUM_ENGINES=${NUM_ENGINES:-12}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}          # what the trainer takes; the engines' limit is set below
export TRAIN_PATHS=final SUMMARIZE=false TRAIN_ON_TIMEOUTS=true
# No evals during training (the trainers test eval_interval > 0).
export EVAL_INTERVAL=${EVAL_INTERVAL:-0} EVAL_BEFORE_TRAIN=${EVAL_BEFORE_TRAIN:-false} EVAL_SAMPLES=${EVAL_SAMPLES:-3}
export CKPT_INTERVAL=${CKPT_INTERVAL:-5} MAX_CKPTS=${MAX_CKPTS:-4}
export EPOCHS=${EPOCHS:-3} MAX_TURNS=${MAX_TURNS:-250} LR=${LR:-4.0e-6}
EVAL_MODEL_LEN=${EVAL_MODEL_LEN:-262144}
AGENT_TIMEOUT=${AGENT_TIMEOUT:-3600}
# Eval-only agent kwargs. Summarization stays off in eval too (user, 11-10: follow the friend's setup).
EVAL_AGENT_KWARGS="${EVAL_AGENT_KWARGS:-{enable_summarize: false}}"
# One reply's token cap, train and eval (user, 11-10: the friend's run caps a turn at 16k). Terminus is told the same
# number (model_info.max_output_tokens), which it quotes when it asks the model to re-issue a truncated reply.
MAX_TOKENS_PER_TURN=${MAX_TOKENS_PER_TURN:-16384}
# Eval sampling = the training sampling (1.0 / 1.0, top_k and min_p off), as Mercor's recipe does (user, 11-11: "the
# training sampling parameter for eval"). The repo launcher's own default is Qwen3.5's 0.6 / 0.95 / top_k 20, under
# which the model loops more and more as training goes on (35% -> 45% of eval trials in 75 steps). Spelled out as a
# mapping (a `null` would reach the entrypoint as a string).
EVAL_SAMPLING_TRAINLIKE='{temperature: 1.0, top_p: 1.0, top_k: -1, min_p: 0.0}'
export EVAL_SAMPLING="${EVAL_SAMPLING:-$EVAL_SAMPLING_TRAINLIKE}"
# RESUME_PATH=<run>/ckpts/global_step_N resumes from another run's checkpoint (weights, optimizer, which tasks were
# consumed); this run then writes its own checkpoints. With EVAL_BEFORE_TRAIN the resumed step is evaluated first.
RESUME_ARGS=()
if [[ -n "${RESUME_PATH:-}" ]]; then
  export RESUME_MODE=from_path
  RESUME_ARGS=(trainer.resume_path="$RESUME_PATH")
fi

# The integration's opt-ins: one-call exec (the in-sandbox runner's long-poll), the Ctrl-C guard, a verifier whose
# tests never ran is retried, per-trial timing events and the event loop's lag.
export SKYCAP_FAST_EXEC=1 SKYCAP_TERMINAL_GUARD=1
# Harbor's tar / gzip of uploaded and downloaded directories in a thread pool instead of on the event loop (it froze
# every trial for up to 376 s), and no re-upload of the in-sandbox agent's own logs (loop_offload.py).
export SKYCAP_OFFLOAD_TAR=1 SKYCAP_SKIP_LOG_REUPLOAD=1
export SKYCAP_VERIFIER_RETRIES=${SKYCAP_VERIFIER_RETRIES:-3}
export SKYCAP_TIMING=1 SKYCAP_TIMING_DIR="$RUN_DIR/timing"
# Backstop per trial attempt: the agent's 3600 s + sandbox build + install + the verifier.
export SKYCAP_TRIAL_TIMEOUT_SEC=${SKYCAP_TRIAL_TIMEOUT_SEC:-6600}
# A provider outage pauses trial starts instead of failing every trial and training on empty batches
# (outage_guard.py; added after the 2 h Dune 503 outage of 11-14). SKYCAP_OUTAGE_GUARD=0 turns it off.
export SKYCAP_OUTAGE_GUARD=${SKYCAP_OUTAGE_GUARD:-1}
# An agent timeout with one model call stuck longer than this is an infrastructure failure, not a reward-0
# rollout (11-14: a lost relay message left 108 trials waiting ~48 min on one call). 0 or empty turns it off.
export SKYCAP_STALL_TIMEOUT_SEC=${SKYCAP_STALL_TIMEOUT_SEC-900}
# Snapshot org only (sandboxes start from ready snapshots in seconds): give up on one creation after 120 s so that
# Harbor's retry creates another sandbox (11-15: one broken Daytona runner held ~3% of creates for ~900 s each).
# Needs harbor_skycap/daytona.py of 11-15 on the workers. Not for orgs where a creation builds / pulls an image.
if [[ "$PROVIDER" == daytona ]]; then export SKYCAP_CREATE_TIMEOUT_SEC=${SKYCAP_CREATE_TIMEOUT_SEC-120}; fi
mkdir -p "$RUN_DIR/timing"

export REPO=$D/SkyRL
cd $D/SkyRL
exec bash $RUN_DIR/run_fully_async_terminus.sh \
  generator.inference_engine.engine_init_kwargs.max_model_len="$EVAL_MODEL_LEN" \
  skycap.train_max_model_len="$MAX_MODEL_LEN" \
  "skycap.eval_agent_kwargs=$EVAL_AGENT_KWARGS" \
  harbor_trial_config.agent.kwargs.model_info.max_input_tokens="$EVAL_MODEL_LEN" \
  harbor_trial_config.agent.kwargs.model_info.max_output_tokens="$MAX_TOKENS_PER_TURN" \
  skycap.max_tokens_per_turn="$MAX_TOKENS_PER_TURN" \
  harbor_trial_config.agent.override_timeout_sec="$AGENT_TIMEOUT" \
  harbor_trial_config.agent.import_path=examples.train_integrations.harbor_skycap.terminus_installed:TerminusInstalled \
  harbor_trial_config.environment.import_path="$ENV_CLASS" \
  harbor_trial_config.environment.kwargs.connection_pool_maxsize=4000 \
  skycap.sandbox_relay.enabled=true \
  skycap.sandbox_relay.transport="$RELAY_TRANSPORT" \
  "${RESUME_ARGS[@]}" \
  "${@:2}"
```

---

## Appendix B. Horizon's final table (written by the Horizon session, 2026-11-23 23:25 CST, after the kill)

Copied as received. Sources on Horizon: `$SCRATCH/calibforge/FINDINGS.md`, `PLAN.md`, `runs/<name>/`, W&B; the same
text is saved there as `$SCRATCH/calibforge/FINAL_SUMMARY_HORIZON.md`.

- W&B: `https://wandb.ai/sky-posttraining-uc-berkeley/<project>/runs/<id>`. Project `skycap-calibforge-fully-async`,
  except where `horizon-smoke` is given. Several ids per run are relaunches or resumes of the same run name, in order.
- HF repo: `CharlieFRuan/nemotron35-calibforge-ckpts/<run>/global_step_<n>`. Each is a Megatron checkpoint of 18
  files, 442.5 GB; 48 files where marked.
- Local checkpoints: `$SCRATCH/calibforge/runs/<run>/ckpts/global_step_<n>` holds the last 2-3 per-step saves;
  `.../ckpt_keep/global_step_<n>` holds the kept steps (hard links). Nothing was deleted at the wind-down.
- Dates are CST. TB2.1 eval setup: 89 tasks x 3 trials, temperature 1, no summarization, 16,384 tokens per reply,
  3,600 s per trial, Dune sandboxes.
- "vs base" is the per-task mean difference against the pool of 3 base evals (1116b, 1119c, 1121d; avg 0.2372, 801
  samples), +- SE over 89 tasks, recomputed uniformly; it can differ from older log lines by 0.004 or less.
- Common main-run recipe unless the row says otherwise: CalibForge v2 train set (`calibforge_v2_train48`, 2,788
  tasks), seed 42, batch 32 groups x 16 rollouts; fully-async GRPO, staleness 3, max_concurrency 1024, Terminus-2
  agent; "full-run loss" with `loss_reduction=prompt_mean`; rollout_is = rollout importance sampling, ratio clipped
  to [0.5, 5]; frz = router expert-bias updates off plus LR 0 on the router and the Mamba mixers
  (`lr_zero_modules=[router,mamba]`); R3 = routing replay (vLLM routes replayed in Megatron).

### B1. Main CalibForge runs (full task set)

| Run (W&B ids) | Dates | Recipe | Trained steps / last ckpt | Kept / HF | Why it ended | Training-side readout | Entropy range | Gap range |
|---|---|---|---|---|---|---|---|---|
| `nemo35-cf-v2-hzn-b32-lr2e-6-1116` "first full run" (rm1l3c7w, m1z8g0s7) | 11-16 -> 11-17 | LR 2e-6, full-run loss, no rollout_is, no freeze, no R3 | 89 logged / ckpt 85 | ckpts 5...85 every 5 local; HF 25, 50 (48 files), 55, 75, 85 (48 files) | Job 96212 ended 11-17 ~20:30 and auto-resume was disarmed because of entropy drift (owner's call); not resumed | Reward flat (first-5 0.390, last-5 0.411); TB2.1 flat at steps 25 and 50 | 0.051 -> 0.418 (7x rise from ~step 35) | 0.025-0.053 |
| `nemo35-cf-v2-hzn-b32-ris-nolb-from25-1118` (8l96k8ug, qtyobckj) | 11-18 | Resume test from the full run's step 25: rollout_is, bias updates off | 26-27 / none | - | Turned into the candidate (next row) | - | 0.055-0.057 | 0.031-0.032 |
| `nemo35-cf-v2-hzn-b32-cand-from25-1118` "candidate" (vmw1j8q5, xvfdrtud) | 11-18 -> 11-20 | From the full run's step 25; LR 2e-6, rollout_is + frz, prompt_mean | 26-80 / ckpt 80 (checkpoint interval 2) | keep 40, 50, 55, 60, 70, 80; HF 50 | Stopped after checkpoint 80 (11-20 00:4x) so R3 could have its job | Entropy held at base; reward over 26-50 = original (0.455 vs 0.463) | 0.052-0.081 | 0.027-0.034 |
| `nemo35-cf-v2-hzn-b32-lr4e-6-ris-frz-1118` "run H" (xfqock0w, djoi3k9a, exnyj3oq) | 11-19 | From base; LR 4e-6, rollout_is + frz, prompt_mean | 25 / ckpt 25 | keep 5-25; HF 25 | Stopped after 25 to branch at a lower LR (plan) | Paired vs the 2e-6 original: +0.042 +- 0.018 at 16-20, +0.082 +- 0.037 at 21-25. The entropy rise is in the thinking tokens | 0.054-0.148 | 0.042-0.063 |
| `nemo35-cf-v2-hzn-b32-lr4to2e-6-ris-frz-from25-1119` "branch" (86b728if, pgbnf8ff, oeab12rn, 0ndsebsn, 62cq4bzm) | 11-19 | From H 25; LR 2e-6, otherwise as H | 26-35 / ckpt 35 | keep 35 | Entropy kept rising at 2e-6 (0.23 -> 0.44); switched to sequence_mean | - | 0.231-0.437 | 0.063-0.071 |
| `nemo35-cf-v2-hzn-b32-lr2e-6-ris-frz-seqmean-from35-1119` (6jkz47g1) | 11-19 | From branch 35; LR 2e-6, `loss_reduction=sequence_mean` | 36-40 / ckpt 40 | keep 40 | Stopped after 40 for the score-centering arm; entropy 0.38-0.57 | - | 0.375-0.571 | 0.063-0.067 |
| `nemo35-hzn-sc-smoke-1119` (s5dyorek) | 11-19 | 4 x 4 short tasks, score centering, 1 step | 1 / none | - | Smoke test passed | - | 0.049 | 0.041 |
| `nemo35-cf-v2-hzn-b32-lr4e-6-ris-frz-sc-1119` "sc" (d8mygg6r, pv20y7eg) | 11-19 -> 11-20 | From base; LR 4e-6, rollout_is + frz + top_k 20 + `enable_score_centering`, prompt_mean | 50 / ckpt 50 | keep 5...50 every 5; HF 25 | Planned stop at 50; its job was handed to R3 (11-20 17:27) | Entropy 0.058 -> 0.065 vs H 0.059 -> 0.148 at the same reward; task-matched vs candidate -0.005 +- 0.004 (control) | 0.054-0.072 | 0.043-0.053 |
| `nemo35-cf-v2-hzn-b32-lr4e-6-ris-frz-r3-1119` "R3" (4y0fwmpi, 0fwsmoej, i1zdvjex, o3kh9qob, e4rry6wb, 4l3p91mx, boqv0qt8, xidu3dwc, obk1ztuj, 4knu9log, 9vewn9vc, hwnic1cp) | 11-20 -> 11-22 | From base; LR 4e-6, rollout_is + frz + R3, prompt_mean | 113 logged / ckpt 114 (2 epochs of ~57 steps) | keep 5...110 every 5, plus 57 and 114; HF 75 | Finished 2 epochs (job 100553 ended 11-22 05:10); not relaunched | Task-matched vs the 2e-6 lineage: 0 through 15, +0.015 +- 0.008 (16-20), then ~+0.036-0.04, flat from 26 to 112. Revisit +0.027 +- 0.005 (765 tasks). Training context share 17-21% | 0.051-0.130 (base-level to ~65, ~0.12 at 104-112) | 0.0107-0.0158 |
| `nemo35-cf-v2-hzn-b32-lr4e-6-ris-frz-r3-seqmean-1122` "sequence_mean twin" (zquith8x, q0vng9ym, mll4u9z2, fzaub650, gsdrl4sf, zz7ory1z, b222v5l8, sc63sko6) | 11-22 -> 11-23 | R3's recipe with `loss_reduction=sequence_mean`; same tasks, seed and order; from base | 58 / ckpt 58 (epoch 1 ended at 57) | keep 5...55 every 5, plus 57; local 56, 57, 58; not on HF | **Kill** (owner, 11-23): pending job 102064 cancelled at step 58 | vs 2e-6 lineage +0.056 +- 0.010 at 26-30; vs R3 task-matched -0.013 / +0.007 / +0.012 (16-30) and -0.010 / -0.009 / +0.019 (31-45), i.e. level. Training context share 0.21-0.29 (last full windows 41-45 0.232, 46-50 0.262, 51-55 0.277). Longer outputs (+3-8k tokens per trial), not more turns | 0.055-0.077 | 0.0119-0.0163 |
| sub512 (512-task repeat-visit arm, R3 + sequence_mean, 6 epochs) | 11-22 | Data built: `data/calibforge_v2_sub512_hzn` (512 tasks) | **never trained** | - | Replaced by the full-set twin before it launched | - | - | - |
| `nemo35-cf-v2-hzn-b32-lr2e-6-octloss-1117` | 11-17 | Prepared: full-run launcher with the October loss block | **never launched** (no W&B run, no ckpt) | - | Superseded by the overfit loss tests | - | - | - |

### B2. Overfit and diagnostic arms (16 short tasks x 8 unless noted; sync trainer; from base)

The October loss block is regular PPO clip 0.2 / 0.28, GRPO std normalization, token_mean, token TIS 2. Full-run loss
is the first full run's loss. Reward is the mean over the first 5 and last 5 steps.

| Run (W&B ids) | Date | Knobs | Steps / last ckpt | Reward first-5 -> last-5 | Entropy | Gap | Outcome |
|---|---|---|---|---|---|---|---|
| `nemo35-gsm8k-hzn-1116a` (horizon-smoke z220jl0b, jmjsj6vq) | 11-16 | gsm8k smoke | 10 / - | - | - | 0.019 (step 1) | Stack works on GB200 |
| `nemo35-gsm8k-hzn-ibtest-1116` (horizon-smoke l5b622sw) | 11-16 | IPoIB / IB HCA test | 1 / - | - | - | - | Weight sync 29-36 s -> 2.3-2.9 s |
| `nemo-hzn-devcheck-1116` (horizon-smoke cozv2o6t) | 11-16 | Dev check | - | - | - | - | Infra check |
| `nemo-hzn-of32-lr1e-5-1116` (z1oqhr67) | 11-16 | 32 tasks, LR 1e-5 | 1 / 1 | 0.531 | 0.053 | 0.0418 | Gap diagnosis |
| `nemo-hzn-gapdiag-nocache-1116` (ripvdiwq) | 11-16 | Prefix cache off | 1 / 1 | 0.545 | 0.057 | 0.0445 | Prefix cache is not the cause of the gap |
| `nemo-hzn-gapdiag-r3-1116` (oljzfk7j) | 11-16 | R3 | 1 / 1 | 0.570 | 0.052 | 0.0108 | R3 cuts the gap from 0.042 to 0.011 |
| `nemo-hzn-mimo-of14x8-1116`, `...b`-`...e` (r2ayc6on, j6knp51x, ua8p5gps, m5u60v35, 94df7nzd) | 11-16 | MiMo tasks 14 x 8, setup iterations | 1 each | - | 0.03-0.04 | ~0.010 | Infra / setup only |
| `nemo-hzn-of16x8-lr5e-6-1116` (31l5yudb) | 11-16 | Full-run loss, LR 5e-6 | 14 / 14 | 0.570 -> 0.608 | 0.050-0.057 | 0.030-0.042 | No learning |
| `nemo-hzn-of16x8-lr5e-6-r3-1116` (luokdz0t) | 11-17 | Same + R3 | 6 / 6 | 0.622 -> 0.613 | 0.049-0.057 | 0.010-0.011 | No learning |
| `nemo-hzn-of16x8-oct-nor3-shorttasks-1117` (fsxb87yk) | 11-17 | October recipe, no R3 | 20 / 20 | 0.573 -> 0.684 | 0.050-0.059 | 0.041-0.046 | Learns |
| `nemo-hzn-of16x8-oct-r3-lilatasks-1117` (ddnr2iad) | 11-17 | October recipe + R3, harder tasks | 20 / 20 | 0.364 -> 0.445 | 0.043-0.066 | 0.009-0.013 | Learns |
| `nemo-hzn-of16x8-octloss-fullrundata-1117` (w7m4krh6) | 11-17 | October loss block + full-run data handling | 14 / 14 | 0.630 -> 0.716 | 0.048-0.088 | 0.038-0.056 | Learns: the loss block is the lever |
| `nemo-hzn-of16x8-fullloss-octdata-1117` (0zu3fhxy) | 11-17 | Full-run loss + October data handling | 7 / 7 | 0.595 -> 0.559 | 0.052-0.056 | 0.036-0.044 | Flat |
| `nemo-hzn-of16x8-fullloss-stdnorm-1117` (uruyjckr) | 11-17 | Full-run loss + std normalization | 14 / 14 | 0.569 -> 0.602 | 0.051-0.061 | 0.030-0.044 | Small or no gain |
| `nemo-hzn-of16x8-fullloss-tokenmean-1117` (pqa4bfpw) | 11-17 | Full-run loss + token_mean | 14 / 14 | 0.544 -> 0.584 | 0.051-0.063 | 0.025-0.042 | Small or no gain |
| `nemo-hzn-of16x8-fullloss-lr2e-6-1117` (w01de9xb) | 11-18 | Full-run loss, LR 2e-6 | 12 / 12-13 | 0.570 -> 0.600 | 0.052-0.058 | 0.039-0.046 | No learning |
| `nemo-hzn-of16x8-fullloss-lr1e-5-1117` (o5wgxfqm) | 11-18 | Full-run loss, LR 1e-5 | 12 / 12-13 | 0.589 -> 0.534 | 0.050-0.154 | 0.014-0.043 | Entropy 3x in 12 steps with reward falling: fast reproducer of the full run's drift |
| `nemo-hzn-of16x8-ris-lr4e-6-1118` "A" (eorb4w6p, ro1azoqq, mbksnvy9) | 11-18 | rollout_is [0.5,5], LR 4e-6 | 9 / 9 | 0.567 -> 0.642 | 0.045-0.057 | 0.036-0.045 | Learns |
| `nemo-hzn-of16x8-dppo-lr4e-6-1118` "DPPO" (677npo93) | 11-18 | DPPO (TV 0.15 mask), LR 4e-6 | 10 / 10 | 0.586 -> 0.636 | 0.048-0.056 | 0.037-0.043 | Grad norm ~3x rollout_is's |
| `nemo-hzn-of16x8-ris-frozenrouter-lr4e-6-1118` "C" (e3d5q0f2) | 11-18 | rollout_is + frozen router | 7 / 7 | 0.589 -> 0.600 | 0.052-0.059 | 0.041-0.046 | Learns about like A (5-7 mean 0.607) |
| `nemo-hzn-of16x8-ris-mamba-router-frozen-lr4e-6-1118` "M" (f4g5lzlz, jioqglao, zcfrbhz9) | 11-18 -> 11-19 | rollout_is + frozen Mamba + router + bias 0 | 30 / 30 (kept) | 0.591 -> 0.831 | 0.045-0.059 | <=0.043 | Best: learns most with entropy at base -> the main-run recipe |
| `nemo-hzn-of16x8-dppo-frozenrouter-lr4e-6-1118` "B" (0rf6c8nt, hwaet18t, swtlnru6, 2fysn45d) | 11-18 -> 11-19 | DPPO + frozen router + bias 0 (Mamba training) | 30 / 30 | 0.580 -> 0.647 | 0.052-0.092 | 0.023-0.045 | Learns less, entropy rises |

### B3. TB2.1 evals (89 x 3)

Endings are complete / context exceeded / agent timeout / other, % of trials. Cap is the share of calls, and of
trials, with a reply cut at 16,384 tokens. Speed is the median per-call output tok/s (calls of >= 100 tokens), then
overall output tok/s.

| Eval run (W&B id) | Policy | Avg | Pass@3 | n trials | Endings % | Cap calls % / trials % | Speed | Per task vs base pool |
|---|---|---|---|---|---|---|---|---|
| `nemo-hzn-base-eval-1116a` (h46fkgwl, efcb1l8t) | base | 0.2397* | 0.4045* | 266 | 56.0 / 0.0 / 44.0 / 0.0 | 1.00 / 22.2 | 25 / 26 | **invalid**: CPU-pinned engines (117 timeouts); not used |
| `nemo-hzn-base-eval-1116b` (rwdj5968, xotxnvzz) | base | 0.2397* | 0.3258 | 267 | 71.5 / 21.0 / 7.1 / 0.4 | 1.65 / 34.5 | 217 / 247 | in pool |
| `nemo-hzn-base-eval-1119c` (57cu034a) | base | 0.2285 | 0.3483 | 268 | 75.4 / 14.9 / 9.0 / 0.7 | 1.53 / 32.5 | 214 / 246 | in pool |
| `nemo-hzn-base-eval-1121d` (97qwrb4q, 70uo223k) | base | 0.2434 | 0.3371 | 269 | 74.0 / 16.7 / 8.6 / 0.7 | 1.77 / 34.6 | 221 / 255 | in pool (pool avg 0.2372) |
| `...lr2e-6-1116-eval25` (2u04hbf7) | full run 25 | 0.2472 | 0.3371 | 268 | 71.3 / 20.5 / 7.8 / 0.4 | 2.27 / 32.1 | 216 / 247 | +0.010 +- 0.019 |
| `...lr2e-6-1116-eval50` (4kikabmh) | full run 50 | 0.2247 | 0.3258 | 280 | 72.9 / 15.0 / 7.1 / 5.0 | 1.55 / 29.3 | 221 / 243 | -0.012 +- 0.023 |
| `...cand-from25-1118-eval50` (s2o65gte) | candidate 50 | 0.2285 | 0.3146 | 268 | 73.5 / 18.3 / 8.2 / 0.0 | 2.06 / 33.6 | 217 / 253 | -0.009 +- 0.019 |
| `...cand-from25-1118-eval50b` (ujab110f) | candidate 50 (repeat) | 0.2285 | 0.3483 | 266 | 72.9 / 18.8 / 8.3 / 0.0 | 2.04 / 37.2 | 214 / 244 | -0.009 +- 0.019 |
| `...cand-from25-1118-eval80` (fzlq832b) | candidate 80 | 0.2360 | 0.3596 | 268 | 72.0 / 21.6 / 5.6 / 0.7 | 2.68 / 35.1 | 213 / 248 | -0.001 +- 0.018 |
| `...lr4e-6-ris-frz-1118-eval25` (43jn1u6l) | H 25 | 0.2734 | 0.3708 | 267 | 69.3 / 29.2 / 1.5 / 0.0 | 2.51 / 49.1 | 216 / 247 | +0.036 +- 0.021 |
| `...lr4e-6-ris-frz-1118-eval25b` (cdrpppwv) | H 25 (repeat) | 0.2509 | 0.3596 | 268 | 70.5 / 26.9 / 2.2 / 0.4 | 2.44 / 48.1 | 215 / 247 | +0.014 +- 0.022 |
| `...sc-1119-eval25` (fcen49uj) | sc 25 | 0.2509 | 0.3708 | 266 | 71.4 / 21.4 / 7.1 / 0.0 | 2.31 / 40.2 | 209 / 236 | +0.014 +- 0.024 |
| `...sc-1119-eval50` (ew7v1e7a) | sc 50 | 0.2584 | 0.3820 | 267 | 70.4 / 27.3 / 2.2 / 0.0 | 3.57 / 48.3 | 197 / 226 | +0.021 +- 0.021 |
| `...r3-1119-eval25` (kpq642w9) | R3 25 | 0.2285 | 0.3708 | 267 | 76.4 / 16.1 / 6.4 / 1.1 | 1.11 / 30.3 | 225 / 257 | -0.009 +- 0.020 |
| `...r3-1119-eval35` (uwegb1ph failed attempt; hrxpphzn) | R3 35 | 0.2434 | 0.3483 | 266 | 74.4 / 17.7 / 7.9 / 0.0 | 1.07 / 29.7 | 209 / 240 | +0.006 +- 0.022 |
| `...r3-1119-eval55` (eld7nn2w) | R3 55 | 0.2434 | 0.3596 | 268 | 76.5 / 16.0 / 6.7 / 0.7 | 0.73 / 23.9 | 209 / 241 | +0.006 +- 0.021 |
| `...r3-1119-eval75` (81bcpd3l) | R3 75 | 0.2659 | 0.3596 | 267 | 80.1 / 13.9 / 5.6 / 0.4 | 0.70 / 25.5 | 231 / 262 | +0.029 +- 0.019 |
| `...r3-1119-eval100` (g4ogiz5r) | **R3 100 (best)** | **0.2697** | **0.3933** | 266 | 76.3 / 18.8 / 4.9 / 0.0 | 0.84 / 31.6 | 219 / 252 | +0.032 +- 0.020 |
| `...r3-1119-eval105.stopped` (0jl2ijov) | R3 105 | - | - | 0 | - | - | - | **stopped** (launched with too little job time); no result |
| `...r3-1119-eval114` (lzmefkub) | R3 114 | 0.2509 | 0.3708 | 268 | 65.3 / 27.6 / 7.1 / 0.0 | 0.95 / 36.9 | 215 / 247 | +0.014 +- 0.026 |
| `...r3-seqmean-1122-eval45` (wikc1e6f) | twin 45 | 0.2360 | 0.3146 | 269 | 66.2 / 28.3 / 4.8 / 0.7 | 2.00 / 42.8 | 211 / 239 | -0.001 +- 0.022 (vs R3 55 -0.007 +- 0.024) |
| `...r3-seqmean-1122-eval50` (e4vm8cyp) | twin 50 | 0.2519 **partial** | 0.3837 **partial** | 246 of 267 | 69.1 / 30.9 / 0 / 0 | 2.53 / 45.9 | 209 / 239 | **partial**: job ended mid-eval; unfinished long trials missing, biased up. +0.004 +- 0.024 on the 77 complete tasks |
| twin 75 (job 103645) | twin 75 | - | - | - | - | - | - | **never ran**: twin stopped at 58; job cancelled at the kill |

\* 1116a and 1116b have no aggregate file; their avg is computed from trial results, with an unfinished trial
counted as 0.

Pooled readouts: late R3 (75 + 100) vs base +0.031 +- 0.016 (2.0 SE); late R3 (75 / 100 / 114) +0.025 +- 0.016;
early R3 (25 / 35 / 55) +0.001 +- 0.016; drifted no-R3 pool (H25 a/b, sc 25, sc 50) +0.021 +- 0.016. Serving speed
is uniform across valid evals (197-231 tok/s per call, median), so it does not separate them.

### B4. Horizon-side code changes (local in `~/SkyRL` on Horizon, uncommitted, not pushed; base commit aada336d)

| File | Change | Diff file (`$SCRATCH/calibforge/ops/` on Horizon) |
|---|---|---|
| `skyrl/train/generators/utils.py` | R3 "routes mix" fix. A batch whose rows are all masked carries no routes, which crashed the fully-async trainer ("received a mix"). Such rows now get a one-token dummy route, and the code refuses if any of them has trainable tokens (11-20) | `r3_routes_mix_fix_1120.diff` (text in appendix A2) |
| `examples/train_integrations/harbor_skycap/entrypoints/main_harbor_skycap_fully_async.py` | `@ray.remote(num_cpus=1, max_retries=0)` on `skyrl_entrypoint`. Before this, Ray silently re-ran the entrypoint on a half torn-down cluster and it hung (11-23) | `entrypoint_max_retries0_1123.diff` |
| `skyrl/backends/skyrl_train/distributed/megatron/optimizer.py`, `workers/megatron/megatron_worker.py`, `skyrl/train/config/config.py` | `trainer.policy.lr_zero_modules` (e.g. `[router,mamba]`): an LR-0 optimizer group, so Adam states still exist and resume works. Also the `SKYRL_FREEZE_PARAM_REGEX` / `SKYRL_FREEZE_PARAM_PRESET=mamba_mixer` experiment hook (requires_grad=False), used by overfit arms M / C / B (11-18) | `lr_zero_modules_1118.diff` |
| `examples/train_integrations/harbor_skycap/run_fully_async_terminus.sh`, `run_sync_terminus.sh` | `MODEL_FAMILY=nemotron-3.5` launcher branch: TP4 / EP, sigmoid router with expert bias, alltoall dispatcher, CPU optimizer offload, MTP off, renderer tokenizer id | `nemotron_launcher.diff` |
| `examples/train_integrations/harbor_skycap/terminus_installed.py` | Sandbox apt goes to cloud Ubuntu mirrors with fallback. A trial that still has no tmux fails as a setup error, which is retried then masked, instead of scoring 0 (11-19) | none (in `git diff`) |
| `skyrl/backends/skyrl_train/workers/worker_utils.py` | Diagnostic only: per-token trainer-vs-rollout logprob dump, off unless `SKYRL_DEBUG_LOGPROB_DUMP_DIR` is set | none (in `git diff`) |
| Checkpoint handling (ops scripts, not SkyRL) | `ops/ckpt_keep.sh` hard-links every 5th step (and the epoch end) into `ckpt_keep/`, only once a later checkpoint pointer is > 120 s old. Unfinished saves go to `runs/<run>/ckpt_partial/`, never inside `ckpts/`: a non-numeric `global_step_*` name there crashes SkyRL's resume (int parse, 11-23) | - |
| Known, not changed | Checkpoint cleanup race: `trainer.py:1817-1828` `_cleanup_old_checkpoints()` runs `cleanup_old_checkpoints()` (`trainer_utils.py:162`) on every node via `run_on_each_node`, then again on the driver. On shared /scratch up to 9 processes remove the same 413 GB directory at once, giving ENOENT warnings; a plausible cause of the twin's step-46 save hang | - |
| Env / launch (ops) | `SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=2400` (the first vLLM cold start takes more than 600 s). Ray on IPoIB and NCCL on the 4 IB HCAs (weight sync 30 s -> 2.5 s). Ray workers started over ssh / `--cpu-bind=none` (`srun` pins to one core: 25 tok/s). Page-cache eviction after checkpoint writes. `ops/ray_rejoin.sh` never starts a second raylet | - |

### B5. Horizon's conclusion, in its words

Nemotron-3.5-Lightning drifts under RL unless the router and the Mamba mixers are frozen. Without that freeze,
entropy climbs 2-7x with flat reward. The drift lives in the Mamba mixers and the router, and the load balancer
contributes a small part. With rollout_is, the freeze and R3, training is stable for about 100 steps. That run gains
about +0.04 on training tasks by step 26 and then holds it without compounding. On TB2.1 it is at base through step
55 and +0.03 at steps 75-100 (2.0 SE, pooled). The best checkpoint is R3 100: 0.2697 / pass@3 0.3933, against a base
pool of 0.2372. By step 114 R3 drifts again (entropy ~0.12, 27.6% context endings) and falls back to +0.014.
sequence_mean is level with prompt_mean on training tasks and shows no TB2.1 gain at step 45 (-0.001 +- 0.022); it
produces longer replies and more context-exceeded endings. One TB2.1 eval carries +-0.02-0.03 of noise, and no
Horizon arm beats base by more than ~2 SE.

---

## Appendix C. JUPITER's final table (written by the JUPITER session, 2026-11-24 06:35 CET, after the kill)

Copied as received, in five parts, except that the time of the relayed kill order is corrected to the clock reading
(21:14 PST). The full file on JUPITER is `$D/NEMOTRON35_JUPITER_RUNS_1124.md` (259 lines); the session scripts and
backups it cites were copied to `$D/ops/jupiter_session_scripts_1124/` (222 files). It covers all 31 Nemotron run
directories since 11-16 plus the code changes. No relay host, IPs, keys, task names or traces are included.

Compiled 2026-11-24 (CET) from read-only sources. All runs are stopped; the last two (C, E) were stopped at 06:15 CET
on 11-24 by the kill-all order (owner via Lila, 21:14 PST 11-23).

### C0. Sources and conventions

- `$D` = the `calibforge` directory of the project's data filesystem on JUPITER (the absolute path is in the JUPITER
  file itself and in Lila's `PLAN.md`, 11-23 entry; left out of this public file).
- **Log, 11-18 10:42 onward:** `$D/PLAN.md` and `$D/FINDINGS.md`.
- **Log, 11-16 04:09 to 11-18 09:47:** `$D/PLAN.md` starts at 11-18 10:42. The earlier history (setup, eval A, scan,
  the overfit arms, the start of the control run) comes from `$D/ops/calibforge_nemo_jup_1116/PLAN.md` and
  `$D/ops/calibforge_nemo_jup_1116/FINDINGS.md`, cited below as "ops PLAN" and "ops FINDINGS".
- **Recipe knobs:** read from the Hydra config dumps in `$D/logs/<run>.log`. Every dump in a log was parsed, and the
  policy block is `trainer.policy.*`. When dumps differ (for example EPOCHS 3 then 6, or CKPT_INTERVAL 5 then 1), both
  values are given. "key absent" means the config field did not exist in the code at that time.
- **W&B ids:** every distinct id matching `wandb.ai/.../runs/<id>` in the run's log, in order of first appearance.
  The project is `sky-posttraining-uc-berkeley/skycap-calibforge-fully-async`.
- **Dates:** "rec" is the mtime span of the run's SkyCap trial records (`runs/<run>/skycap/*.json.zst`), from the
  first to the last trial written. ".jobs" is the span from the first to the last timestamped line of
  `$D/logs/<run>.jobs`. All times are CET.
- **Trained steps:** the value in `ckpts/latest_ckpt_global_step.txt`. When PLAN records steps trained after the last
  checkpoint, they are noted.
- **Results:** quoted as written in PLAN / FINDINGS, with +- SE and n where given. Visit readouts for the rv512 arms
  are paired over tasks against `calibforge_v2_rv512_1121.base_ref.json` unless stated otherwise.
- "?" = not findable in the sources.
- **Common recipe (all v2 / rv512 runs):** fully async (sync for the overfit runs), 4-node trainer TP4 EP16 plus TP2
  engines (8 engines; 6 with the head-node block from 11-21), agent timeout 3600 s (1800 s for the 16x8 overfits), and
  16k tokens per turn. The 11-16 overfit arms are the exception: they ran a 2-node trainer TP4 EP8 with 4 engines.

### C1. Evals and scans

| Run | PLAN label | Purpose | W&B id(s) | Dates (CET) | Slurm jobs | Evaluated policy / layout | Why it ended | Key result (as recorded) |
|---|---|---|---|---|---|---|---|---|
| nemo-jup-base-eval-1116a | "Base TB2.1 eval A", "eval A", "base 1116a" | base-model TB2.1 x3 eval (89 tasks x 3) | 58i3jr9j | rec 11-16 07:34 -> 10:51; .jobs 11-16 07:17 -> 10:52 | 2930441 (eval B 2930568 queued, cancelled 11-16 13:51, never ran) | base weights; 12 engines (2-node trainer layout); T 1.0, no summarization, 3600 s, 262k ctx | completed ("eval written" 10:52; rc=134 is the SIGTERM sent after the write) | avg **0.2659**, pass@3 **0.3708** (+-~0.03 single-eval noise); 14% out-of-context, 25 agent timeouts (ops FINDINGS 11-16 10:52) |
| nemo-jup-scan320-1116 | "Scan", "overfit task scan" | pick 32 overfit tasks: 320 random train tasks (seed 1116) x 4 | g3fgrera | rec 11-16 13:54 -> 14:55; .jobs 13:26 -> 14:55 | 2930566 | base, train settings, eval-only | stopped early at 1,242 / 1,280 trials once the pick was made (14:55) | tasks by #solved/4 {0: 122, 1: 36, 2: 32, 3: 35, 4: 67}; 89 candidates; 32 picked (`data/overfit32_nemo-jup-scan320-1116.txt`), mean pass 0.523 |
| nemo-jup-eval-ris-r2-s35-1120 | "J35 eval", "J-r2 ckpt-35 eval" | TB2.1 x3 of J-r2 ckpt 35 | cz83t5bl | rec 11-20 06:39 -> 08:22; .jobs 06:05 -> 08:23 | 3022520 | `runs/nemo-jup-cf-v2-b32-lr4e-6-ris-frz-1118-r2/ckpt_keep/global_step_35`; 8-engine layout (eval_ckpt.sh) | completed 08:23 | FINAL **0.2472 / 0.3596** (+-0.03); equal to base8 on the same layout (J35 - base8 = 0.000 / 0.000). The first comparison to 1116a was confounded by layout. 106 attempts stalled by the 06:52 proxy outage were retried (FINDINGS 11-20 08:24) |
| nemo-jup-eval-base-8eng-1120 | "base8" | base-model eval on the 8-engine eval layout (baseline for the ckpt evals) | od1k730z | rec 11-20 18:10 -> 20:13; .jobs 17:53 -> 20:14 | 3024355 | base weights; 8 engines | completed (Slurm FAILED 6:0 = normal teardown after "eval written") | FINAL **0.2472 / 0.3596** (+-0.03); excluding one task 0.2462 / 0.3523. The 8-engine layout reads about 0.02 below 1116a |
| nemo-jup-eval-ris-r2-s25-1119 | "J25", "J-r2 ckpt-25 eval" | TB2.1 x3 of J-r2 ckpt 25 | cazwfmhf | rec 11-20 23:38 -> 11-21 01:12; .jobs 11-20 23:09 -> 11-21 01:13 | 3018873 | `.../nemo-jup-cf-v2-b32-lr4e-6-ris-frz-1118-r2/ckpt_keep/global_step_25`; 8 engines | completed | FINAL **0.2921 / 0.4157** (+-0.03); vs base8 +0.045 avg / +0.056 pass@3, "about 1.5-2x noise" (not significant) |
| nemo-jup-eval-dppo-r2-s25-1119 | "H25", "H-r2 ckpt-25 eval" | TB2.1 x3 of H-r2 ckpt 25 (decision eval for continuing DPPO + freeze) | n1smvvm6 | rec 11-21 01:41 -> 04:08; .jobs 01:23 -> 04:08 | 3015990 | `.../nemo-jup-cf-v2-b32-lr4e-6-dppo-frz-1118-r2/ckpt_keep/global_step_25`; 8 engines | completed | FINAL **0.2584 / 0.3708** (+-0.03); vs base8 +0.011 / +0.011, within noise |

Recipe knobs do not apply to the evals. Their config dumps carry the launcher's training defaults: 1116a and scan
show LR 2e-6 / dppo / prompt_mean / train48 / 32x16; the four checkpoint evals and base8 show 4e-6, lr_zero
[router, mamba], bias rate 0.0 and ckpt_interval 1000. No eval trained, and none has checkpoints.

Not run (no run dir): `nemo-jup-eval-ris-r3-s25-1121`, job 3037627 (R3 ckpt-25 eval). It was held from 11-21 04:46 and
cancelled in the 11-24 06:19 kill-all.

### C2. Overfit runs

| Run | PLAN label | Purpose | W&B id(s) | Dates (CET) | Slurm jobs | LR | loss type | reduction | R3 | Frozen (policy lr_zero_modules) | Tasks | batch x group | epochs | Trained steps | Why it ended | Key result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nemo-jup-of32-lr2e-6-1116 | "overfit arm 2e-6", "lr2e-6" | sync overfit, 32 scan-picked tasks, LR bracket | g4drmqdu, rfec0ms7, ifuknpgv | rec 11-16 21:55 -> 23:52; .jobs 11-16 21:32 -> 11-17 00:46 | 2942884 (4 nodes, DEBUG_HOLD; replaces 2937561, cancelled before start) | 2e-6 | dppo | prompt_mean | off | none (key absent; freeze_moe_router false); bias rate not set | overfit32_nemo-jup-scan320-1116 | 32 x 16 | 15 | 1 | Host OOM at step-2 policy_train (23:53:08; Ray memory monitor). The relaunch in the held job died in load_checkpoints with a CUDA OOM (host memory spilled into HBM). Conclusion: the 2-node trainer does not fit on GH200, so the run was superseded by the tr4 layout | step 1 (pre-update): reward 0.504, pass@16 0.969, lp_gap 0.0425 |
| nemo-jup-of32-lr5e-6-1116 | "overfit arm 5e-6", "lr5e-6" | as above, LR 5e-6 | o8qas8pd, qzhsiaci, 098uc947 | rec 11-16 21:55 -> 11-17 00:21; .jobs 21:32 -> 00:49 | 2942887 (replaces 2937562) | 5e-6 | dppo | prompt_mean | off | none (key absent) | overfit32_nemo-jup-scan320-1116 | 32 x 16 | 15 | 1 | Same OOM at 00:22:05. The relaunch with OVERLAP_CPU_OPT=false also died in load_checkpoints (CUDA OOM). Superseded by tr4 | step 1: reward 0.525, pass@16 1.000, lp_gap 0.0441 |
| nemo-jup-of32-lr5e-6-tr4-1117 | "tr4 run", "32-task run" | validate the 8-node full-run layout (4-node trainer EP16 + 8 engines) on the 32-task overfit; resume test | q0roz1sz, ofypjjdc | rec 11-17 01:04 -> 05:53; .jobs 00:52 -> 06:01 | 2942884 + 2942887 (one Ray cluster over both) | 5e-6 | dppo | prompt_mean | off | none (key absent) | overfit32_nemo-jup-scan320-1116 | 32 x 16 | 15 | 4 | Owner decision 11-17 05:53: switch to Horizon's 16 tasks. Stopped after step 4 (06:01) | rewards 0.537 / 0.512 / 0.523 / 0.502, "flat within SE ~0.02 after 3 updates at 5e-6"; layout validated (no OOM at step-2 train; resume passed) |
| nemo-jup-of16x8-lr5e-6-1117 | "of16x8", "16x8 overfit" | cross-cluster overfit on Horizon's 16 tasks | 5ciahmo7 | rec 11-17 06:18 -> 09:16; .jobs 06:02 (single launch line) | 2942884 + 2942887 | 5e-6 | dppo | prompt_mean | off | none (key absent) | overfit16_horizon-1117 | 16 x 8 | 30 | 8 | Hold jobs 2942884 / 2942887 reached walltime (~09:30; noted 09:35) | 8 steps: 0.594 / 0.594 / 0.617 / 0.617 / 0.602 / 0.703 / 0.617 / 0.664; "Steps 1-5 mean 0.605, steps 6-8 mean 0.661 (+0.056, ~1.4 SE at SE ~0.04/step): weak upward drift, not an overfit" |
| nemo-jup-of16x8-ris-nolb-trdefault-lr4e-6-1118 | "trdefault" | overfit on Lila's 16 tasks, rollout_is, balancer off, thinking_retention=null, train every turn | 9quoqx5a | rec 11-18 23:51 -> 11-19 00:08; .jobs 11-18 22:55 -> 11-19 00:51 | 2980646 (rollis link 2's allocation) | 4e-6 | rollout_is | prompt_mean | off | [] (none); bias rate 0.0 | overfit16_lila-1118 | 16 x 8 | 30 | 0 (no ckpt; stopped during step-1 policy_train) | Lila: stop after step 1. Findings: history thinking was not dropped (skycap.use_raw_content=true), and train_paths=all forwarded 18.7x the tokens. Relaunched as -k4 | step 1: generate 1081 s, reward 0.367; 2885 rows; 114.9M tokens forwarded (18.7x the final path) |
| nemo-jup-of16x8-ris-nolb-trdefault-lr4e-6-1118-k4 | "k4" | as trdefault, with use_raw_content=false (stock thinking drop) and uniform 4 turns per trajectory (path_rules:uniform_k4) | riblvyjy, s19thxfo, 1jnxn9ig | rec 11-19 01:41 -> 19:19; .jobs 00:51 -> 08:52 | 2980646, 2995686; queued and never ran: 2998158 (cancelled), 2999255 -> re-queued as 2999553 (held from 11-19 20:20 for the evals, cancelled 11-24 06:19) | 4e-6 | rollout_is | prompt_mean | off | []; bias 0.0 | overfit16_lila-1118 | 16 x 8 | 30 | 24 (step 24 last reported at 11-19 18:50) | 2995686 reached its end (planned 19:22; the end itself is not logged). The next link 2999553 was put on hold for the eval slot and never released, then cancelled in the 11-24 kill-all | 5-step means 0.283 (6-10) and 0.322 (11-15) vs Lila's thinking-kept 0.428 / 0.424; 16-20 excl. outage step 17: 0.348 vs Lila 0.475; entropy 0.1016 at step 20 (Lila 0.074); step 17 = outage step |

### C3. A/B-30 arms (from control ckpt 30) and the control full run

All run on `calibforge_v2_train48` (2,788 tasks), 32 x 16, EPOCHS 2, CKPT_INTERVAL 5, R3 off, LR 2e-6. The A/B arms
resume from `runs/nemo-jup-cf-v2-b32-lr2e-6-1117/ckpt_keep/global_step_30`.

| Run | PLAN label | Purpose | W&B id(s) | Dates (CET) | Slurm jobs | loss type | reduction | Frozen modules (policy) / bias rate | Trained steps (latest ckpt) | Why it ended | Key result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nemo-jup-cf-v2-b32-lr2e-6-1117 | "full run", "control" | first full run, 1:1 with Horizon's lr2e-6 run (R3 off, LR 2e-6 per the owner 11-17) | 1h2s9nt6, lx4apacy, wuomk3ka | rec 11-17 10:56 -> 11-18 10:48; .jobs 11-17 10:10 -> 11-18 10:49 | 2955306, 2957901, 2964965; queued and never ran: 2976099 (replaced), 2977791 (cancelled 11-18 09:07) | dppo | prompt_mean | none (lr_zero key absent); bias rate not overridden (Megatron default 1e-3 per ops PLAN 11-18 08:25) | 35; trained through step 37 (step 38 killed in rollout) | Job 2964965 was cancelled 11-18 10:48:10 by root (maintenance reservation). No restart. The control reference is steps 30-37 | Steps 1-24 reward flat (1-5 mean 0.454, 20-24 0.469); entropy 0.053-0.064 through step 30, then 0.0684 (31) -> 0.0792 (37); step-25 ckpt uploaded to HF |
| nemo-jup-cf-v2-b32-lr2e-6-ab30-nolb-1118 | "nolb", "nolb A/B" | balancer off (expert-bias update rate 0) | f1a76z1c, ja3gw9zw | rec 11-18 09:47 -> 13:55; .jobs 09:07 -> 16:13 | 2977953; 2977975 (link 2, held, cancelled 11-18 21:16) | dppo | prompt_mean | none; bias rate 0.0 | 35; trained 31-37 | Swapped out for frz at 13:55 under Lila's rule (mean entropy over steps 35-37 = 0.0766 >= 0.072). The fallback restart from 35 at 14:35 crashed at load (RESUME_MODE bug in swap_in_alloc.sh). Lila: nolb concluded | "nolb 31-37 mean entropy 0.0706 vs control 0.0714: no difference within noise. Verdict: turning off the expert-bias balancer does NOT stop the entropy rise"; paired vs control -0.0007 (31-37) |
| nemo-jup-cf-v2-b32-lr2e-6-ab30-frz-1118 | "frz" | router weights and bias frozen | 44y1u68u, d7hxmltp | rec 11-18 15:00 -> 21:07; .jobs 13:55 -> 14:48 | 2977953 (in nolb's allocation); 2983522 (follow-on, cancelled 14:34) | dppo | prompt_mean | attempt 1 (13:58): freeze_moe_router=true, which failed at the optimizer load (bucket shape mismatch). Attempt 2 (14:48): router_lr_zero=true (router weights LR 0) + bias rate 0.0 | 40; trained 31-42 | Allocation 2977953 reached walltime (TIMEOUT 21:07:07). No follow-on link: the STOP_CHAIN removal was blocked | "router freeze alone does not slow the entropy climb (rollis-frz 31-42 mean +0.0012; frz-control 31-37 mean -0.0001)"; bitwise freeze check at 35 passed |
| nemo-jup-cf-v2-b32-lr2e-6-ab30-rollis-1118 | "rollis" | bounded-ratio loss (rollout_is, band (0.5, 5)), balancer on | n7aucc8o, z8jf8vyy | rec 11-18 12:36 -> 22:41; .jobs 09:09 -> 11-19 00:26 | 2977976 (cancelled by root 11:23 before any step), 2978021 (Ray failed to start, cancelled), 2980622, 2980646 (took over at 50; then reused for trdefault) | rollout_is | prompt_mean | none; bias rate default | 50; trained 31-50 | Lila: stop at step 50 (verdict reached). Link 2980646 reused for the overfit arm | Verdict 48-50: entropy 0.1194 / 0.1345 / 0.1292 (mean 0.1277) vs Horizon original 0.099@50; reward mean 0.457; "rollout_is does not slow the entropy rise" |
| nemo-jup-cf-v2-b32-lr2e-6-ab30-mzf-1118 | "mzf" | router + all Mamba mixer params frozen (LR 0), bias rate 0 | vhxxkzae, 8i9z9b0t, bl8f6fuj | rec 11-18 18:01 -> 11-19 06:23; .jobs 11-18 17:08 -> 11-19 07:53 | 2987658, 2987772 | dppo | prompt_mean | lr_zero_modules [router, mamba]; bias 0.0. The first start, 17:16, missed conv1d; restarted 18:09 before step 31 with the fixed patterns | 50; trained 31-50 | Stopped 11-19 06:23 before step 51 and swapped for run H (Lila's decision) | "mzf entropy 48-50 0.0738 0.0679 0.0712 (mean 0.0710), reward mean 0.512" vs rollis 0.1277 / 0.457; "freezing Mamba mixers + router (with the balancer bias) stops the entropy climb at no reward cost through 20 steps; freezing the router alone, turning the balancer off, or rollout_is do not"; bitwise check at 35 passed (414 / 414 frozen tensors) |

### C4. v2 full runs from base (J / H, J-r2 / H-r2, S, R3, R3+SC, unfrozen-mixer arm)

All on `calibforge_v2_train48`, 32 x 16, EPOCHS 2, LR 4e-6, bias rate 0.0.

| Run | PLAN label | Purpose | W&B id(s) | Dates (CET) | Slurm jobs | loss type | reduction | R3 | Frozen (policy lr_zero_modules) | ckpt interval | Trained steps (latest ckpt) | Why it ended | Key result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nemo-jup-cf-v2-b32-lr4e-6-ris-frz-1118 | "J", "Full run J" | full run from base with the mzf freeze + rollout_is | lvxc7hn1 | rec 11-19 06:09 -> 08:19; .jobs 04:44 -> 09:49 | 2997191; 2997446 (cancelled) | rollout_is | prompt_mean | off | [router, mamba] | 5 | 5 | Stopped 08:19-08:21: contaminated by the Dune apt outage (07:20-08:00). Lila: restart from base as -r2 | steps 1-5 reward 0.508 / 0.389 / 0.279 / 0.152 / 0.008 (outage; ~90% 'error' endings); step-5 bitwise vs base PASSED |
| nemo-jup-cf-v2-b32-lr4e-6-dppo-frz-1118 | "H", "Run H (DPPO version of J)" | DPPO twin of J (replaced mzf) | td0ohoul | rec 11-19 06:42 -> 08:18; .jobs 06:23 -> 08:21 | 2987772 (mzf's allocation); 2997901 (follow-on, cancelled) | dppo | prompt_mean | off | [router, mamba] | 5 | 0 checkpoints (steps 1-3 trained per PLAN) | Same apt outage; restarted as -r2 | H 1-3 reward 0.512 / 0.266 / 0.090 (outage) |
| nemo-jup-cf-v2-b32-lr4e-6-ris-frz-1118-r2 | "J-r2" | J restarted from base with the apt mirror fix + tmux guard | 1nvvf22s, lce9kfit, uufisuec | rec 11-19 08:38 -> 11-20 21:38; .jobs 11-19 08:26 -> 11-20 23:08 | 2997191, 2999551, 3014196; re-queued: 2999195 -> 2999551, 3013043 -> 3014196; 3032407 (held 11-20 21:38, cancelled 11-24 06:19) | rollout_is | prompt_mean | off | [router, mamba] | 5, then 1 (links from 3014196) | 40 | Entropy stop-and-ask rule fired at step 40 (0.630 > 0.6); stopped at complete ckpt 40 on 11-20 21:38. Lila 21:45: "J-r2 is finished as an arm". (It was also swapped out 11-20 04:08 at ckpt 35 for R3 and resumed 18:04 in 3014196) | J25 eval 0.2921 / 0.4157 (+0.045 / +0.056 vs base8, ~1.5-2x noise); J35 eval 0.2472 / 0.3596 (= base8). Clean 5-step windows 0.529 (1-5) ... 0.624 (31-35), 0.582 (36-40). Entropy 0.054-0.061 (1-10), 0.103 (21-25 mean), 0.169 (26-30), 0.280-0.367 (31-35), 0.475-0.630 (36-40). Paired vs dppo-r2, steps 16-30: +0.004 +- 0.009, +0.002 +- 0.012, +0.005 +- 0.011 (FINDINGS 11-21 06:19) |
| nemo-jup-cf-v2-b32-lr4e-6-dppo-frz-1118-r2 | "H-r2" | H restarted from base (DPPO + freeze) | lq60kl2k, p0tanb8t | rec 11-19 08:39 -> 11-20 01:14; .jobs 11-19 08:27 -> 11-20 01:14 | 2987772, 2999552; re-queued: 2999196 -> 2999552, 3004388 -> 3014197 (cancelled 11-20 00:24 for the S arm) | dppo | prompt_mean | off | [router, mamba] | 5 | 35 | Link 2999552 reached walltime 11-20 01:14. Its next link was given to the sequence_mean arm, and H-r2 "gets no further link unless its step-25 eval >= 0.30"; H25 came in at 0.2584 | H25 eval 0.2584 / 0.3708 (+-0.03), +0.011 / +0.011 vs base8, within noise. Entropy flat 0.052-0.060; clean windows 0.538 / 0.487 / 0.523 / 0.521 / 0.563 / 0.502 / 0.555 (1-5 ... 31-35) |
| nemo-jup-cf-v2-b32-lr4e-6-ris-frz-seqmean-1119 | "S arm", "sequence_mean arm" | J-r2 recipe with loss_reduction=sequence_mean (replicate of Lila's arm) | none (no log) | never ran; submitted 11-20 00:24, cancelled 14:35 | 3019525 (nemo-full-S; held by the cap enforcer the whole time) | rollout_is (per PLAN) | sequence_mean (per PLAN) | off | [router, mamba] (per PLAN) | 1 (per PLAN) | none (run dir holds only keeper / reaper markers) | Superseded: Lila's sequence_mean-only run "took off at step ~42", so S was replaced by R3+SC (11-20 14:35) | none |
| nemo-jup-cf-v2-b32-lr4e-6-ris-frz-r3-1119 | "R3 arm", later "frozen R3" | J-r2 recipe + routing replay (R3), from base | fo7reydo, pv4ye15o, ai4sqcle, y262518b, nvrb3w2h, jruqdhu8, bo4vp16j | rec 11-20 04:37 -> 11-21 04:48; .jobs 11-20 04:08 -> 11-21 04:51 | 2999551 (04:11, no step before walltime 04:58), 3022474 (restarted in place 11:56 for the route-pad patch), 3014196 (from 23:27; OOM at steps 25 and 26; head-block rehearsal); 3027303 never ran (cancelled 11-21 10:22 by swap_queue) | rollout_is | prompt_mean | **on** | [router, mamba] | 1 | 26 | Lila 11-21 10:18: the pending R3 slot goes to the unfrozen-mixer arm. "Frozen R3 chain ended at ckpt 26" (3027303 cancelled). Its ckpt-25 eval (3037627) never ran | step 1 gap 0.0110 (J-r2 0.0447), clip 0.145% (J-r2 1.50%); clean windows 0.538 (1-5), 0.536 (6-10), 0.539 (11-15), 0.5506 (16-20), 0.5632 (21-25); entropy 0.056-0.066; gap 0.011-0.013. Paired J-r2 - R3, steps 16-25: +0.003 +- 0.011, -0.007 +- 0.020 |
| nemo-jup-cf-v2-b32-lr4e-6-ris-frz-r3sc-1120 | "R3+SC" (job name nemo-full-C, which is not rv512 arm C) | R3 + score centering + sample-support replay, top_k 20 | 53bp1o04, dj9slujz, dgysxwfy, 1gncsxky | rec 11-20 22:06 -> 11-21 10:42; .jobs 11-20 21:45 -> 11-21 12:12 | 3014196 (smoke, 11-20 21:45-23:08, step 1), 3029512; 3037620 (cancelled 11-21 10:42) | rollout_is | prompt_mean | on (+ enable_score_centering true, top_k 20) | [router, mamba] | 1 | 5 | Lila 11-21 10:26: swap the unfrozen-mixer arm into R3+SC's allocation; "R3+SC paused" at ckpt 5 (kept). Not resumed | Smoke step 1: reward 0.504, clean 0.562, entropy 0.0525, gap 0.0110, clip 0.148%, support TV 0.0053. Steps 2-4: reward 0.395 / 0.588 / 0.441, entropy 0.058 / 0.054 / 0.059 |
| nemo-jup-cf-v2-b32-lr4e-6-ris-rfrz-r3-1121 | "unfrozen-mixer arm", "new arm", "NR" (job name nemo-full-M) | R3 with only the router frozen (Mamba mixers trained), from base | njfw5njk, x1ywhr7t | rec 11-21 11:06 -> 21:11; .jobs 10:25 -> 22:42 | 3029512, 3039870; queued and never ran: 3039816 (cancelled), 3044143 (cancelled 21:10) | rollout_is | prompt_mean | on | [router] | 1 | 15 | Retired 11-21 21:11 for the rv512 repeat-visit arms (Lila's call); its allocation went to arm A | Paired reward vs frozen R3: step 1 +0.014 +- 0.008 (178 groups, noise floor); steps 1-5 +0.020 +- 0.009 (278); 6-10 +0.014 +- 0.011 (n=129); 11-15 +0.012 +- 0.014 (n=111), none significant. Entropy 11-15 0.0748 vs 0.0597 (+25%), still rising |

### C5. rv512 repeat-visit arms (512-task subset, visits = epochs)

All on `calibforge_v2_rv512_1121` (512 tasks), 32 x 16, LR 4e-6 except A, rollout_is, R3 on, bias rate 0.0, seed 42,
CKPT_INTERVAL 1, head block + 6 engines, TPS 1. EPOCHS=3 at launch. C, E and D2 were relaunched with EPOCHS=6 on 11-22
at 23:50 (C restarted 11-23 00:07; E and D2 got EPOCHS=6 links). Plan as of 11-23 19:27 (Lila, under the owner's
wind-down): C ends after visit 6, D2 at visit 3 or the guard, E up to 6 visits; superseded by the 11-24 kill-all.

| Run | PLAN label | Purpose | W&B id(s) | Dates (CET) | Slurm jobs | LR | reduction | Frozen (policy lr_zero_modules) | Epochs planned | Trained steps (latest ckpt) | Why it ended | Key result (paired vs base ref unless noted) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nemo-jup-cf-rv512-b32-lr1e-5-ris-frz-r3-1121 | "Arm A" (the "lr1e-5 arm") | frozen R3 at LR 1e-5 | 2k1cr4e8, 95gylg9a, bp6lp84f | rec 11-21 21:36 -> 11-22 13:55; .jobs 11-21 20:36 -> 11-22 15:26 | 3039870, 3047278; 3051248 (cancelled) | 1e-5 | prompt_mean | [router, mamba] | 3 | 19 | Lila 11-22 13:52: stop A (-0.033 +- 0.008 vs base after two passes and growing). Ckpt 19 kept; D took its slot. A false crash-restart by the spill guard (11-22 00:34) cost ~1 h | v1 - base -0.011 +- 0.006 (n=484); v2 - base -0.033 +- 0.008 (n=511); v2 - v1 -0.016 +- 0.007 (483); A - C v1 -0.006 +- 0.006 (n=482); v2 cap share 0.426, ctx 0.324 |
| nemo-jup-cf-rv512-b32-lr4e-6-ris-frz-r3-1121 | "C" (arm C) | frozen R3 at 4e-6, prompt_mean (reference arm) | 1heg2a7d, v6eycqlv, 3yrec2by, r3eo21np, hnt7pbt7, j0vajcij | rec 11-22 04:35 -> 11-24 06:15; .jobs 11-22 04:10 -> 11-24 06:17 | 3047276, 3051163, 3066415, 3073196; queued and never ran: 3057382 (cancelled, EPOCHS=3), 3088723 (cancelled 11-24) | 4e-6 | prompt_mean | [router, mamba] | 3, then 6 | 51 | Kill-all 11-24 06:15 (owner via Lila): stopped by PID at global_step_51 (state file present). Also paused by the Dune setup-failure rule 11-23 12:55-14:08 | v1 -0.006 +- 0.006 (510); v2 +0.003 +- 0.006 (489); v3 +0.012 +- 0.006 (509); v4 +0.022 +- 0.006 (484; contains Dune-affected steps 36-37); v4 - v1 +0.027 +- 0.006 (482). v5 +0.019 +- 0.007 (501), final (visit 5 closed when visit 6 opened 11-24 06:11; no step trained in visit 6) |
| nemo-jup-cf-rv512-b32-lr4e-6-ris-rfrz-r3-1121 | "B" (arm B) | router frozen, Mamba mixers trained (B - C isolates the mixers) | xjyq09u0, wak64a1y | rec 11-22 04:50 -> 18:42; .jobs 04:10 -> 20:12 | 3047277, 3051164; 3057383 (held, then cancelled) | 4e-6 | prompt_mean | [router] | 3 | 21 | Lila 11-22 18:08: stop B when visit 2 closes and start E in its allocation; done at 18:42, ckpt 21 kept | v1 -0.004 +- 0.006 (510); B - C v1 +0.003 +- 0.006 (509); v2 +0.008 +- 0.006 (499); B - C v2 +0.005 +- 0.006 (479) |
| nemo-jup-cf-rv512-b32-lr4e-6-ris-nofrz-r3-1122 | "D" (attempt 1) | router and mixers trained (D - B isolates the router) | tkf4zqgq, bpat1o9n | rec 11-22 14:17 -> 15:30; .jobs 13:55 -> 16:15 | 3047278 (A's allocation), 3056082 (FAILED 16:15); 3057479 (cancelled) | 4e-6 | prompt_mean | [] (none) | 3 | 1 (no fully_async_state.pt) | Chain handoff race: the handoff cancelled 3047278 after latest_ckpt was written but before fully_async_state.pt existed, so 3056082 could not resume. Restarted from base as ...-1122b (attempt 1 records not used) | step 1: entropy 0.0547, gap 0.0115, grad norm 0.080; HBM spill 0-1 MB ("no memory obstacle to training the router on GH200") |
| nemo-jup-cf-rv512-b32-lr4e-6-ris-nofrz-r3-1122b | "D2" | D restarted from base under a fresh name | wwsm00tr, qzugx9lt | rec 11-22 23:41 -> 11-23 22:04; .jobs 11-22 23:11 -> 11-23 23:35 | 3058646, 3066494; queued and never ran: 3065859 (cancelled, EPOCHS=3), 3079375 (cancelled 11-24) | 4e-6 | prompt_mean | [] (none) | 3, then 6 (D2 limited to visit 3) | 23 | Entropy guard: 5-step mean 0.1017 > 0.10 (steps 19-23). Stopped at global_step_23 on 11-23 22:06; ckpt 23 kept; E swapped into 3066494 | v1 -0.006 +- 0.006 (508), D2 - C +0.001 +- 0.005 (508); v2 +0.007 +- 0.006 (506), D2 - C +0.001 +- 0.006 (486), ctx 0.231 vs C 0.174; v3 PARTIAL (263 groups): v3 - base -0.013 +- 0.011, D2 - C v3 -0.025 +- 0.011 (n=262); entropy 0.058 -> 0.115 over 23 steps |
| nemo-jup-cf-rv512-b32-lr4e-6-ris-frz-r3-seqm-1122 | "E" (arm E) | C with loss_reduction=sequence_mean (Lila's reference reduction) | sakpeofz, kv6u6w0r, bjajwtt6, nh82dbip, uht6nw1t | rec 11-22 19:04 -> 11-24 06:15; .jobs 11-22 18:42 -> 11-24 06:17 | 3051164 (B's allocation), 3066493, 3066494 (D2's allocation, 11-23 22:08 -> 11-24 01:53), 3069118; queued and never ran: 3061187 (cancelled, EPOCHS=3), 3086502 (cancelled 11-24) | 4e-6 | **sequence_mean** | [router, mamba] | 3, then 6 | 35 | Kill-all 11-24 06:15: stopped by PID at global_step_35 (state file present). Also paused by the Dune rule 11-23 12:55-13:30 | v1 -0.001 +- 0.006 (512), E - C +0.005 +- 0.005 (510); v2 +0.005 +- 0.006 (497), E - C -0.001 +- 0.006 (475); v3 +0.024 +- 0.007 (488), E - C v3 +0.013 +- 0.006 (n=487); v4 PARTIAL (394 groups at the stop): +0.035 +- 0.008, E - C v4 +0.015 +- 0.008 (371) |

Note on labels: on JUPITER the router-only-frozen rv512 arm is B. D is the first no-freeze attempt, 1 step, unusable.
D2 is its restart.

### C6. Checkpoints on disk (JUPITER; every path is `$D/runs/<run>/...`)

"Latest" is `ckpts/global_step_<latest_ckpt_global_step.txt>`. "Kept" lists every `ckpt_keep/global_step_N` (hard
links that survive rotation). Sync-launcher runs (of32, of16x8, trdefault / k4) write no `fully_async_state.pt`; that
is expected. The R3 ckpt_keep extras (steps 1-4, 6-9, 11-13) were deleted on 11-20 at Lila's instruction. A's ckpt 10
was rotated out (MAX_CKPTS=6).

| Run | Latest | Latest ckpt (under `$D/runs/<run>/`) | Kept (ckpt_keep) |
|---|---|---|---|
| nemo-jup-base-eval-1116a, nemo-jup-scan320-1116, nemo-jup-eval-ris-r2-s35-1120, nemo-jup-eval-base-8eng-1120, nemo-jup-eval-ris-r2-s25-1119, nemo-jup-eval-dppo-r2-s25-1119 | evals, none | none | none |
| nemo-jup-of32-lr2e-6-1116 | 1 | ckpts/global_step_1 (sync) | none |
| nemo-jup-of32-lr5e-6-1116 | 1 | ckpts/global_step_1 (sync) | none |
| nemo-jup-of32-lr5e-6-tr4-1117 | 4 | ckpts/global_step_4 (sync) | none |
| nemo-jup-of16x8-lr5e-6-1117 | 8 | ckpts/global_step_8 (sync) | none |
| nemo-jup-of16x8-ris-nolb-trdefault-lr4e-6-1118 | none | none | none |
| nemo-jup-of16x8-ris-nolb-trdefault-lr4e-6-1118-k4 | 24 | ckpts/global_step_24 (sync) | none |
| nemo-jup-cf-v2-b32-lr2e-6-1117 (control) | 35 | ckpts/global_step_35 (state present) | 25, 30, 35 |
| nemo-jup-cf-v2-b32-lr2e-6-ab30-nolb-1118 | 35 | ckpts/global_step_35 (state present) | 35 |
| nemo-jup-cf-v2-b32-lr2e-6-ab30-frz-1118 | 40 | ckpts/global_step_40 (state present) | 35, 40 |
| nemo-jup-cf-v2-b32-lr2e-6-ab30-rollis-1118 | 50 | ckpts/global_step_50 (state present) | 50 |
| nemo-jup-cf-v2-b32-lr2e-6-ab30-mzf-1118 | 50 | ckpts/global_step_50 (state present) | 35, 40, 45, 50 |
| nemo-jup-cf-v2-b32-lr4e-6-ris-frz-1118 (J) | 5 | ckpts/global_step_5 (state present) | 5 |
| nemo-jup-cf-v2-b32-lr4e-6-dppo-frz-1118 (H) | none | none | none |
| nemo-jup-cf-v2-b32-lr4e-6-ris-frz-1118-r2 (J-r2) | 40 | ckpts/global_step_40 (state present) | 5, 10, 15, 20, 25, 30, 35, 40 |
| nemo-jup-cf-v2-b32-lr4e-6-dppo-frz-1118-r2 (H-r2) | 35 | ckpts/global_step_35 (state present) | 5, 10, 15, 20, 25, 30, 35 |
| nemo-jup-cf-v2-b32-lr4e-6-ris-frz-seqmean-1119 (S) | never ran | none | none (empty ckpt_keep dir) |
| nemo-jup-cf-v2-b32-lr4e-6-ris-frz-r3-1119 (R3) | 26 | ckpts/global_step_26 (state present) | 5, 10, 15, 20, 25 |
| nemo-jup-cf-v2-b32-lr4e-6-ris-frz-r3sc-1120 (R3+SC) | 5 | ckpts/global_step_5 (state present) | 5 |
| nemo-jup-cf-v2-b32-lr4e-6-ris-rfrz-r3-1121 (unfrozen-mixer) | 15 | ckpts/global_step_15 (state present) | 5, 10, 15 |
| nemo-jup-cf-rv512-b32-lr1e-5-ris-frz-r3-1121 (A) | 19 | ckpts/global_step_19 (state present) | 19 |
| nemo-jup-cf-rv512-b32-lr4e-6-ris-frz-r3-1121 (C) | 51 | ckpts/global_step_51 (state present) | 31, 51 |
| nemo-jup-cf-rv512-b32-lr4e-6-ris-rfrz-r3-1121 (B) | 21 | ckpts/global_step_21 (state present) | 21 |
| nemo-jup-cf-rv512-b32-lr4e-6-ris-nofrz-r3-1122 (D attempt 1) | 1 | ckpts/global_step_1 (NO fully_async_state.pt; not resumable) | none |
| nemo-jup-cf-rv512-b32-lr4e-6-ris-nofrz-r3-1122b (D2) | 23 | ckpts/global_step_23 (state present) | 23 |
| nemo-jup-cf-rv512-b32-lr4e-6-ris-frz-r3-seqm-1122 (E) | 35 | ckpts/global_step_35 (state present) | 35 |

Each `ckpts/` directory also holds up to MAX_CKPTS=6 older steps, for example C global_step_46-51 and R3 21-26. HF
repo (`CharlieFRuan/nemotron35-calibforge-ckpts`): control global_step_25 was uploaded and verified on 11-18 02:37.
R3 steps 1-4 were uploaded by a watcher bug and deleted on 11-21 00:08. HF watchers have been paused since 11-20
18:35 (storage quota).

Trained past the last ckpt (from PLAN): control trained through step 37 (ckpt 35); nolb 37 (ckpt 35); frz 42 (ckpt
40). H trained steps 1-3 with no ckpt.

### C7. JUPITER-side code and script changes (chronological)

Repo state, checked read-only on 11-24:

- calibforge-SkyRL is on branch `jupiter-nemotron` at HEAD 561e763b (the brief commit). All local code changes are
  uncommitted in the working tree and none are pushed. `git diff --stat`: 15 files, +468 / -45. Untracked:
  `examples/train_integrations/harbor_skycap/path_rules.py`, `handoff/ops/calibforge_rl_1110/test_env_retry.py`.
- Ops scripts live in `$D/ops/calibforge_nemo_jup_1116/` (not a git repo).
- Watcher / controller scripts were in the session scratchpad. On 11-24 they were all copied with their backups to
  `$D/ops/jupiter_session_scripts_1124/` (222 files); "scratchpad/" below means that copy.
- Relay scripts live in the rl-env-cleaner checkout under `skycap_overfit/jupiter/` (git: `relay/connect_proxy.py`
  and `relay/login_services.sh` modified, +11 / -6; `ray_cluster.sh`, `build_nemo_env.sh`, `gsm8k_nemotron.sbatch`
  and `relay/proxy_watchdog.sh` untracked).

"Diff" = a saved diff file. "Backup" = a pre-change copy. "git" = recoverable only through `git diff` against HEAD.

| Date (CET) | File(s) | Change | Backup / diff |
|---|---|---|---|
| 11-16 04:09 | calibforge-SkyRL `pyproject.toml`, `uv.lock` | pip nvcc / crt / nvvm pinned to 13.0.88 (TileLang / fla JIT on sm_90) | git only |
| 11-16 04:09 | calibforge-SkyRL `skyrl/backends/skyrl_train/workers/worker.py` | numa_set_preferred instead of numa_set_membind (strict bind OOM-killed ckpt loads on GH200) | git only |
| 11-16 04:09 / 07:59 | calibforge-SkyRL `examples/.../harbor_skycap/run_fully_async_terminus.sh`, `run_sync_terminus.sh` | Horizon's MODEL_FAMILY=nemotron-3.5 launcher port + MEGATRON_EP / MEGATRON_CP / POLICY_GPUS_PER_NODE knobs; later OVERLAP_CPU_OPT (11-17 00:35) | git only |
| 11-16 04:09 | rl-env-cleaner `relay/connect_proxy.py`, `relay/login_services.sh` | login-node CONNECT proxy: provider allowlist, 8192 fds | git only (tracked, modified) |
| 11-16 05:21 | rl-env-cleaner `ray_cluster.sh` (untracked) | SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=1800 | none |
| 11-17 00:12 | `ray_cluster.sh` | RAY_memory_usage_threshold=0.98 | none |
| 11-17 00:12 / 00:52 | `ops/.../relaunch_in_hold.sh`, `relaunch_two_jobs.sh` (new); `ray_cluster.sh` MIMO_EXTRA_NO_PROXY | relaunch in held DEBUG_HOLD jobs; one Ray cluster over two jobs | new files; none |
| 11-17 07:15-07:27 | `ops/.../calibforge_chain.sbatch` (new); `relay/proxy_watchdog.sh` (new, untracked) | 12 h chain links, link-start sandbox cleanup, driver watchdog; proxy wait / stop; proxy watchdog | new files |
| 11-17 07:29 / 11:40 | `ops/.../hf_ckpt_upload.py`, `hf_ckpt_watch.sh` (new) | HF upload of every 25th + epoch-end ckpt; 11:40 fix: stage hard links and verify the repo tree | new files |
| 11-17 18:34 | `ops/.../dune_reaper.sh` (new) | delete a run's sandboxes after each link ends | new file |
| 11-18 ~08:3x | `ops/.../run_rl_jupiter.sh` | RESUME_PATH -> resume_mode from_path, ignored once the run has its own ckpt (chain-safe) | none found |
| 11-18 ~08:35 | `ops/.../calibforge_chain.sbatch`, `calibforge_nemo.sbatch` | Slurm account change | none |
| 11-18 08:55-09:07 | `ops/.../calibforge_chain.sbatch` | overlap handoff (--begin = end - LEAD_MIN, PREV_JOB, wait for next ckpt then scancel); AB_FILE; cross-run handoff (PREV_RUN, HANDOFF_STEP) | none (tested offline) |
| 11-18 12:27 | `ops/.../swap_in_alloc.sh` (new) | swap a run in a live allocation (SIGSTOP launcher, kill driver by PID, srun --overlap) | new file |
| 11-18 14:48 | `ops/.../swap_in_alloc.sh` | also unset RESUME_MODE (nolb restart crash fix) | none |
| 11-18 14:48 | calibforge-SkyRL `skyrl/train/config/config.py`, `.../distributed/megatron/optimizer.py`, `.../workers/megatron/megatron_worker.py` | router_lr_zero: zero-LR param group for the router + optimizer load patch | diff `ops/calibforge_nemo_jup_1116/router_lr_zero_1118.diff` |
| 11-18 17:08 / 17:55 | same three files | generalized to lr_zero_modules [router, mamba]; 17:55 conv1d_weight / conv1d_bias pattern fix | diff `ops/calibforge_nemo_jup_1116/lr_zero_modules_1118.diff` |
| 11-18 17:08 | `ops/.../calibforge_chain.sbatch` | pass script args to the next link | none |
| 11-18 18:09 | `ops/.../swap_in_alloc.sh` | batch pgrep matches batch scripts with args | none |
| 11-18 22:54 | `ops/.../swap_in_alloc.sh` | SWAP_ENV + NEW_RESUME=none | backup `swap_in_alloc.sh.bak1118` |
| 11-19 00:43 | calibforge-SkyRL `examples/.../harbor_skycap/path_rules.py` (new, untracked) | uniform_k4 train-path rule | new file |
| 11-19 08:33 | calibforge-SkyRL `examples/.../harbor_skycap/terminus_installed.py` | SKYCAP_APT_MIRRORS rewrite + tmux guard (TmuxMissingError); test knob SKYCAP_APT_MIRROR_BROKEN_PROB | diff `ops/.../apt_mirror_tmux_guard_1119.diff`; backup `scratchpad/terminus_installed.py.bak1119` |
| 11-19 13:56-13:58 | calibforge-SkyRL `examples/.../harbor_generator.py`, `outage_guard.py`; new `handoff/ops/calibforge_rl_1110/test_env_retry.py` | bounded env retry (SKYCAP_ENV_RETRIES=3) for TmuxMissing / AgentSetupTimeout / EnvironmentStartTimeout, + RelayStubNotStarted | diff `ops/.../env_retry_1119.diff`; backups `scratchpad/harbor_generator.py.bak1119`, `scratchpad/outage_guard.py.bak1119`, combined `scratchpad/outage_guard_local.diff` |
| 11-19 13:56 / 18:50 / 20:20 | `ops/.../step_table.py`, `ops/.../outage_steps.txt`, `scratchpad/alarm_jh.py` | outage-step exclusion; stop-and-ask rules | backups `scratchpad/step_table.py.bak1119b`, `scratchpad/alarm_jh.py.bak1119b` |
| 11-19 18:05 | `ops/.../requeue_link.py` (new) | re-queue links with CKPT_INTERVAL=1 | new file |
| 11-19 20:20 | `ops/.../eval_ckpt.sh` (new); `ops/.../calibforge_nemo.sbatch` | ckpt eval (EVAL_ONLY + RESUME_PATH); EVAL_STEP + label-only sandbox cleanup | backup `scratchpad/calibforge_nemo.sbatch.bak1119` |
| 11-19 20:20 to 11-20 00:24 | `ops/.../reply_len.py`, `think_split.py`, `reply_len_sup.py`, `adv_len.py`, `masked_groups.py` (new) | analysis only | new files |
| 11-20 03:59 | calibforge-SkyRL `skyrl/backends/skyrl_train/workers/worker_utils.py` | signed-diff-mean and exp-abs-diff>2 mismatch metrics (report only) | diff `ops/.../mismatch_metrics_1120.diff`; backup `scratchpad/worker_utils.py.bak1120` |
| 11-20 04:15 | `ops/.../step_table.py` | J-r2 continuation rules, WCTX | backup `scratchpad/step_table.py.bak1120` |
| 11-20 05:46 | `ops/.../dune_reaper.sh` | cutoff = sacct End + 5 min | backup `scratchpad/dune_reaper.sh.bak1120` |
| 11-20 07:50 | `ops/.../eval_prelim.py`; `ops/.../eval_time_split.py` (new) | count stalled attempts as superseded | backup `scratchpad/eval_prelim.py.bak1120` |
| 11-20 07:50 | calibforge-SkyRL `examples/.../sandbox_relay/exec_transport.py`, `tests/integrations/harbor_skycap/test_sandbox_relay_exec.py` | retry a relay reply until SKYCAP_RELAY_REPLY_RETRY_SEC (600 s) + new test | diff `ops/.../relay_reply_retry_1120.diff`; backup `scratchpad/exec_transport.py.bak1120` |
| 11-20 11:51 | calibforge-SkyRL `skyrl/train/generators/utils.py` | _pad_missing_routes (R3 "received a mix" fix) | diff `ops/calibforge_nemo_jup_1116/r3_route_pad_1120.diff`; backup `scratchpad/gen_utils.py.bak1120` |
| 11-20 18:32 | `ops/.../hf_ckpt_watch.sh` | keep / upload rule n % 25 == 0 only (bug: uploaded R3 steps 1-4) | backups `scratchpad/hf_ckpt_watch.sh.bak1120`, `.bak1121` |
| 11-20 14:35 to 21:45 | scratchpad `cap_enforce*.sh`, `r3_into_j*.sh`, `r3sc_smoke*.sh`, `resubmit_eval.py`, `submit_r3sc_link.py` | slot cap / swap / smoke controllers | backups `scratchpad/r3_into_j.sh.bak2142`, `scratchpad/r3sc_first_step.sh.bak2145` |
| 11-21 02:49 | `ops/.../head_gpu_block.py`, `ops/.../restart_in_alloc.sh` (new); `swap_in_alloc.sh` EXPECT_GPUS; `scratchpad/spill_guard.sh` | head-node GPU block, in-allocation restart, spill guard | backup `scratchpad/swap_in_alloc.sh.bak1121` |
| 11-21 04:47 | `ops/.../keep_ckpts_policy2.sh` (new) | keep every 5th, link only once fully_async_state.pt exists | new file |
| 11-21 05:04 | `ops/.../run_rl_jupiter.sh` | HEAD_BLOCK hook (block head GPUs, NUM_ENGINES=6) | backup `scratchpad/run_rl_jupiter.sh.bak1121` |
| 11-21 05:42 | `ops/.../run_rl_jupiter.sh`; `ops/.../dune_dns_probe.py` (new); `scratchpad/pace_ctl*.sh` | HOLD_STARTS and LAUNCH_ENV hooks (TPS=1); DNS / apt probe; pacing gates | backup `scratchpad/run_rl_jupiter.sh.bak1121b` |
| 11-22 04:13 | `scratchpad/spill_guard2.sh` | crash needs 3 explicit "no driver" answers + log silent > 5 min (false crash-restart fix) | backup `scratchpad/spill_guard2.sh.bak_1122` |
| 11-22 05:37 / 10:52 | `scratchpad/visit_table.py` | split re-dispatches; cache v2 (turns, tokens per turn, cap share) | backup `scratchpad/visit_table.py.bak_1122b` |
| 11-22 16:23 | `ops/.../calibforge_chain.sbatch` | handoff also waits for a non-empty fully_async_state.pt (D handoff race) | backup `calibforge_chain.sbatch.bak_1122` |
| 11-22 18:11 | `scratchpad/submit_rv.sh`, `scratchpad/auto_watch.sh` | XOV extra overrides | backups `.bak_1122e` |
| 11-22 20:24 | scratchpad `handoff_check` | 3 squeue tries with 60 s gaps | none found |
| 11-23 08:09 | `scratchpad/visit_table.py`; `scratchpad/matched_turns.py` (new) | cache v3 (end context, terminal tokens per turn); paired shifts | backup `scratchpad/visit_table.py.bak_1123` |
| 11-23 09:37 | `scratchpad/handoff_check.sh` | match both link-start lines | backup `scratchpad/handoff_check.sh.bak_1123` |
| 11-23 10:49 | `scratchpad/auto_watch.sh` | squeue retries + sacct before "gone" | backup `scratchpad/auto_watch.sh.bak_1123` |
| 11-23 12:55-13:01 | scratchpad `dune_pause.sh`, `dune_gate.sh`, `sf_watch.sh` (new) | Dune setup-failure pause rule and gate | new files |
| 11-23 17:33-17:48 | scratchpad `d2_guard.sh`, `ctx_window.py`, `arm_pause.sh`, `d2_end.sh`, `e_swap.sh`, `arm_guard.sh` (new); launcher-lookup fix in `dune_pause.sh` / `arm_pause.sh`; `dune_gate2.sh` (GATE_SKIP) | D2 / C / E guards; srun-step launcher lookup | backup `scratchpad/dune_pause.sh.bak_1123b` |
| 11-23 18:22 | `scratchpad/sf_watch2.sh` (new) | amended pause rule + proxy check | new file |
| 11-23 18:25 | `relay/proxy_watchdog.sh` (untracked) | restart only after 3 consecutive failed checks, 15 s connect timeout, per-check log with load average | backup `relay/proxy_watchdog.sh.bak_1123` |
| 11-23 19:29 | `scratchpad/arm_final.sh` (new) | at an arm's final ckpt: stop watchers, STOP_CHAIN, keep ckpt, clean sandboxes | new file |
| 11-23 21:37 | `scratchpad/arm_pause.sh`, `d2_end.sh`, `e_swap.sh`; `ops/.../swap_in_alloc.sh` | empty Slurm answer = unknown (sacct state); batch host / nodes from the job start line | backups `.bak_2130`, `.bak_2135`, `swap_in_alloc.sh.bak_2140` |
| 11-23 21:39 | `scratchpad/slurm_st.sh` (new), sourced by e_swap, arm_guard, arm_final, sf_watch2, dune_gate2 | shared squeue -> sacct -> UNKNOWN state; arm_guard follows the chain up to 24 h | backups `.bak_2145` |

Not changed: `loss_reduction: prompt_mean` is hard-coded at `run_fully_async_terminus.sh:183` and was overridden only
for E (and the never-run S arm). The `ray_cluster.sh` gap noted 11-18 11:55 (a failed `ray start` is not detected) was
never fixed.

### C8. rv512 arms: final per-visit tables (numbers as of 06:38 CET 11-24)

512-task `calibforge_v2_rv512_1121` set, batch 32 groups x 16, LR 4e-6 unless noted, rollout_is, R3 routing replay on.

- Paired reward vs base = mean over tasks of (visit reward - base reference reward for that task) +- SE (n tasks).
- ctx = context-exceeded share of trials.
- Entropy, gap and grad norm ranges are over the steps trained while that visit was being dispatched. Gap =
  `minibatch_rollout_logprobs_abs_diff_mean`.
- Each full visit is about 10 trained steps, as measured from the step log (not derived).
- PARTIAL = the visit had not closed when the run ended. Its complete groups lean toward shorter tasks.

**C: prompt_mean, frozen router + mamba** (`nemo-jup-cf-rv512-b32-lr4e-6-ris-frz-r3-1121`)

| visit | groups | reward | vs base | ctx | median turns | reply tok/turn (mean / median) | end ctx mean / median | steps | entropy | gap | grad norm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 510 | 0.390 | -0.006 +- 0.006 (510) | 0.167 | 23 | 1151 / 288 | 61.5k / 49.5k | 1-10 | 0.052-0.059 | 0.0116-0.0133 | 0.057-0.079 |
| 2 | 489 | 0.402 | +0.003 +- 0.006 (489) | 0.174 | 23 | 1117 / 281 | 62.4k / 50.9k | 11-20 | 0.053-0.061 | 0.0114-0.0129 | 0.059-0.079 |
| 3 | 509 | 0.408 | +0.012 +- 0.006 (509) | 0.170 | 23 | 1071 / 279 | 62.5k / 50.2k | 21-30 | 0.058-0.067 | 0.0117-0.0130 | 0.062-0.091 |
| 4 | 484 | 0.421 | +0.022 +- 0.006 (484) | 0.164 | 23 | 1114 / 307 | 60.6k / 46.8k | 32-41 (step 31 has no metrics line paired with its save) | 0.062-0.075 | 0.0116-0.0135 | 0.059-0.091 |
| 5 | 501 | 0.415 | +0.019 +- 0.007 (501) | 0.180 | 24 | 1127 / 330 | 62.6k / 50.9k | 42-51 | 0.073-0.088 | 0.0124-0.0147 | 0.058-0.095 |
| 6 | 0 | - | (opened 11-24 06:11; stopped 06:15) | | | | | | | | |

Visit 5 closed when visit 6 opened at 06:11 on 11-24, so it is final. Visit 4 includes Dune-outage steps 36-37. The
11-23 17:57-18:00 proxy outage (228 setup failures) fell in visit 5.

**E: sequence_mean, frozen router + mamba** (`nemo-jup-cf-rv512-b32-lr4e-6-ris-frz-r3-seqm-1122`)

| visit | groups | reward | vs base | ctx | median turns | reply tok/turn | end ctx | steps | entropy | gap | grad norm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 512 | 0.393 | -0.001 +- 0.006 (512) | 0.164 | 21 | 1225 / 310 | 60.5k / 48.0k | 1-10 | 0.054-0.063 | 0.0118-0.0140 | 0.061-0.085 |
| 2 | 497 | 0.405 | +0.005 +- 0.006 (497) | 0.174 | 20 | 1347 / 336 | 61.0k / 48.2k | 11-20 | 0.056-0.060 | 0.0118-0.0134 | 0.059-0.083 |
| 3 | 488 | 0.422 | +0.024 +- 0.007 (488) | 0.190 | 21 | 1435 / 350 | 62.7k / 49.8k | 21-29 | 0.059-0.064 | 0.0124-0.0142 | 0.061-0.084 |
| 4 PARTIAL | 394 | 0.432 | +0.035 +- 0.008 (394) PARTIAL | 0.204 | 22 | 1440 / 338 | 64.7k / 53.0k | 30-35 | 0.061-0.065 | 0.0126-0.0139 | 0.055-0.079 |

**E - C, paired on common tasks (same task, same visit)**

| visit | n | reward | ctx | turns | reply tok/turn | end ctx |
|---|---|---|---|---|---|---|
| 1 | 510 | +0.005 +- 0.005 | -0.003 +- 0.004 | -2.4 +- 0.5 | +72 +- 16 | -0.9k +- 0.4k |
| 2 | 475 | -0.001 +- 0.006 | +0.004 +- 0.004 | -5.5 +- 0.5 | +232 +- 22 | -1.1k +- 0.4k |
| 3 | 487 | +0.013 +- 0.006 | +0.022 +- 0.006 | -7.2 +- 0.6 | +341 +- 25 | +0.5k +- 0.4k |
| 4 PARTIAL | 371 | +0.015 +- 0.008 | +0.025 +- 0.007 | -5.0 +- 0.6 | +375 +- 30 | +1.8k +- 0.6k |

**D2: prompt_mean, nothing frozen (lr_zero [])** (`nemo-jup-cf-rv512-b32-lr4e-6-ris-nofrz-r3-1122b`)

| visit | groups | reward | vs base | ctx | median turns | reply tok/turn | end ctx | steps | entropy | gap | grad norm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 508 | 0.392 | -0.006 +- 0.006 (508) | 0.165 | 21 | 1244 / 313 | 60.7k / 48.4k | 1-10 | 0.054-0.067 | 0.0121-0.0157 | 0.061-0.079 |
| 2 | 506 | 0.404 | +0.007 +- 0.006 (506) | 0.231 | 21 | 1710 / 394 | 67.1k / 56.1k | 11-19 | 0.067-0.091 | 0.0132-0.0149 | 0.057-0.083 |
| 3 PARTIAL | 263 | 0.385 | -0.013 +- 0.011 (263) PARTIAL | 0.317 | 20 | 2232 / 459 | 75.6k / 71.8k | 20-23 | 0.103-0.115 | 0.0142-0.0146 | 0.047-0.053 |

D2 - C: v1 +0.001 +- 0.005 (508), v2 +0.001 +- 0.006 (486), v3 PARTIAL -0.025 +- 0.011 (262). Paired D2 v3 - C v3
(257 tasks): ctx +0.120 +- 0.012, turns -12.6 +- 1.1, reply +1199 +- 59 tok/turn, end ctx +10.3k +- 0.8k.

**B: router frozen, Mamba mixers trained** (`nemo-jup-cf-rv512-b32-lr4e-6-ris-rfrz-r3-1121`)

| visit | groups | reward | vs base | ctx | median turns | reply tok/turn | end ctx | steps | entropy | gap | grad norm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 510 | 0.392 | -0.004 +- 0.006 (510) | 0.172 | 22 | 1223 / 297 | 62.6k / 50.7k | 1-10 | 0.053-0.067 | 0.0114-0.0144 | 0.059-0.087 |
| 2 | 499 | 0.404 | +0.008 +- 0.006 (499) | 0.199 | 24 | 1271 / 285 | 65.6k / 55.2k | 11-20 | 0.067-0.096 | 0.0129-0.0151 | 0.054-0.077 |

B - C: v1 +0.003 +- 0.006 (509), v2 +0.005 +- 0.006 (479).

**A: LR 1e-5, prompt_mean, frozen router + mamba** (`nemo-jup-cf-rv512-b32-lr1e-5-ris-frz-r3-1121`)

| visit | groups | reward | vs base | ctx | median turns | reply tok/turn | end ctx | steps | entropy | gap | grad norm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 484 | 0.381 | -0.011 +- 0.006 (484) | 0.194 | 20 | 1437 / 300 | 62.8k / 49.6k | 1-9 | 0.056-0.064 | 0.0116-0.0177 | 0.057-0.084 |
| 2 | 511 | 0.363 | -0.033 +- 0.008 (511) | 0.324 | 20 | 1965 / 300 | 74.5k / 69.0k | 10-18 | 0.064-0.079 | 0.0126-0.0175 | 0.056-0.076 |

A - C: v1 -0.006 +- 0.006 (482), v2 -0.035 +- 0.007 (489).

Visits completed: C 5; E 3 plus a partial visit 4; D2 stopped in visit 3; A and B after visit 2. D (attempt 1,
`nemo-jup-cf-rv512-b32-lr4e-6-ris-nofrz-r3-1122`) trained 1 step and is not used.

JUPITER's summary of the rv512 results, in its words:

- Only the frozen router + mamba arms improve over base. C peaks at +0.022 +- 0.006 (v4) and E at +0.024 +- 0.007
  (v3), with E's partial v4 at +0.035 +- 0.008.
- E sits about +0.013 above C from visit 3, but its replies are longer and its ctx share grows.
- Unfreezing the router (D2) or only the mixers (B) brings no gain. D2's entropy and reply length run away by visit 3.
- LR 1e-5 (A) is negative by visit 2.
- No arm moved far from base: the largest gain is a few hundredths on a ~0.39 base.
