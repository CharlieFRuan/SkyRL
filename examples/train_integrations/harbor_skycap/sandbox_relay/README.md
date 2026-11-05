# Sandbox relay: agents in remote sandboxes reaching a cluster they can't connect to

Status: review draft, 2026-11-05. Code on branch `skycap-evals-1104` (mainline `f854d6d9` plus this port); history and
measurements in the cluster notes (`ops/TUNNEL_CRASH.md`, `ops/dune/`, `ops/evals_1104/PLAN.md`).

## 0. Context: how we got here (for a reader without the history)

This work spans four clusters and several weeks (2026-10-08 to 11-05). Everything a new reader can reach is linked below.
W&B runs are in entity `sky-posttraining-uc-berkeley` (needs access). The MiMo task dataset is private on Hugging Face.

### MiMo task data (before any RL)
- **Repo:** [`CharlieFRuan/rl-env-cleaner`](https://github.com/CharlieFRuan/rl-env-cleaner), mostly `mimo_swe/`. It holds
  the cleaning and provisioning of MiMo-v2.6 SWE tasks for Harbor:
  - `HANDOFF.md`, `FINDINGS.md`;
  - `QUALITY_PIPELINE.md` (on branch `qwen36-b200-run`);
  - `RESIDUE_AUDIT.md`, `sandbox_provision_pass.md`;
  - `results_qwen36/`.
- **Data:** Hugging Face dataset `CharlieFRuan/mimo-v2.6-harbor-tasks` (private).
- **Other branches:** `cyber-b200-run` adds an inventory of the MiMo cyber tasks (`mimo_cyber/RECONNAISSANCE.md`).
- **Sandbox environment:** `mimo_swe/labeled_daytona.py` was the original Daytona environment class (labels, TTL,
  agent-phase allowlist). Its relay-stub start became `environment.py` here.

### 1. GCP B200 cluster (2026-10-08 to 10-11): SkyCap + mini-swe-agent RL on Qwen3.6-35B-A3B
- **Code:** SkyRL fork branch
  [`skycap-mini-swe`](https://github.com/CharlieFRuan/SkyRL/tree/skycap-mini-swe) (on NovaSky-AI/SkyRL main `b8963321`).
  - Recipe: `run_overfit_qwen36_mini_swe.sh`.
  - **Parity:** SkyCap matches stock Harbor + vLLM on SWE-bench Verified 100
    ([`PARITY_SWEBENCH.md`](https://github.com/CharlieFRuan/SkyRL/blob/skycap-mini-swe/examples/train_integrations/harbor_skycap/PARITY_SWEBENCH.md))
    and on Terminal-Bench (TB-Lite, TB 2.1; `[skycap] PARITY` commits on that branch).
- **How sandboxes reached the model:**
  - First a Cloudflare quick tunnel: 125 s response cap, ~200 calls in flight.
  - Then a **relay VM** on GCP in the cluster's own VPC: Caddy plus an `sslip.io` hostname, since Daytona's allowlist is
    hostname-based.
  - Details: [`mimo_swe/model_endpoint_relay.md`](https://github.com/CharlieFRuan/rl-env-cleaner/blob/skycap-overfit/mimo_swe/model_endpoint_relay.md).
  - This works only when the relay VM can route to the GPU nodes.
- **Overfit runs**, W&B project `skycap-overfit`. The results table is in
  [`skycap_overfit/cross_cluster_handoff.md` §8](https://github.com/CharlieFRuan/rl-env-cleaner/blob/skycap-overfit/skycap_overfit/cross_cluster_handoff.md):
  - 4 tasks × 16: [8arpmq6z](https://wandb.ai/sky-posttraining-uc-berkeley/skycap-overfit/runs/8arpmq6z),
    [bed481kr](https://wandb.ai/sky-posttraining-uc-berkeley/skycap-overfit/runs/bed481kr),
    [e939ah4g](https://wandb.ai/sky-posttraining-uc-berkeley/skycap-overfit/runs/e939ah4g),
    [evstgyyg](https://wandb.ai/sky-posttraining-uc-berkeley/skycap-overfit/runs/evstgyyg),
    [flwlwm8g](https://wandb.ai/sky-posttraining-uc-berkeley/skycap-overfit/runs/flwlwm8g) (R3).
  - 16 fast tasks × 8: [02dwfjb3](https://wandb.ai/sky-posttraining-uc-berkeley/skycap-overfit/runs/02dwfjb3),
    [vci3ga5w](https://wandb.ai/sky-posttraining-uc-berkeley/skycap-overfit/runs/vci3ga5w) (peak 0.98).
  - 1 node, lr 1e-6: [ohm8trpo](https://wandb.ai/sky-posttraining-uc-berkeley/skycap-overfit/runs/ohm8trpo).
- **Handoff to other clusters:**
  [`skycap_overfit/cross_cluster_handoff.md`](https://github.com/CharlieFRuan/rl-env-cleaner/blob/skycap-overfit/skycap_overfit/cross_cluster_handoff.md).

### 2. JSC JUPITER (2026-10-12 to 10-13): the same runs on aarch64 GH200, offline compute nodes
- **Code:** SkyRL fork branches [`jupiter`](https://github.com/CharlieFRuan/SkyRL/tree/jupiter) (aarch64 deps, CP knob,
  Qwen3.5 renderer knob) and `jupiter-aarch64-main`; rl-env-cleaner branch `jupiter`, documented in
  [`skycap_overfit/jupiter/README.md`](https://github.com/CharlieFRuan/rl-env-cleaner/blob/jupiter/skycap_overfit/jupiter/README.md).
- **Network:**
  - Outbound: a CONNECT proxy on a login node.
  - Inbound: a GCP relay VM (Caddy), reached by a chisel client on the login node through a SOCKS gate that admits only
    the user's running Slurm nodes.
  - This was the last design that needed a tunnel binary.

### 3. Anyscale cluster, AWS ap-south-2, 4 × 8 B200 (2026-10-13 to 11-05): where this relay was born
The notes for this period, sanitized (no IPs, instance IDs or credentials), are in rl-env-cleaner branch
[`anyscale-b200-run`](https://github.com/CharlieFRuan/rl-env-cleaner/tree/anyscale-b200-run/anyscale_b200), with an
index in its README:
- `notes/RUN_NOTES.md`: run-by-run history.
- `notes/TUNNEL_CRASH.md`: networking, chisel and the relay design.
- `dune/DUNE_EVAL.md`: Dune Cloud.
- `upstream_mismatch/REPORT.md`: the Nemotron mismatch.
- `evals_1104/PLAN.md`: the November evals.

The essentials:
- **Why the relay exists.**
  - Nodes are Kubernetes pods with no ingress, and the GCP relay VM has no route to them.
  - Reverse SSH is reset by the egress filter.
  - Downloading or running the **chisel** binary got a node killed within ~15 s by the platform's security scanner,
    reproduced deliberately; on the head node it took down the whole cluster.
  - `cloudflared` survived but needs `--protocol http2`.
  - Hence Colby's design (§2). A survey of other RL frameworks found none that handles "remote sandbox + outbound-only
    cluster" with an in-sandbox agent.
- **Relay stress test** (2026-10-13): SWE-bench Verified 100 × 5 at 300 concurrent: 0.654 vs 0.663 (relay VM) and 0.673
  (stock Harbor), 0 call errors (§4).
- **Qwen3.6 fully async RL** on MiMo, through the relay, W&B project `skycap-mimo-fully-async`:
  - Run 1: [08c34mur](https://wandb.ai/sky-posttraining-uc-berkeley/skycap-mimo-fully-async/runs/08c34mur).
  - Run 2: [14wm4ep6](https://wandb.ai/sky-posttraining-uc-berkeley/skycap-mimo-fully-async/runs/14wm4ep6),
    [8klyboql](https://wandb.ai/sky-posttraining-uc-berkeley/skycap-mimo-fully-async/runs/8klyboql) and
    [okvj1xcm](https://wandb.ai/sky-posttraining-uc-berkeley/skycap-mimo-fully-async/runs/okvj1xcm).
  - Run 2 relayed ~2.75M calls; SWE-bench-100 eval was flat at ~0.70 over 120 steps.
  - The delta encoding (§2) came from this run, when long histories made calls slow.
- **Nemotron-3.5-Lightning-30B-A3B:**
  - **Parity.** SkyCap vs stock Harbor on SWE-bench Verified 100.
  - **Overfits.** W&B `skycap-overfit`, runs `nem35-*`.
  - **Logprob-mismatch investigation.** Root cause: SkyRL's weight sync rounded the fp32 router `e_score_correction_bias`
    to bf16. Mainline later fixed it in NovaSky-AI/SkyRL #2543.
    - W&B `nemotron_3_5_lightning_30b_grpo`: before [fc1kdvsd](https://wandb.ai/sky-posttraining-uc-berkeley/nemotron_3_5_lightning_30b_grpo/runs/fc1kdvsd),
      after [63aqqnsp](https://wandb.ai/sky-posttraining-uc-berkeley/nemotron_3_5_lightning_30b_grpo/runs/63aqqnsp).
    - Mainline A/B: [5xr52ubp](https://wandb.ai/sky-posttraining-uc-berkeley/nemotron_3_5_lightning_30b_grpo/runs/5xr52ubp)
      vs [c5jy53xs](https://wandb.ai/sky-posttraining-uc-berkeley/nemotron_3_5_lightning_30b_grpo/runs/c5jy53xs).
    - Cross-framework checks: W&B `nemotron-mismatch-xfw`.
  - **A 3-epoch fully async run** (W&B `skycap-mimo-fully-async`,
    [5doiopns](https://wandb.ai/sky-posttraining-uc-berkeley/skycap-mimo-fully-async/runs/5doiopns) and
    [uc73dejf](https://wandb.ai/sky-posttraining-uc-berkeley/skycap-mimo-fully-async/runs/uc73dejf)).
  - **Final evals:** SWE-bench Verified 500 × 5 at base 50.3 / step 218 51.8 (card 50.7); Terminal-Bench 2.1 at base 26.3.
- **This port** (2026-11-04 to 11-05): the fork's SkyCap and relay work moved onto current mainline (this PR).
  - SWE-bench Verified and Terminal-Bench evals of Nemotron-3.5-Lightning and Qwen3.5-9B, with and without SkyCap.
  - These ran on **Dune Cloud**, because the Daytona key was invalid at the time. Dune has no preview URLs, hence the
    exec transport.

## 1. The problem

With Harbor's installed agents (mini-swe-agent, Claude Code, Codex), the agent runs **inside** the task's sandbox
(Daytona, Modal, Dune) and calls the model over HTTP. skycap, which renders every prompt and records each rollout's
tokens, runs on the training cluster. So every model call is a connection **from the sandbox to the cluster**.

On the Anyscale cluster these evals and RL runs used, that connection is impossible:
- **No inbound route.** Nodes are Kubernetes pods with cluster-internal addresses (`100.64.0.0/10`), no public IP
  and no ingress. A sandbox on the public internet cannot reach them.
- **Tunnels are out:**
  - **GCP relay VM.** Our earlier setup put a VM with Caddy on GCP, in the same VPC as the GPU nodes. Here it has no
    route into AWS/Anyscale pods.
  - **Reverse SSH.** The cluster's egress resets SSH to the relay VMs.
  - **chisel**, a reverse tunnel over HTTPS. A security scanner kills any node the chisel binary lands on, within
    ~15 s (reproduced on 2026-10-13; on the head it takes the whole cluster down).
  - **Cloudflare quick tunnels.** `cloudflared` survives (it needs `--protocol http2`; UDP egress is blocked), but a
    quick tunnel caps in-flight requests (~200) and has no SLA.

(Terminus-2 is not affected: it calls the model from the trial's Ray worker on the cluster and only sends commands
into the sandbox.)

## 2. The solution (Colby's design): every connection starts on the cluster

Instead of the sandbox calling the cluster, the cluster **pulls** the sandbox's calls, through the sandbox provider's
own API, which the cluster already talks to for every trial (create, exec, upload).

```
inside the sandbox                                           on the cluster (outbound connections only)
┌──────────────────────────────┐                             ┌────────────────────────────────────────────┐
│ mini-swe-agent               │                             │ skycap server actor                        │
│   base_url =                 │                             │   ┌──────────────────────┐                 │
│   http://127.0.0.1:9876/t/<id>/v1                          │   │ relay manager        │──▶ skycap ──▶ vLLM
│        │                     │                             │   │  long-polls stubs,   │   (token mode)  │
│        ▼                     │   1. poll: "calls?"         │   │  forwards each call, │                 │
│ stub.py (stdlib, 127.0.0.1)  │ ◀──────────────────────────── │  posts the reply     │                 │
│   holds each call open ──────│ ──────── 2. call ─────────▶ │   └──────────────────────┘                 │
│   until its reply arrives ◀──│ ◀─────── 3. reply ────────── │                                          │
└──────────────────────────────┘   via the provider:         └────────────────────────────────────────────┘
                                   signed preview URL (Daytona) or exec API (Dune)
```

1. When a trial's sandbox starts, the environment class writes `stub.py` into it and starts it on `127.0.0.1:9876`
   as root, before the agent is installed, with a random secret. It then hands the relay manager the sandbox's
   address (`/attach`).
2. The agent is pointed at the stub. `OPENAI_API_BASE` and friends point to `http://127.0.0.1:9876/t/<trajectory>/v1`.
   The stub queues each request and holds the agent's connection open.
3. The relay manager, one per skycap server, in the same actor, keeps one long-poll in flight per sandbox. It picks
   up queued calls, forwards each to its skycap server's harness route, and posts the reply back to the stub, which
   answers the agent.

Nothing runs on the cluster beyond our own Python making HTTPS requests to the provider. There is no tunnel binary,
no public port and no VM.

### Two transports to the stub

| | `preview` (default) | `exec` |
|---|---|---|
| For | Daytona | providers without preview URLs (Dune Cloud) |
| How the manager reaches the stub | HTTPS to the sandbox's **signed preview URL** for port 9876 | the provider's **exec API**: runs a tiny client (`rx.py`, stdlib) in the sandbox that talks to the stub on 127.0.0.1 and prints framed, gzipped, base64 output |
| Poll | `GET <preview>/_relay/poll?timeout=20` | `exec("python rx.py poll <timeout> <ack>")`, held in the sandbox |
| Reply | `POST <preview>/_relay/reply` | `exec` with the reply inline (< 512 KB) or `fs.upload_file` then `exec` |
| Lost answers | HTTP retry | `rx.py` spools every non-empty poll answer until the manager acknowledges it, and replays it on the next poll |
| Overhead per call | ~0.27 s (with deltas; cluster in AWS ap-south-2, Daytona proxy in us-east-1) | ~0.37 s (2 KB replies), ~0.74 s (200 KB) |
| Code | `manager.py:SandboxRelayManager`, `environment.py:RelayDaytonaEnvironment` | `exec_transport.py:ExecRelayManager`, `exec_environment.py:ExecRelayDaytonaEnvironment` |

### What is a preview URL?

Daytona can expose any port of a sandbox (3000-9999) as a public HTTPS URL served by Daytona's preview proxy
(host `*.daytonaproxy01.net` for our org; the SDK returns the exact URL). Requests to it are proxied into that port.
- `get_preview_link(port)` returns the URL plus a token you send as a header.
- `create_signed_preview_url(port, expires_in_seconds)` returns a URL with the token embedded, valid until it expires.

The relay manager long-polls the stub through the signed URL. So the **cluster** opens the connection, to Daytona's
public proxy, and the sandbox never connects to anything. Dune does not support preview URLs (422 "Preview URLs are
not supported for this sandbox"), hence the exec transport.

Nuances:
- **Warning page.** The proxy answers with a browser warning page instead of proxying, unless the request carries
  `X-Daytona-Skip-Preview-Warning: true`. `manager.py` sends it on every request (`PREVIEW_HEADERS`).
- **Expiry.** `RelayDaytonaEnvironment` signs the URL for the sandbox's `ttl_minutes` (default 360 min when none is
  set). A sandbox that outlives its signed URL can no longer be polled. Keep the TTL above the longest trial
  (build + agent + verifier).
- **Public, so authenticated.** Anyone with the URL can reach the port. Hence the per-sandbox `X-Relay-Secret` on
  every `/_relay/*` route (§ Protocol details).
- **Scope.** A preview URL reaches one port of one sandbox, through Daytona's own proxy. Nothing on the cluster
  becomes reachable from outside, unlike a tunnel, which opens the cluster network to whoever holds the far end.
  Miles uses the same pattern to reach an OpenEnv server inside a Daytona sandbox (`create_signed_preview_url(8000)`).
- **Throughput.** Bodies cross the proxy at a few hundred KB/s (~170 KB/s measured), which is why calls go as deltas
  and polls are gzipped.

### Protocol details that matter

- **Auth.** Every `/_relay/*` route needs the `X-Relay-Secret` header (random per sandbox, given at startup), so
  whoever learns the preview URL can neither read the agent's prompts nor answer them.
- **Deltas.**
  - mini-swe-agent resends its whole history on every call, and a body crosses the provider's proxy at only
    ~170 KB/s, so a 300 KB history cost 2.7 s per call.
  - The stub now sends a call that extends the previous one on the same path as a delta: base id, shared prefix
    length, the new bytes. The manager rebuilds the body, or fetches it whole (`/_relay/body`) when it lacks the base.
  - Poll responses are gzipped.
  - Result: 0.27 s per call at ~500 sandboxes, with 99% of calls sent as deltas.
- **Gone sandboxes.** A sandbox whose polls fail for `give_up_after` (600 s) is dropped. `/detach` on trial stop.
- **Stats.** `GET /stats` on each manager splits each call's time: skycap's (`mean_forward_seconds`), what the agent
  waited as the stub measured it (`mean_agent_seconds`), and the relay's (`mean_overhead_seconds`).
- **Interpreter.** The stub runs on the image's `python3` if it is >= 3.6. Otherwise a uv-managed Python is installed
  first, after installing curl if the image has neither curl nor wget (some Terminal-Bench images).

## 3. Code (branch `skycap-evals-1104`)

| File | What |
|---|---|
| `sandbox_relay/stub.py` | sandbox side: the stand-in server the agent calls; stdlib only, Python 3.6+ |
| `sandbox_relay/manager.py` | cluster side: `SandboxRelayManager` (aiohttp, own thread): `/attach`, `/detach`, `/stats`, the polling and forwarding |
| `sandbox_relay/environment.py` | `RelayDaytonaEnvironment` (subclass of `harbor_skycap/daytona.py:LabelledDaytonaEnvironment`): starts the stub, creates the signed preview URL, attaches. Also ephemeral sandboxes, labels and TTL. |
| `sandbox_relay/exec_transport.py` | `ExecRelayManager` (the exec transport) and the in-sandbox `rx.py` client |
| `sandbox_relay/exec_environment.py` | `ExecRelayDaytonaEnvironment`: starts the stub and attaches by sandbox id instead of a preview URL |
| `servers.py` | each skycap server actor also runs a relay manager (`relay=` options; `transport` picks the class); `SkycapServers.relay_urls` |
| `entrypoints/main_harbor_skycap.py` | config `skycap.sandbox_relay.{enabled, port, poll_timeout, give_up_after, transport}`; it replaces `skycap.exposure` |
| `harbor_generator.py` | per trial: the agent URL on the stub (`_agent_url`), and the environment kwarg `skycap_relay` = {manager, target server, port} (`_relay_kwargs`) |
| `run_eval_skycap.sh` | `RELAY_TRANSPORT=preview\|exec` |
| tests | `tests/integrations/harbor_skycap/test_sandbox_relay.py`, `test_sandbox_relay_exec.py` |

Without skycap (a stock `harbor run` baseline against plain vLLM), the same stub and manager are used. The manager
forwards to an OpenAI-compatible endpoint instead (cluster script `ops/relay_manager_standalone.py`, not in the repo),
and the trial gets the relay through `--ek skycap_relay=...`.

Usage:

```bash
# Daytona (signed preview URLs)
skycap.exposure.type=none skycap.sandbox_relay.enabled=true \
harbor_trial_config.environment.import_path=examples.train_integrations.harbor_skycap.sandbox_relay.environment:RelayDaytonaEnvironment \
harbor_trial_config.environment.kwargs.labels.owner=<you> harbor_trial_config.environment.kwargs.labels.run=<run>
# Dune: also skycap.sandbox_relay.transport=exec and ...sandbox_relay.exec_environment:ExecRelayDaytonaEnvironment;
# the skycap actors need DAYTONA_API_URL / DAYTONA_API_KEY of the same account.
```

## 4. What has been tested

### Unit tests (CPU; 44 passed with the generator tests, 2026-11-04)
- `test_sandbox_relay.py`:
  - calls go through and the stats split the agent's wait;
  - a poll without the secret is refused;
  - a growing history crosses as deltas and arrives whole;
  - common-prefix edge cases.
- `test_sandbox_relay_exec.py`, against a fake sandbox whose exec runs locally:
  - calls, deltas and large (uploaded) replies cross by exec;
  - a poll answer lost on the way back is replayed;
  - setup fails cleanly without a stub;
  - frame parsing ignores noise.

### Preview transport on Daytona (the fork's code, 2026-10-13 to 10-19; ported unchanged in logic)

| Test | Result |
|---|---|
| Stress: SWE-bench Verified 100 x 5, mini-swe-agent, **300 concurrent** (`swe100-relay-c300-x5-1013-1709`) | 0.654, vs 0.663 (sd 0.027) for the old GCP relay-VM setup and 0.673 for stock Harbor. 33,008 calls, **0 call errors**, 3 transient poll errors (retried), 0 sandboxes dropped. |
| Qwen3.6-35B-A3B fully async RL run (~450-500 concurrent sandboxes, days) | 2.75M relayed calls, 8 call errors, 1 dropped sandbox; overhead 0.27-0.28 s per call after the delta fix |
| Nemotron-3.5-Lightning fully async RL run (3 epochs, 218 steps) and its final evals (SWE-bench Verified 500 x 5, Terminal-Bench 2.1 x 5, stock Harbor via the standalone manager) | ran through the relay end to end; base model SWE-bench Verified 50.3 vs the card's 50.7 |
| Overhead breakdown | ~0.44 s per call before deltas (geography: AWS ap-south-2 ↔ Daytona's proxy in us-east-1, ~235 ms RTT); 0.27 s with deltas and gzip |

### Exec transport on Dune (the port, 2026-11-04 / 05)

| Test | Result |
|---|---|
| End to end, 1 and 20 sandboxes, fake target on the cluster | 372 calls, all bodies byte-correct; median overhead 0.37 s (2 KB) / 0.74 s (200 KB); attach ~1.2 s |
| SkyCap SWE-bench smoke (4 tasks) | 4/4 complete, 0 unbridged calls |
| Stock Harbor baseline, Nemotron, SWE-bench Verified 500 x 3, 300 concurrent | ~48k calls, **1 call error, 0 poll errors**; 0.481 over clean trials (card 50.7) |
| SkyCap, Nemotron, same | 0.474 over clean trials: parity with the baseline |

### Not yet tested
- **The preview transport on the mainline port** has not run end to end against Daytona. Our Daytona key was invalid
  while the port was done. The logic is the fork's, moved unchanged, and the unit tests pass. A short Daytona smoke
  test is the next step.
- The exec transport above ~300 sandboxes per manager group (Dune itself was unstable around 750 concurrent sandboxes;
  see `ops/dune/DUNE_EVAL.md`).
- The relay combined with a network-restricted sandbox (§5).

## 5. Network restrictions: does the sandbox need egress to the model?

**No.** The agent only talks to `127.0.0.1`, and the model traffic crosses the provider's control plane (the preview
proxy or the exec API), pulled by the cluster. So the sandbox needs **no egress to the inference endpoint**, and no
allowlist entry for it. That is a property of the design, not a setting.

What the sandbox may still need the internet for:
- **Installing the agent.** Harbor runs `uv tool install mini-swe-agent`, which reaches PyPI and astral.sh. Use an
  allowlist (PyPI, astral.sh, GitHub) or bake the agent into the image / a snapshot.
- **The stub's interpreter**, only when the image lacks Python 3.6+. That means a uv download, or apt/apk for curl.
- **The verifier.** Some task verifiers need the network (about 18% of MiMo tasks did).

Measured on Dune (2026-11-05, one sandbox, `update_network_settings`):

| Setting | pypi.org | github.com | api.openai.com | exec API |
|---|---|---|---|---|
| open | 200 | 200 | 421 (reachable) | works |
| `domain_allow_list="pypi.org"` | 200 | blocked | blocked | works |
| `network_block_all=True` | blocked | blocked | blocked | works |
| back to open (`network_block_all=False, domain_allow_list=""`) | 200 | 200 | reachable | works |

A CIDR `network_allow_list` at create time is accepted too.
- **Dune:** the exec transport keeps working with all egress blocked, so with the agent pre-installed a fully offline
  sandbox would still reach the model.
- **Daytona:** domain allowlists worked in our October runs (MiMo agent phase), with the same need to clear the list
  when restoring.
- **Not tested:** whether Daytona's preview proxy still reaches a sandbox with `network_block_all`. It is inbound
  through Daytona's own proxy, so it most likely does.

## 6. Known limits and open questions
- **Overhead is geography-bound.** A call costs about one round trip between the cluster and the provider's proxy
  (~0.27 s here). Co-locating the cluster and the sandboxes (same region) is the biggest lever.
- **Scaling.** Each relay manager shares its skycap server's process (and GIL). 8 servers handled ~500 sandboxes at
  ~10-13% of a core each; raise `skycap.num_servers` beyond that.
- **Exec transport.** One exec held open per sandbox, plus two execs per call. Each provider's API limits for that are
  unknown; ramp gradually. A lost poll answer can make one call generate twice (the duplicate reply is rejected
  harmlessly).
- **skycap trajectory TTL.** A trajectory opens before the sandbox is built. If the build takes longer than
  `skycap.ttl` (default 3600 s), the agent's first call gets `trajectory_closed` (127 trials hit this during a slow
  Dune hour). `run_eval_skycap.sh` now passes `skycap.ttl=14400`.
- **Secret.** The stub's secret is visible inside the sandbox (to the agent), which could in principle answer its own
  calls. It can't reach the cluster, so it can't do more than that.
