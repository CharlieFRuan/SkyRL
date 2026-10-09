"""mini-swe-agent through skycap: what its LiteLLM client sends and gets back, and how a trial that ran
out of context is scored.

The client side is real LiteLLM with mini-swe-agent's settings (``openai/`` provider, the shared
``mini_swe_agent_qwen36.yaml``), replaying each reply as mini-swe-agent does
(``message.model_dump()``); the model side is a canned SkyRL generate endpoint, recording exactly what
reaches vLLM.
"""

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import pytest_asyncio
import yaml

pytest.importorskip("skycap")
pytest.importorskip("harbor")
litellm = pytest.importorskip("litellm")

import aiohttp  # noqa: E402
import orjson  # noqa: E402
from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestServer  # noqa: E402

from examples.train_integrations.harbor_skycap.engine import SkyRLEngine  # noqa: E402
from examples.train_integrations.harbor_skycap.harbor_generator import (  # noqa: E402
    HarborSkycapGenerator,
)
from skycap import CapturePool, CaptureService, record  # noqa: E402
from skycap.exposure import Exposure  # noqa: E402
from tests.integrations.harbor_skycap import test_harbor_skycap  # noqa: E402
from tests.integrations.harbor_skycap.fakes import FakeRenderer  # noqa: E402
from tests.integrations.harbor_skycap.test_harbor_skycap import (  # noqa: E402
    batch,
    generator_cfg,
    harbor_cfg,
)

pytestmark = pytest.mark.integrations

router = test_harbor_skycap.router
trials = test_harbor_skycap.trials

MSWEA_CONFIG = Path(__file__).parents[3] / "examples/train_integrations/harbor_skycap/mini_swe_agent_qwen36.yaml"
#: The trainer's sampling, which skycap imposes on every call (generator.sampling_params).
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}


class LoopbackExposure(Exposure):
    def start(self, harness_url: str) -> str:
        return harness_url


def mini_swe_agent_cfg() -> dict:
    cfg = harbor_cfg()
    cfg["agent"]["name"] = "mini-swe-agent"
    return cfg


# -- context overflow -------------------------------------------------------------------------
@pytest.fixture
def small(router, tmp_path):
    """A server whose model holds 1024 tokens, exposed to agents in sandboxes."""
    service = CaptureService(
        router.url,
        mode="tokens",
        renderer=FakeRenderer(),
        engine=SkyRLEngine(),
        model="policy",
        max_model_len=1024,
        record_dir=str(tmp_path / "record"),
        host="127.0.0.1",
        exposure=LoopbackExposure(),
        require_api_key=True,
    )
    service.start()
    yield service
    service.stop()


@pytest.mark.asyncio
async def test_an_installed_agent_out_of_context_stops_on_context_length_scored_by_its_verifier(small, trials) -> None:
    gen = HarborSkycapGenerator(
        generator_cfg(), mini_swe_agent_cfg(), [small.url], SimpleNamespace(weight_version=7), train_paths="all"
    )
    try:
        outcomes, _ = await gen.run_trials(batch("overflow", "overflow_unverified"), disable_tqdm=True)
        out = await gen.generate(batch("overflow"), disable_tqdm=True)
    finally:
        await gen.close()

    verified, unverified = outcomes
    # Harbor ran the verifier after the agent exited non-zero: its reward is the trial's, as stock Harbor scores it.
    assert (verified.stop_reason, verified.reward, verified.verifier_reward) == ("context_length", 1.0, 1.0)
    assert (unverified.stop_reason, unverified.reward, unverified.verifier_reward) == ("context_length", 0.0, None)
    assert verified.exception == "NonZeroAgentExitCodeError" and verified.samples
    # Trained (not masked), with the trial's reward; overlong filtering is off here.
    assert out["stop_reasons"] == ["context_length"] and out["rewards"] == [1.0] and sum(out["loss_masks"][0]) > 0
    # skycap recorded the refusal with its code, and finished the trajectory with the right annotations.
    document = record.read_document(small.server.record_dir, verified.skycap_id)
    assert [f["code"] for f in document["failures"]] == ["context_length_exceeded"]
    assert document["annotations"] == {"reward": 1.0, "stop_reason": "context_length"}


@pytest.mark.asyncio
async def test_litellm_raises_context_window_exceeded_on_the_refusal_and_finish_reports_it(small) -> None:
    async with CapturePool([small.url]) as pool:
        trajectory = await pool.create()
        messages = [{"role": "user", "content": "hi"}]
        await litellm.acompletion(
            model="openai/policy", messages=messages, api_base=trajectory.base_url, api_key=trajectory.api_key
        )
        with pytest.raises(litellm.ContextWindowExceededError):
            await litellm.acompletion(
                model="openai/policy",
                messages=[{"role": "user", "content": "x" * 4096}],
                api_base=trajectory.base_url,
                api_key=trajectory.api_key,
                num_retries=0,
            )
        finished = await trajectory.finish({"reward": 0.0})
    assert finished.context_length_exceeded and finished.status == "finished"


# -- Qwen3.6, thinking kept, through LiteLLM -----------------------------------------------------
LOCAL = "/mnt/local_storage/charlieruan/hf/Qwen3.6-35B-A3B"
TOKENIZER = os.environ.get("SKYCAP_QWEN36_TOKENIZER") or (LOCAL if os.path.isdir(LOCAL) else "Qwen/Qwen3.6-35B-A3B")


def bash(command: str) -> str:
    return f"<tool_call>\n<function=bash>\n<parameter=command>\n{command}\n</parameter>\n</function>\n</tool_call>"


#: Sampled after ``<|im_start|>assistant\n<think>\n``: reasoning, text and a call; a call alone; then an answer.
COMPLETIONS = [
    "Look at the layout.\n</think>\n\nListing files.\n\n" + bash("ls -la"),
    "setup.py is there.\n</think>\n\n" + bash("cat setup.py | head -50"),
    "Done.\n</think>\n\nAll fixed.",
]


class CannedEngine:
    """SkyRL's ``/skyrl/v1/generate``, answering each call with the next canned completion."""

    def __init__(self, completions: List[List[int]]) -> None:
        self.completions = iter(completions)
        self.requests: List[Dict[str, Any]] = []

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/skyrl/v1/generate", self.generate)
        app.router.add_post("/finish_session", self.finish_session)
        return app

    async def finish_session(self, request: web.Request) -> web.Response:
        return web.json_response({})

    async def generate(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.requests.append(body)
        completion = next(self.completions)
        choice = {
            "token_ids": completion,
            "finish_reason": "stop",
            "logprobs": {"content": [{"logprob": -0.5} for _ in completion]},
        }
        return web.Response(body=orjson.dumps({"choices": [choice]}), content_type="application/json")


@pytest.fixture(scope="module")
def qwen36():
    pytest.importorskip("renderers")
    from renderers.base import load_tokenizer

    from skycap.tokens.renderer import RenderersRenderer

    template = {"enable_thinking": True, "preserve_thinking": True}
    try:
        renderer = RenderersRenderer(
            TOKENIZER, size=1, thinking_retention="all", chat_template_kwargs=template, renderer="qwen3.6"
        )
    except Exception as error:  # noqa: BLE001 - no network and no cache
        pytest.skip(f"Qwen3.6 tokenizer unavailable: {error}")
    return renderer, load_tokenizer(TOKENIZER)


@pytest_asyncio.fixture
async def canned(qwen36):
    renderer, tokenizer = qwen36
    end = renderer.stop_token_ids()[0]
    engine = CannedEngine([tokenizer.encode(text, add_special_tokens=False) + [end] for text in COMPLETIONS])
    server = TestServer(engine.app())
    await server.start_server()
    service = CaptureService(
        str(server.make_url("")).rstrip("/"),
        mode="tokens",
        renderer=renderer,
        engine=SkyRLEngine(),
        model="qwen3.6-35b-a3b",
        max_model_len=73728,
        sampling_overrides=SAMPLING,
        host="127.0.0.1",
        require_api_key=True,
    )
    service.start()
    yield engine, service, renderer, tokenizer
    await asyncio.to_thread(service.stop)
    await server.close()


#: mini-swe-agent 2.4.6's one tool (minisweagent.models.utils.actions_toolcall.BASH_TOOL).
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
            "required": ["command"],
        },
    },
}


@pytest.mark.asyncio
async def test_qwen36_thinking_survives_litellms_openai_provider_and_sampling_reaches_vllm(canned) -> None:
    engine, service, renderer, tokenizer = canned
    # As mini-swe-agent calls LiteLLM: its model kwargs from the shared config, plus a penalty to see it pass.
    model_kwargs = yaml.safe_load(MSWEA_CONFIG.read_text())["model"]["model_kwargs"]
    model_kwargs = {**model_kwargs, "presence_penalty": 0.0}
    async with aiohttp.ClientSession() as http:
        async with http.post(f"{service.url}/trajectories", json={"meta": {}}) as response:
            created = await response.json()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant that can interact with a computer."},
        {"role": "user", "content": "Please solve this issue: the build fails."},
    ]
    for turn in range(len(COMPLETIONS)):
        response = await litellm.acompletion(
            model="openai/qwen3.6-35b-a3b",
            messages=messages,
            tools=[BASH_TOOL],
            api_base=created["base_url"],
            api_key=created["api_key"],
            **model_kwargs,
        )
        message = response.choices[0].message
        assert message.reasoning_content, "the reasoning came back parsed, as vLLM's qwen3 reasoning parser does"
        # mini-swe-agent keeps `message.model_dump()` (minus its own `extra`) and sends it back as is.
        messages.append(message.model_dump())
        if turn < len(COMPLETIONS) - 1:
            (call,) = message.tool_calls
            assert call.function.name == "bash"
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": f'{{"returncode": 0, "output": "{turn}"}}'}
            )

    requests = engine.requests
    end = renderer.stop_token_ids()[0]
    sampled = [tokenizer.encode(text, add_special_tokens=False) + [end] for text in COMPLETIONS]
    for previous, current, completion in zip(requests, requests[1:], sampled):
        # Every prompt extends the previous prompt and its completion token for token: thinking included.
        exact = previous["token_ids"] + completion
        assert current["token_ids"][: len(exact)] == exact
    # And the template, rendering the replayed history whole (as LiteLLM sent it back), agrees.
    replayed = [{k: v for k, v in m.items() if k != "extra"} for m in messages[:-1]]
    assert renderer.render(replayed, [BASH_TOOL]).token_ids == requests[-1]["token_ids"]
    graph = service.server.trajectories[created["id"]].graph
    assert len(graph.paths()) == 1 and graph.branch_points() == []
    assert graph.unbridged_calls() == 0

    # What reaches vLLM: the trainer's sampling (temperature, top_p, top_k, min_p) whatever the agent asks; its
    # max_tokens (capped only by the room left in max_model_len) and penalties as sent; the template's stop tokens.
    params = requests[0]["sampling_params"]
    assert params == {
        **SAMPLING,
        "max_tokens": 8192,
        "presence_penalty": 0.0,
        "stop_token_ids": renderer.stop_token_ids(),
        "logprobs": 0,
    }
    # skycap rendered the prompt itself, so the agent's chat_template_kwargs never reach vLLM.
    assert "chat_template_kwargs" not in requests[0] and "messages" not in requests[0]


# -- evaluation only ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_evaluation_scores_each_trial_as_stock_harbor_and_writes_the_results(small, trials, tmp_path) -> None:
    import json

    from examples.train_integrations.harbor_skycap.entrypoints.main_harbor_skycap_eval import (
        HarborSkycapEvalExp,
    )

    exp = object.__new__(HarborSkycapEvalExp)
    exp.cfg = SimpleNamespace(
        generator=SimpleNamespace(eval_n_samples_per_prompt=1), trainer=SimpleNamespace(export_path=str(tmp_path))
    )
    # The fake trial's scripts, as task paths.
    exp.eval_dataset = [{"prompt": name} for name in ("linear", "timeout", "overflow")]
    gen = HarborSkycapGenerator(generator_cfg(), mini_swe_agent_cfg(), [small.url], SimpleNamespace(weight_version=7))
    await exp._evaluate(gen)

    rows = [json.loads(line) for line in (tmp_path / "eval" / "results.jsonl").read_text().splitlines()]
    by_task = {row["task"]: row for row in rows}
    # An agent timeout is masked for training; its score is the verifier's, here none (the fake runs none after it).
    timeout = by_task["timeout"]
    assert (timeout["stop_reason"], timeout["exception"], timeout["reward"], timeout["verified"]) == (
        "agent_timeout",
        "AgentTimeoutError",
        0.0,
        False,
    )
    assert by_task["overflow"]["reward"] == 1.0 and by_task["overflow"]["stop_reason"] == "context_length"
    assert by_task["linear"]["reward"] == 1.0 and by_task["linear"]["verified"] and by_task["linear"]["skycap_id"]
    summary = json.loads((tmp_path / "eval" / "summary.json").read_text())
    assert summary["num_trials"] == 3 and summary["mean_reward"] == pytest.approx(2 / 3)
    assert summary["exceptions"] == {"AgentTimeoutError": 1, "NonZeroAgentExitCodeError": 1}
