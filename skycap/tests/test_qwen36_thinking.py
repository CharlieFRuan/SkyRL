"""Qwen3.6 with thinking kept in history, on the real tokenizer and chat template.

A tool-calling harness (mini-swe-agent) replays each reply as parsed ``content``,
``reasoning_content`` and ``tool_calls``. With ``preserve_thinking`` and
``thinking_retention="all"``, every turn must then extend the previous prompt and
completion token for token, and a full render of the replayed history must give
the same tokens, so the graph stays one path instead of forking every turn.

Needs the ``tokens`` extra and the Qwen3.6 tokenizer: a local checkout
(``SKYCAP_QWEN36_TOKENIZER``, or the path below) or the Hub's; skipped otherwise.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

pytest.importorskip("renderers")

from skycap.tokens.renderer import RenderersRenderer  # noqa: E402
from tests.test_tokens import client, token_stack, user  # noqa: E402

LOCAL = "/mnt/local_storage/charlieruan/hf/Qwen3.6-35B-A3B"
TOKENIZER = os.environ.get("SKYCAP_QWEN36_TOKENIZER") or (LOCAL if os.path.isdir(LOCAL) else "Qwen/Qwen3.6-35B-A3B")
TEMPLATE = {"enable_thinking": True, "preserve_thinking": True}

#: mini-swe-agent's tool (minisweagent.models.utils.actions_toolcall.BASH_TOOL).
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


def bash(command: str) -> str:
    return f"<tool_call>\n<function=bash>\n<parameter=command>\n{command}\n</parameter>\n</function>\n</tool_call>"


#: What the model samples after the generation prompt (``<|im_start|>assistant\n<think>\n``), turn by turn:
#: reasoning, then text and a call; a call with no text; and a call with a multi-line command.
COMPLETIONS = [
    "The repo layout first.\n</think>\n\nLet me look around.\n\n" + bash("ls -la"),
    "There's a setup.py.\nI'll read it.\n</think>\n\n" + bash("cat setup.py"),
    "Fix it.\n</think>\n\nPatching.\n\n" + bash("cat <<'EOF' > fix.py\nprint(1)\nEOF\npython fix.py"),
]
#: And after the last tool result, an answer with no call.
FINAL = "Done.\n</think>\n\nAll fixed."


@pytest.fixture(scope="module")
def renderer() -> RenderersRenderer:
    try:
        return RenderersRenderer(
            TOKENIZER, size=1, thinking_retention="all", chat_template_kwargs=TEMPLATE, renderer="qwen3.6"
        )
    except Exception as error:  # noqa: BLE001 - no network and no cache
        pytest.skip(f"tokenizer unavailable: {error}")


@pytest.fixture(scope="module")
def tokenizer(renderer: RenderersRenderer) -> Any:
    from renderers.base import load_tokenizer

    return load_tokenizer(TOKENIZER)


def test_a_local_checkout_needs_the_renderer_named(renderer: RenderersRenderer) -> None:
    # Picked by Hub name otherwise: a local path matches none, and the template's kwargs can't be applied.
    if TOKENIZER.startswith("/"):
        with pytest.raises(ValueError, match="chat_template_kwargs"):
            RenderersRenderer(TOKENIZER, size=1, chat_template_kwargs=TEMPLATE)
    # Keeping thinking in the renderer and dropping it in the template contradict each other.
    with pytest.raises(ValueError, match="thinking_retention"):
        RenderersRenderer(TOKENIZER, size=1, chat_template_kwargs={"preserve_thinking": False}, renderer="qwen3.6")


async def test_replayed_thinking_and_tool_calls_extend_every_turn(renderer: RenderersRenderer, tokenizer) -> None:
    end = renderer.stop_token_ids()[0]
    completions = [tokenizer.encode(text, add_special_tokens=False) + [end] for text in [*COMPLETIONS, FINAL]]
    calls = iter(completions)

    async with token_stack(completion=lambda prompt, sampling: next(calls)) as stack:
        stack.server.backend.renderer = renderer  # type: ignore[attr-defined]
        created = await stack.create()
        llm = client(created["base_url"])
        messages: list[dict[str, Any]] = [{"role": "system", "content": "You are a helpful assistant."}, user("fix it")]
        for turn in range(len(COMPLETIONS)):
            reply = (
                (await llm.chat.completions.create(model="policy", messages=messages, tools=[BASH_TOOL]))
                .choices[0]
                .message
            )
            assert reply.reasoning_content and reply.tool_calls and reply.tool_calls[0].function.name == "bash"
            # As LiteLLM's message model_dump() comes back from mini-swe-agent: every field, None ones too.
            messages.append(reply.model_dump())
            messages.append({"role": "tool", "tool_call_id": reply.tool_calls[0].id, "content": f"output {turn}"})
        await llm.chat.completions.create(model="policy", messages=messages, tools=[BASH_TOOL])

        requests = stack.engine.requests
        assert len(requests) == len(COMPLETIONS) + 1
        for previous, current, sampled in zip(requests, requests[1:], completions):
            # Each prompt is the previous prompt and the sampled completion, then the new messages.
            exact = previous["token_ids"] + sampled
            assert current["token_ids"][: len(exact)] == exact
        # Not only by bridging: the template, rendering the replayed history whole, gives the same tokens.
        full = renderer.render(messages, [BASH_TOOL])
        assert full.token_ids == requests[-1]["token_ids"]
        text = tokenizer.decode(requests[-1]["token_ids"])
        assert text.count("<think>") == len(COMPLETIONS) + 1 and "The repo layout first." in text

        graph = stack.server.trajectories[created["id"]].graph
        assert len(graph.paths()) == 1 and graph.branch_points() == []
        assert (await stack.finish(created["id"]))["unbridged_calls"] == 0


def test_a_call_without_its_tool_call_tag_is_still_parsed(renderer: RenderersRenderer, tokenizer: Any) -> None:
    # Qwen3.6 sometimes drops the opening `<tool_call>` (seen on SWE-bench Verified: ~15% of mini-swe-agent rollouts
    # hit it at least once). vLLM's qwen3_coder parser still extracts the call; so must skycap, or the agent gets a
    # "no tool calls" error a stock endpoint never gives.
    prompt = tokenizer.encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n", add_special_tokens=False)
    text = (
        "Look first.\n</think>\n\nLet me look.\n"
        '<function=bash>\n<parameter=command>\nfind . -name "*.py"\n</parameter>\n</function>\n</tool_call><|im_end|>'
    )
    message = renderer.parse(tokenizer.encode(text, add_special_tokens=False), [BASH_TOOL], prompt_ids=prompt)
    (call,) = message["tool_calls"]
    assert call["function"]["name"] == "bash"
    assert json.loads(call["function"]["arguments"]) == {"command": 'find . -name "*.py"'}
    assert message["content"] == "Let me look."
    assert message["reasoning_content"] == "Look first."
