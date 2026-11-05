"""Terminus-2 for thinking models served as raw text: parse the reply after its thinking, replay it whole.

Served without a reasoning parser (skycap's ``use_raw_content``, or ``vllm serve`` with none), a thinking model's reply
is ``<reasoning></think><answer>`` (the chat template opens ``<think>`` in the prompt). Terminus-2 keeps that whole
reply in its history, so every earlier turn's thinking is replayed, as NVIDIA's Nemotron-3.5-Lightning recipe does
(NeMo Evaluator's ``reasoning_replay`` in ``think_tags`` mode). But its JSON parser takes the reply's first balanced
``{...}``, which may sit in the thinking. This agent parses only what follows the last ``</think>``.

    harbor_trial_config.agent.import_path=examples.train_integrations.harbor_skycap.terminus_think:TerminusThink

Importing this module also applies three of NeMo Evaluator's Terminus-2 patches (NVIDIA-NeMo/Evaluator 230c8411,
``solvers/harbor.py``), used for Nemotron-3.5-Lightning's Terminal-Bench 2.1 number. Without the first, a long-thinking
model whose history fills the context loops forever in summarization (the summary overflows too, every fallback fails,
and the next turn overflows again) until the agent timeout:

* context reset: when the full summary fails on the context length, the chat is cut back to its first message before
  the summary is sent, and the last fallback answers with a parseable "no commands, not complete" reply;
* unwinding drops at least 10 message pairs;
* token counts start from the API's reported usage, plus litellm's count of the messages since.
"""

import inspect
import json
import logging
import textwrap

from harbor.agents.terminus_2 import terminus_2 as _terminus2_module
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser
from harbor.agents.terminus_2.terminus_xml_plain_parser import TerminusXMLPlainParser

THINK_END = "</think>"


def after_thinking(response: str) -> str:
    """The reply after its last ``</think>``; the whole reply when it has none."""
    end = response.rfind(THINK_END)
    return response[end + len(THINK_END) :] if end >= 0 else response


class _AfterThinking:
    def parse_response(self, response: str):
        return super().parse_response(after_thinking(response))


class ThinkJSONParser(_AfterThinking, TerminusJSONPlainParser):
    pass


class ThinkXMLParser(_AfterThinking, TerminusXMLPlainParser):
    def salvage_truncated_response(self, truncated_response: str):
        return super().salvage_truncated_response(after_thinking(truncated_response))


class TerminusThink(Terminus2):
    @staticmethod
    def name() -> str:
        return "terminus-think"

    def _get_parser(self):
        if self._parser_name == "json":
            return ThinkJSONParser()
        if self._parser_name == "xml":
            return ThinkXMLParser()
        return super()._get_parser()


logger = logging.getLogger(__name__)


def _patch_source(function, replacements, name):
    """``function`` re-made from its source with each (anchor, replacement), every anchor exactly once."""
    src = inspect.getsource(inspect.unwrap(function))
    for anchor, replacement in replacements:
        if src.count(anchor) != 1:
            raise RuntimeError(f"can't patch Terminus2.{name}: an anchor matches {src.count(anchor)} times")
        src = src.replace(anchor, replacement, 1)
    namespace = {}
    exec(textwrap.dedent(src), _terminus2_module.__dict__, namespace)
    return namespace[name]


_CLE_RESET = [
    (
        "            summary_prompt = None\n            # Fallback 1: Try full summary\n",
        "            summary_prompt = None\n            full_summarize_failed_with_cle = False\n"
        "            # Fallback 1: Try full summary\n",
    ),
    (
        "            except Exception as e:\n"
        '                self.logger.debug(f"SUMMARIZATION: Full summary failed: {e}")\n',
        "            except Exception as e:\n"
        "                full_summarize_failed_with_cle = isinstance(e, ContextLengthExceededError)\n"
        '                self.logger.debug(f"SUMMARIZATION: Full summary failed: {e}")\n',
    ),
    (
        "            try:\n                start_time = time.time()\n                llm_response = await chat.chat(\n"
        "                    summary_prompt,\n",
        "            if full_summarize_failed_with_cle:\n"
        "                chat._messages = [chat.messages[0]]\n"
        "                chat.reset_response_chain()\n"
        "\n"
        "            try:\n                start_time = time.time()\n                llm_response = await chat.chat(\n"
        "                    summary_prompt,\n",
    ),
    (
        "                llm_response = LLMResponse(\n"
        '                    content="Technical difficulties. Please continue with the task."\n'
        "                )\n",
        "                llm_response = LLMResponse(content=self._build_local_fallback_llm_content())\n",
    ),
]

_UNWIND_MIN_PAIRS = [
    (
        "        context_limit = self._llm.get_model_context_limit()\n\n"
        "        while len(chat.messages) > 1:  # Keep at least the first message\n",
        "        context_limit = self._llm.get_model_context_limit()\n\n"
        "        min_pairs_to_remove = 10\n        pairs_removed = 0\n"
        "        while len(chat.messages) > 1:  # Keep at least the first message\n",
    ),
    (
        "            if free_tokens >= target_free_tokens:\n                break\n",
        "            if free_tokens >= target_free_tokens and pairs_removed >= min_pairs_to_remove:\n                break\n",
    ),
    (
        "            if len(chat.messages) >= 2:\n                chat._messages = chat.messages[:-2]\n"
        "            else:\n                break\n",
        "            if len(chat.messages) >= 2:\n                chat._messages = chat.messages[:-2]\n"
        "                pairs_removed += 1\n            else:\n                break\n",
    ),
]


def _build_local_fallback_llm_content(self) -> str:
    analysis = "Technical difficulties during context recovery. All summarization fallbacks failed."
    plan = "Technical difficulties. Please continue with the task."
    if self._parser_name == "xml":
        return (
            f"<response>\n<analysis>{analysis}</analysis>\n<plan>{plan}</plan>\n<commands>\n</commands>\n"
            "<task_complete>false</task_complete>\n</response>"
        )
    return json.dumps({"analysis": analysis, "plan": plan, "commands": [], "task_complete": False})


def _count_total_tokens(self, chat) -> int:
    from litellm.utils import token_counter

    anchor = getattr(chat, "_api_token_anchor", None)
    if anchor is not None and len(chat.messages) >= anchor[0]:
        anchor_len, anchor_total = anchor
        if len(chat.messages) == anchor_len:
            return anchor_total
        return anchor_total + token_counter(model=self._model_name, messages=chat.messages[anchor_len:])
    return token_counter(model=self._model_name, messages=chat.messages)


def _apply_nemo_evaluator_patches() -> None:
    from harbor.llms.chat import Chat

    if getattr(Terminus2, "_nemo_evaluator_patched", False):
        return
    Terminus2._query_llm = _patch_source(Terminus2._query_llm, _CLE_RESET, "_query_llm")
    Terminus2._build_local_fallback_llm_content = _build_local_fallback_llm_content
    Terminus2._unwind_messages_to_free_tokens = _patch_source(
        Terminus2._unwind_messages_to_free_tokens, _UNWIND_MIN_PAIRS, "_unwind_messages_to_free_tokens"
    )
    original_chat = Chat.chat

    async def chat_with_token_anchor(self, *args, **kwargs):
        response = await original_chat(self, *args, **kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None and getattr(usage, "prompt_tokens", None) is not None:
            self._api_token_anchor = (len(self._messages), usage.prompt_tokens + (usage.completion_tokens or 0))
        return response

    Chat.chat = chat_with_token_anchor
    Terminus2._count_total_tokens = _count_total_tokens
    Terminus2._nemo_evaluator_patched = True
    logger.info("Applied NeMo Evaluator's Terminus-2 patches: context reset, unwind >= 10 pairs, API token anchor")


_apply_nemo_evaluator_patches()
