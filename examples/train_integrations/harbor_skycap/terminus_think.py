"""Terminus-2 for thinking models served as raw text: parse the reply after its thinking, replay it whole.

Served without a reasoning parser (skycap's ``use_raw_content``, or ``vllm serve`` with none), a thinking model's reply
is ``<reasoning></think><answer>`` (the chat template opens ``<think>`` in the prompt). Terminus-2 keeps that whole
reply in its history, so every earlier turn's thinking is replayed, as NVIDIA's Nemotron-3.5-Lightning recipe does
(NeMo Evaluator's ``reasoning_replay`` in ``think_tags`` mode). But its JSON parser takes the reply's first balanced
``{...}``, which may sit in the thinking. This agent parses only what follows the last ``</think>``.

    harbor_trial_config.agent.import_path=examples.train_integrations.harbor_skycap.terminus_think:TerminusThink
"""

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
