from typing import List, Tuple, Optional
import logging
import torch
from transformers import AutoTokenizer
from jaxtyping import Float

logger = logging.getLogger(__name__)


def _verify_inputs(
    prompts: List[List[int]],
    responses: List[List[int]],
    rewards: Optional[List[torch.Tensor]],
    loss_masks: List[List[int]],
):
    assert (
        len(prompts) == len(responses) and len(prompts) > 0
    ), "prompts and responses must have the same length and length must be greater than 0, got {} and {}".format(
        len(prompts), len(responses)
    )

    if rewards is not None:
        assert len(rewards) == len(prompts), "rewards must have the same length as prompts, got {} and {}".format(
            len(rewards), len(prompts)
        )
    assert len(loss_masks) == len(prompts), "loss_masks must have the same length as prompt, got {} and {}".format(
        len(loss_masks), len(prompts)
    )


def convert_prompts_responses_to_batch_tensors(
    tokenizer: AutoTokenizer,
    prompts: List[List[int]],
    responses: List[List[int]],
    rewards: List[List[float]],
    loss_masks: List[List[int]],
    logprobs: Optional[List[List[float]]] = None,
    max_seq_len: Optional[int] = None,
) -> Tuple[
    Float[torch.Tensor, "batch seq_len"],
    Float[torch.Tensor, "batch seq_len"],
    Float[torch.Tensor, "batch response_len"],
    Float[torch.Tensor, "batch response_len"],
    Float[torch.Tensor, "batch response_len"],
    Optional[Float[torch.Tensor, "batch response_len"]],
]:
    """Port of PR #1285: unified left-pad layout, right-aligned response tensors.

    Each row is a single left-padded block::

        | [PAD] [PAD] prompt prompt prompt respon respon |
        | [PAD] prompt prompt prompt respon respon respon |
        | prompt prompt prompt respon respon respon respon |
                                |<---- max_response_len ---->|

    The padded sequence length is ``max(prompt_len_i + response_len_i)`` rather than
    the old ``max_input_len + max_output_len``; in step-wise training prompts and
    responses are anti-correlated across turns (turn 1 has a short prompt + long
    response, turn N has a long prompt + short response), so the old formula
    inflated sequences to nearly ``2 * max_seq_len``.

    Response-level tensors (``action_mask``, ``rewards``, ``loss_masks``,
    ``logprobs``) are **right-aligned** within ``(batch, max_response_len)`` so
    they match the model's ``log_probs[:, -num_actions-1:-1]`` slicing where the
    response tokens naturally land at the end of the left-padded sequence.

    Assumes that the responses already contain an eos token at index -1.

    Args:
        tokenizer: Model tokenizer.
        prompts: Tokenized prompts, one per row.
        responses: Tokenized responses, one per row.
        rewards: Per-row rewards (scalar) or per-token rewards (list).
        loss_masks: Per-row loss masks over response tokens.
        logprobs: Per-row rollout logprobs over response tokens.
        max_seq_len: If provided and ``max(prompt_i + response_i)`` exceeds it, a
            warning is logged (no truncation; generator should have respected it).

    Returns:
        sequences: ``(batch, max_total)`` left-padded concatenation.
        attention_mask: ``(batch, max_total)``.
        action_mask: ``(batch, max_response_len)`` — right-aligned.
        rewards: ``(batch, max_response_len)`` — right-aligned.
        loss_masks: ``(batch, max_response_len)`` — right-aligned.
        logprobs: ``(batch, max_response_len)`` — right-aligned, or None.
    """
    _verify_inputs(prompts, responses, rewards, loss_masks)

    prompt_token_lens = [len(p) for p in prompts]
    response_token_lens = [len(r) for r in responses]

    max_response = max(response_token_lens)
    # Pad to the tightest bound: max per-sample total.
    max_total = max(p + r for p, r in zip(prompt_token_lens, response_token_lens))

    if max_seq_len is not None and max_total > max_seq_len:
        logger.warning(
            f"Max sequence length in batch ({max_total}) exceeds max_seq_len ({max_seq_len}). "
            f"No truncation is performed; consider checking generator settings."
        )

    pad_token_id = tokenizer.pad_token_id
    sequences = []
    attention_masks = []
    action_masks = []
    for i in range(len(prompts)):
        total_real = prompt_token_lens[i] + response_token_lens[i]
        pad_len = max_total - total_real

        # Unified left-pad: [PAD ... PAD  PROMPT  RESPONSE]
        seq = [pad_token_id] * pad_len + list(prompts[i]) + list(responses[i])
        attention_mask_i = [0] * pad_len + [1] * total_real

        # Response indicator within the last max_response positions (right-aligned).
        resp_pad = max_response - response_token_lens[i]
        action_mask_i = [0] * resp_pad + [1] * response_token_lens[i]

        sequences.append(seq)
        attention_masks.append(attention_mask_i)
        action_masks.append(action_mask_i)

    sequences = torch.tensor(sequences)
    attention_mask = torch.tensor(attention_masks, dtype=torch.int64)
    action_mask = torch.tensor(action_masks, dtype=torch.int64)

    # Response-level tensors are RIGHT-ALIGNED to match the model output.
    # The model's log_probs[:, -num_actions-1:-1] returns logprobs where
    # response tokens occupy the last response_len_i positions.
    ret_loss_masks = torch.zeros(len(prompts), max_response, dtype=torch.float)
    for i, lm in enumerate(loss_masks):
        if len(lm) == 0:
            continue
        ret_loss_masks[i, max_response - len(lm):] = torch.tensor(lm, dtype=torch.float)

    ret_rewards = torch.zeros(len(prompts), max_response, dtype=torch.float)
    for i, custom_reward in enumerate(rewards):
        if isinstance(custom_reward, list):
            custom_reward = torch.tensor(custom_reward, dtype=torch.float)
        if custom_reward.numel() == 0:
            continue
        ret_rewards[i, max_response - custom_reward.numel():] = custom_reward

    logprobs_tensor = None
    if logprobs:
        logprobs_tensor = torch.zeros(len(prompts), max_response, dtype=torch.float)
        for i, sample_logprobs in enumerate(logprobs):
            if len(sample_logprobs) == 0:
                continue
            lp = torch.tensor(sample_logprobs, dtype=torch.float)
            logprobs_tensor[i, max_response - len(sample_logprobs):] = lp

    return sequences, attention_mask, action_mask, ret_rewards, ret_loss_masks, logprobs_tensor
