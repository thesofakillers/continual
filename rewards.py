import numpy as np

import torch
from custom_trainer import MyCustomTrainer
from trl.trainer.utils import pad


def reward_len(completions, trainer_instance: MyCustomTrainer, **kwargs):
    """
    A simple reward function that rewards completions based on their length.
    It rewards completions that are close to 50 characters.
    """
    # The 'completions' argument is a list of strings
    rewards = [-abs(50 - len(completion)) for completion in completions]
    register_cached_output(completions, rewards, trainer_instance)
    return rewards


def reward_cumulative_logprob(
    completions,
    completion_ids,
    trainer_instance: MyCustomTrainer,
    prompt_completion_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    **kwargs,
):
    """
    Reward function that minimizes cumulative log probability of the completion,
    normalized by the vocabulary size.
    """
    rewards = []
    vocab_size = trainer_instance.processing_class.vocab_size
    for i, completion in enumerate(completion_ids):
        completion_len = len(completion)
        cumulative_logprob = trainer_instance.custom_cached_logps[i, :completion_len].sum()
        normalized_cumulative_logprob = cumulative_logprob / torch.log2(
            torch.tensor(vocab_size, dtype=torch.float)
        )
        rewards.append(-normalized_cumulative_logprob.item())
    register_cached_output(completions, rewards, trainer_instance)
    return rewards


def reward_surprisal_moments(
    completions,
    completion_ids,
    trainer_instance: MyCustomTrainer,
    prompt_completion_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    **kwargs,
):
    """
    Reward function based on centered moments of surprisal.
    Calculates a reward based on a list of moments specified in `args.surprisal_reward_moments`.
    - k=1: entropy (incentivized to be low)
    - k=2: varentropy (incentivized to be high)
    - k>2: k-th centered moment of surprisal, with alternating sign incentives.
    Each moment's contribution is normalized to have the same units.
    """
    model = trainer_instance.model
    moments = trainer_instance.args.surprisal_reward_moments
    if not moments:
        return [0.0] * len(completion_ids)

    logits_to_keep = max(len(c) for c in completion_ids)

    with torch.no_grad():
        # TODO: use cached log probs instead, cache the logits if necessary
        logits = model(
            prompt_completion_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).logits

        completion_logits = logits[:, -logits_to_keep - 1 : -1, :]
        log_probs_dist = torch.nn.functional.log_softmax(completion_logits, dim=-1)
        probs = torch.exp(log_probs_dist)
        # surprisal for each vocabulary item
        surprisal_dist = -log_probs_dist  # shape: (batch_size, seq_len, vocab_size)

        token_rewards = torch.zeros(
            completion_logits.shape[:2], device=logits.device
        )  # shape: (batch_size, seq_len)

        # E[s] = entropy
        entropy = torch.sum(
            probs * surprisal_dist, dim=-1
        )  # shape: (batch_size, seq_len)

        if 1 in moments:
            token_rewards += -entropy  # sign is (-1)^1

        centered_surprisal = surprisal_dist - entropy.unsqueeze(
            -1
        )  # shape: (batch_size, seq_len, vocab_size)

        for k in moments:
            if k <= 1:
                continue

            # m_k = E[(s - E[s])^k]
            m_k = torch.sum(
                probs * (centered_surprisal**k), dim=-1
            )  # shape: (batch_size, seq_len)

            # root_k(m_k) = sign(m_k) * |m_k|^(1/k)
            root_k_mk = torch.sign(m_k) * torch.pow(torch.abs(m_k) + 1e-9, 1.0 / k)

            token_rewards += ((-1) ** k) * root_k_mk

    rewards = []
    for i, completion in enumerate(completion_ids):
        completion_len = len(completion)
        total_reward = token_rewards[i, :completion_len].sum()
        rewards.append(total_reward.item())
    register_cached_output(completions, rewards, trainer_instance)
    return rewards


def register_cached_output(completions, rewards, trainer_instance):
    index = np.argmax(rewards)
    print("COMPLETIONS", completions)
    trainer_instance.custom_cached_output = completions[index]
