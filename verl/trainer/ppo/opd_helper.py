# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Helper functions for On-Policy Distillation (OPD) with selective teacher guidance.

OPD applies knowledge distillation from a teacher model only on:
1. Prompts with low pass rates (per-prompt gating)
2. Failed rollouts from those prompts (per-sample gating within group)
3. Within a limited token horizon (to avoid importing verbosity)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from verl import DataProto


def compute_prompt_pass_rates(
    batch: DataProto,
    n_samples_per_prompt: int,
    reward_key: str = "token_level_scores",
) -> torch.Tensor:
    """
    Compute pass rate for each unique prompt across its rollouts.

    For verifiable tasks with binary rewards, pass rate = mean(R_i) over all rollouts for prompt x.

    Args:
        batch: DataProto containing rollout data
        n_samples_per_prompt: Number of rollout samples per prompt (e.g., 8)
        reward_key: Key in batch.batch containing rewards (default: "token_level_scores")
                   Expected to be token-level rewards with shape [batch_size, seq_len]

    Returns:
        pass_rates: Tensor of shape [num_prompts] with pass rate per prompt
                   pass_rate[i] = fraction of rollouts for prompt i that succeeded

    Example:
        If batch_size=16, n=4, we have 4 prompts with 4 rollouts each:
        - Indices [0,1,2,3] are rollouts for prompt 0
        - Indices [4,5,6,7] are rollouts for prompt 1
        - etc.
    """
    # Get rewards (token-level)
    token_level_rewards = batch.batch[reward_key]  # [batch_size, seq_len]

    # For verifiable tasks, reward is typically sparse (only on final answer token)
    # Sum across sequence to get per-rollout reward (0 or 1 for correctness)
    response_mask = batch.batch.get("response_mask", None)
    if response_mask is not None:
        # Only sum over response tokens
        rollout_rewards = (token_level_rewards * response_mask).sum(dim=-1)  # [batch_size]
    else:
        rollout_rewards = token_level_rewards.sum(dim=-1)  # [batch_size]

    # Binarize: any positive reward = success
    rollout_success = (rollout_rewards > 0).float()  # [batch_size]

    # Reshape to [num_prompts, n_samples_per_prompt]
    batch_size = rollout_success.shape[0]
    num_prompts = batch_size // n_samples_per_prompt

    assert batch_size % n_samples_per_prompt == 0, (
        f"Batch size {batch_size} must be divisible by n_samples_per_prompt {n_samples_per_prompt}"
    )

    rollout_success_grouped = rollout_success.view(num_prompts, n_samples_per_prompt)  # [num_prompts, n]

    # Compute pass rate per prompt (mean success rate across rollouts)
    pass_rates = rollout_success_grouped.mean(dim=1)  # [num_prompts]

    return pass_rates


def create_opd_eligibility_mask(
    batch: DataProto,
    pass_rates: torch.Tensor,
    threshold: float = 0.3,
    n_samples_per_prompt: int = 8,
    reward_key: str = "token_level_scores",
) -> torch.Tensor:
    """
    Create per-sample binary mask indicating which rollouts should receive OPD.

    OPD eligibility criteria (both must be satisfied):
    1. Prompt-level: pass_rate(x) < threshold (prompt is underperforming)
    2. Sample-level: R_i = 0 (this specific rollout failed)

    Args:
        batch: DataProto containing rollout data
        pass_rates: Per-prompt pass rates from compute_prompt_pass_rates() [num_prompts]
        threshold: Pass rate threshold (e.g., 0.3 = apply OPD when ≥70% fail)
        n_samples_per_prompt: Number of rollout samples per prompt
        reward_key: Key in batch.batch containing rewards

    Returns:
        opd_mask: Binary mask of shape [batch_size] where 1 = eligible for OPD

    Example:
        threshold = 0.3, n = 4
        prompt 0: pass_rate = 0.25 (< 0.3) → underperforming
            rollout 0: reward = 0 → eligible (mask = 1)
            rollout 1: reward = 1 → not eligible (mask = 0, correct solution)
            rollout 2: reward = 0 → eligible (mask = 1)
            rollout 3: reward = 0 → eligible (mask = 1)
        prompt 1: pass_rate = 0.75 (> 0.3) → performing well
            rollout 4-7: all get mask = 0 (regardless of individual success)
    """
    # Get per-rollout success (same logic as compute_prompt_pass_rates)
    token_level_rewards = batch.batch[reward_key]
    response_mask = batch.batch.get("response_mask", None)

    if response_mask is not None:
        rollout_rewards = (token_level_rewards * response_mask).sum(dim=-1)
    else:
        rollout_rewards = token_level_rewards.sum(dim=-1)

    rollout_failed = (rollout_rewards <= 0).float()  # [batch_size], 1 = failed

    # Determine which prompts are underperforming
    prompt_underperforming = (pass_rates < threshold).float()  # [num_prompts], 1 = underperforming

    # Expand to per-rollout: repeat each prompt decision n_samples_per_prompt times
    num_prompts = pass_rates.shape[0]
    batch_size = num_prompts * n_samples_per_prompt

    # Repeat interleaved: [p0, p0, p0, p1, p1, p1, ...]
    prompt_underperforming_expanded = prompt_underperforming.repeat_interleave(n_samples_per_prompt)  # [batch_size]

    # OPD mask: underperforming prompt AND failed rollout
    opd_mask = prompt_underperforming_expanded * rollout_failed  # [batch_size]

    return opd_mask


def create_horizon_mask(
    batch: DataProto,
    horizon: int = 512,
    stop_before_answer_tokens: bool = True,
    answer_token_ids: Optional[list[int]] = None,
) -> torch.Tensor:
    """
    Create token-level mask for KD horizon.

    KD is only applied to thinking tokens (before answer). The horizon is determined by:
    - If stop_before_answer_tokens=True and answer detected: H_i = min(horizon, t_ans)
    - Otherwise: H_i = horizon

    Args:
        batch: DataProto containing rollout data
        horizon: Global maximum horizon (e.g., 512 tokens)
        stop_before_answer_tokens: Whether to stop KD before answer tokens (default: True)
        answer_token_ids: List of token IDs that indicate start of final answer (e.g., <answer>)
                         If None, no answer token detection is performed

    Returns:
        horizon_mask: Binary mask of shape [batch_size, seq_len] where 1 = within horizon

    Example:
        horizon = 512, stop_before_answer_tokens = True
        Sequence 1: 600 tokens, no answer token → mask first 512 tokens
        Sequence 2: 300 tokens, answer at t=200 → mask first min(512, 200) = 200 tokens (stop before <answer>)
        Sequence 3: 800 tokens, answer at t=100 → mask first min(512, 100) = 100 tokens
    """
    response_mask = batch.batch["response_mask"]  # [batch_size, seq_len]
    batch_size, seq_len = response_mask.shape
    device = response_mask.device

    # Initialize horizon mask with response_mask (only mask response tokens)
    horizon_mask = response_mask.clone().float()  # [batch_size, seq_len]

    # Create position indices [0, 1, 2, ..., seq_len-1]
    position_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len]

    # Detect answer token positions (if answer_token_ids provided and stop requested)
    if stop_before_answer_tokens and answer_token_ids is not None and len(answer_token_ids) > 0:
        input_ids = batch.batch["input_ids"]  # [batch_size, seq_len]

        # Find first occurrence of any answer token in each sequence
        answer_positions = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)  # [batch_size]

        for token_id in answer_token_ids:
            # Find positions where token appears
            token_matches = (input_ids == token_id).long()  # [batch_size, seq_len]
            # Get first occurrence per sequence (use argmax trick: first 1 in binary array)
            first_occurrences = token_matches.argmax(dim=1)  # [batch_size]
            # Only update if token was actually found (check if position has the token)
            found_mask = token_matches.gather(1, first_occurrences.unsqueeze(1)).squeeze(1).bool()  # [batch_size]
            # Update answer_positions with minimum between current and new finding
            answer_positions = torch.where(
                found_mask,
                torch.minimum(answer_positions, first_occurrences),
                answer_positions
            )

        # Compute effective horizon per sequence: min(horizon, answer_pos)
        # Stop KD before answer token (thinking tokens only)
        effective_horizons = torch.minimum(
            torch.tensor(horizon, device=device),
            answer_positions
        )  # [batch_size]
    else:
        # No answer detection: use fixed horizon for all
        effective_horizons = torch.full((batch_size,), horizon, dtype=torch.long, device=device)  # [batch_size]

    # Apply horizon mask: zero out positions beyond effective horizon
    # position_indices < effective_horizons.unsqueeze(1) → [batch_size, seq_len]
    within_horizon = (position_indices < effective_horizons.unsqueeze(1)).float()  # [batch_size, seq_len]

    horizon_mask = horizon_mask * within_horizon  # Element-wise multiply

    return horizon_mask


def compute_opd_metrics(
    pass_rates: torch.Tensor,
    opd_mask: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    """
    Compute essential diagnostics metrics for OPD application.

    Args:
        pass_rates: Per-prompt pass rates [num_prompts]
        opd_mask: Per-sample OPD eligibility mask [batch_size]
        threshold: Pass rate threshold used for gating

    Returns:
        metrics: Dictionary with essential OPD diagnostic metrics
    """
    num_prompts = pass_rates.shape[0]
    batch_size = opd_mask.shape[0]

    # Key metric 1: What fraction of prompts get OPD?
    # (Expect ~10-30% early, should decrease as model improves)
    num_underperforming = (pass_rates < threshold).sum().item()
    frac_underperforming_prompts = num_underperforming / num_prompts

    # Key metric 2: What fraction of rollouts get teacher guidance?
    # (< frac_failures due to prompt gating)
    frac_opd_samples = opd_mask.sum().item() / batch_size

    metrics = {
        "opd/frac_underperforming_prompts": frac_underperforming_prompts,
        "opd/frac_opd_samples": frac_opd_samples,
    }

    return metrics


def should_apply_opd(
    global_step: int,
    warmup_steps: int,
    enable_opd: bool = True,
) -> bool:
    """
    Determine whether OPD should be applied at current training step.

    Two-phase schedule:
    - Phase 1 (steps 0 to warmup_steps): Pure RL, no OPD
    - Phase 2 (steps > warmup_steps): RL + OPD on underperforming prompts

    Args:
        global_step: Current training step
        warmup_steps: Number of warmup steps before enabling OPD (e.g., 100-200)
        enable_opd: Global flag to enable/disable OPD

    Returns:
        should_apply: True if OPD should be applied at this step
    """
    if not enable_opd:
        return False

    return global_step >= warmup_steps
