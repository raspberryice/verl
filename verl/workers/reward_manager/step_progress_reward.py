"""
Step Progress Reward Manager

Reward manager that computes process rewards for overthinking mitigation.
Combines outcome-based rewards (R_final) with step-level progress rewards (U_i).

Based on design document: design/process_reward.md
"""

from typing import Any, Callable, Optional

import torch
from transformers import PreTrainedTokenizer

from verl.protocol import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

# Import our process reward components
import sys
from pathlib import Path

# Add reward_functions to path if not already there
reward_functions_path = Path(__file__).parent.parent.parent.parent.parent / "reward_functions"
if str(reward_functions_path) not in sys.path:
    sys.path.insert(0, str(reward_functions_path))

from episode_segmenter import EpisodeSegmenter
from prefix_value_estimator import PrefixValueEstimator
from step_progress_reward import (
    compute_step_progress_reward_phase1,
    compute_step_progress_reward_phase2,
    compute_step_progress_reward_anchor,
    AnchorLengthTracker,
)
from length_baseline import LengthBaselineTracker


@register("step_progress_reward")
class StepProgressRewardManager(AbstractRewardManager):
    """
    Reward manager for step progress rewards (overthinking mitigation).

    Computes rewards as:
        Phase 1: R = R_final + λ * Σ_i clip(U_i)
        Phase 2: R = R_final + λ * Σ_i U_i - Σ_i c_i (quantile-based length baseline)
        Anchor:  R = R_final + λ * Σ_i U_i - Σ_i c_i (pass-rate anchor baseline)

    where:
        U_i = V(prefix_i) - V(prefix_{i-1})  (marginal utility)
        V(prefix) = P(correct | stop at prefix and answer)

    Length baseline approaches:
        - Phase 2: Uses 60% quantile of correct response lengths (LengthBaselineTracker)
        - Anchor: Uses average length when pass rate >= 80%, continuously updated
                  while pass rate stays above threshold (AnchorLengthTracker)

    This encourages the model to make consistent progress and avoid
    redundant verification loops after already solving the problem.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Callable,  # Base reward function (e.g., bigmath_reward)
        reward_fn_key: str = "data_source",
        step_progress_reward_config: Optional[dict] = None,
    ):
        """
        Initialize StepProgressRewardManager.

        Args:
            tokenizer: Tokenizer for text processing
            compute_score: Base reward function for outcome correctness
            reward_fn_key: Key to use for reward function selection (default: "data_source")
                             Required for process rewards, falls back to base rewards if None
            step_progress_reward_config: Configuration dict with keys:
                - phase: 1, 2, or "anchor" (default: 1)
                - lambda: Weight for process rewards (default: 0.1)
                - clip_min, clip_max: Clipping bounds for utilities (default: -0.5, 0.5)
                - discourse_markers: List of markers for episode segmentation
                - beta_length: Weight for length penalty (Phase 2/Anchor, default: 0.01)
                - waste_threshold: Threshold for wasteful episodes (Phase 2/Anchor, default: 0.0)
                - use_solve_gating: Whether to only penalize post-solve episodes (default: False)

                Phase 2 specific (quantile-based baseline):
                - length_quantile: Quantile for baseline (default: 0.6)
                - length_ema_alpha: EMA smoothing factor (default: 0.1)

                Anchor specific (pass-rate-based baseline):
                - anchor_pass_threshold: Pass rate to set/update anchor (default: 0.8)
                - anchor_window_size: Sliding window size (default: 32)
                - anchor_min_samples: Min samples before setting anchor (default: 16)
        """
        self.tokenizer = tokenizer
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key

        # Parse configuration
        config = step_progress_reward_config or {}
        self.phase = config.get("phase", 1)  # 1, 2, or "anchor"
        self.lambda_process = config.get("lambda", 0.1)

        # Common parameters
        self.clip_min = config.get("clip_min", -0.5)
        self.clip_max = config.get("clip_max", 0.5)

        # Length penalty parameters (Phase 2 and Anchor)
        self.beta_length = config.get("beta_length", 0.01)
        self.waste_threshold = config.get("waste_threshold", 0.0)
        self.use_solve_gating = config.get("use_solve_gating", False)
        self.solve_threshold = config.get("solve_threshold", 0.8)

        # Initialize episode segmenter
        self.segmenter = EpisodeSegmenter(
            tokenizer=tokenizer,
            discourse_markers=config.get("discourse_markers", None),  # Uses defaults if None
            min_episode_tokens=config.get("min_episode_tokens", 32),
        )

        # Initialize prefix value estimator
        self.value_estimator = PrefixValueEstimator(
            tokenizer=tokenizer,
            force_answer_prompt=config.get(
                "force_answer_prompt",
                "I need to stop thinking. I think the final answer is \\boxed{"
            ),
            ground_truth_max_tokens=config.get("ground_truth_max_tokens", 32),
            value_computation=config.get("value_computation", "geometric_mean"),
        )

        # Initialize length baseline tracker (Phase 2 or Anchor)
        if self.phase == 2:
            self.length_tracker = LengthBaselineTracker(
                quantile=config.get("length_quantile", 0.6),
                ema_alpha=config.get("length_ema_alpha", 0.1),
            )
        elif self.phase == "anchor":
            self.anchor_tracker = AnchorLengthTracker(
                pass_threshold=config.get("anchor_pass_threshold", 0.8),
                window_size=config.get("anchor_window_size", 32),
                min_samples=config.get("anchor_min_samples", 16),
                min_anchor=config.get("anchor_min_length", 256.0),
                lenient_fallback=config.get("anchor_lenient_fallback", 16384.0),
            )

        # Enable prefix value pre-computation (REQUIRED, not optional)
        self.enable_prefix_value_cache = config.get("enable_prefix_value_cache", True)


    def _compute_base_rewards(self, data: DataProto) -> torch.Tensor:
        """
        Compute base correctness rewards (R_final).

        This reuses the logic from PrimeRewardManager to compute
        binary correctness rewards using the compute_score function.

        Args:
            data: DataProto containing batch data

        Returns:
            reward_tensor: [batch_size, seq_len] with R_final at last response token
        """
        # Initialize reward tensor (all zeros)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        # Get prompt and response info
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)

        # Decode responses
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        # Get ground truths
        ground_truths = [
            item["ground_truth"]
            for item in data.non_tensor_batch["reward_model"]
        ]

        # Get data sources
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        # Compute scores using base reward function
        scores = []
        for completion, reference, task in zip(sequences_str, ground_truths, data_sources):
            try:
                score = self.compute_score(task, completion, reference, extra_info=None)
                if isinstance(score, (list, tuple)):
                    scores.append(float(score[0]))
                else:
                    scores.append(float(score))
            except Exception as e:
                print(f"Error computing score: {e}")
                scores.append(0.0)

        # Assign rewards to last token of each response
        for i in range(len(data)):
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

        return reward_tensor

    def _get_precomputed_prefix_values(self, data: DataProto) -> tuple:
        """
        Retrieve pre-computed prefix values from batch metadata.

        This is REQUIRED - training will fail if values are missing. There is no
        fallback to recomputing on the critical path.

        Expected structure in data.meta_info["prefix_value_cache"]:
        {
            "prefix_values": torch.Tensor [batch_size, max_episodes],
            "episode_boundaries": torch.Tensor [batch_size, max_episodes, 2],
            "cache_timestamp": int (for debugging)
        }

        Returns:
            (prefix_values, episode_boundaries)

        Raises:
            RuntimeError: If pre-computed values are missing or invalid
        """
        if not self.enable_prefix_value_cache:
            raise RuntimeError(
                "[StepProgressReward] Prefix value pre-computation disabled, but reward computation requires it. "
                "Set enable_prefix_value_cache=True in config."
            )

        cache = data.meta_info.get("prefix_value_cache", None)
        if cache is None:
            raise RuntimeError(
                "[StepProgressReward] Missing prefix_value_cache in batch meta_info. "
                "PrefixLogProbPopulator.populate_cache() must be called during generation phase."
            )

        # Validate values match current batch
        cached_batch_size = cache["prefix_values"].shape[0]
        current_batch_size = len(data)

        if cached_batch_size != current_batch_size:
            raise RuntimeError(
                "[StepProgressReward] Pre-computed prefix_value tensor does not match batch size "
                f"({cached_batch_size} vs {current_batch_size}). Generation pipeline error."
            )

        # Values are valid and ready
        print(f"[StepProgressReward] Using pre-computed prefix values (shape: {cache['prefix_values'].shape})")
        return cache["prefix_values"], cache["episode_boundaries"]

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """
        Compute step progress rewards for a batch.

        Requires prefix values to be pre-computed during generation phase.
        Will raise RuntimeError if values are missing.

        Args:
            data: DataProto containing batch data
            return_dict: If True, return dict with "reward_tensor" and "reward_extra_info"

        Returns:
            reward_tensor: [batch_size, seq_len] token-level rewards
            OR dict with {"reward_tensor": ..., "reward_extra_info": {...}}
        """
        base_reward_tensor = self._compute_base_rewards(data)
        base_only = data.meta_info.get("base_reward_only", False)

        if base_only:
            if return_dict:
                return {"reward_tensor": base_reward_tensor, "reward_extra_info": {}}
            return base_reward_tensor

        prefix_values, episode_boundaries = self._get_precomputed_prefix_values(data)

        # Step 3: Compute process rewards (cheap: just math)
        if self.phase == 1:
            reward_tensor, utility_stats = compute_step_progress_reward_phase1(
                batch=data,
                base_reward_tensor=base_reward_tensor,
                prefix_values=prefix_values,
                episode_boundaries=episode_boundaries,
                lambda_process=self.lambda_process,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
            )
        elif self.phase == 2:
            # Reset step-level stats before processing this batch
            self.length_tracker.reset_step_stats()

            # Update length baselines (keep in training phase - lightweight)
            prompt_ids = data.batch["prompts"]
            problem_ids = data.non_tensor_batch.get("problem_id", [f"sample_{i}" for i in range(len(data))])
            response_mask = data.batch["attention_mask"][:, prompt_ids.shape[-1]:]
            response_lengths = response_mask.sum(dim=-1).tolist()
            is_correct = (base_reward_tensor.sum(dim=-1) > 0).tolist()

            self.length_tracker.update(
                problem_ids=problem_ids,
                lengths=response_lengths,
                is_correct=is_correct,
            )

            reward_tensor, utility_stats = compute_step_progress_reward_phase2(
                batch=data,
                base_reward_tensor=base_reward_tensor,
                prefix_values=prefix_values,
                episode_boundaries=episode_boundaries,
                length_baseline_tracker=self.length_tracker,
                lambda_process=self.lambda_process,
                beta_length=self.beta_length,
                waste_threshold=self.waste_threshold,
                use_solve_gating=self.use_solve_gating,
                solve_threshold=self.solve_threshold,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
            )
        elif self.phase == "anchor":
            # Reset step-level stats before processing this batch
            self.anchor_tracker.reset_step_stats()

            # Update anchor baselines (tracks both correct and incorrect for pass rate)
            prompt_ids = data.batch["prompts"]
            problem_ids = data.non_tensor_batch.get("problem_id", [f"sample_{i}" for i in range(len(data))])
            response_mask = data.batch["attention_mask"][:, prompt_ids.shape[-1]:]
            response_lengths = response_mask.sum(dim=-1).tolist()
            is_correct = (base_reward_tensor.sum(dim=-1) > 0).tolist()

            self.anchor_tracker.update(
                problem_ids=problem_ids,
                lengths=response_lengths,
                is_correct=is_correct,
            )

            reward_tensor, utility_stats = compute_step_progress_reward_anchor(
                batch=data,
                base_reward_tensor=base_reward_tensor,
                prefix_values=prefix_values,
                episode_boundaries=episode_boundaries,
                anchor_tracker=self.anchor_tracker,
                lambda_process=self.lambda_process,
                beta_length=self.beta_length,
                waste_threshold=self.waste_threshold,
                use_solve_gating=self.use_solve_gating,
                solve_threshold=self.solve_threshold,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
            )
        else:
            raise ValueError(f"Unknown phase: {self.phase}. Must be 1, 2, or 'anchor'.")

        # Prepare extra info for logging
        per_sample_episode_counts = (episode_boundaries[..., 0] != -1).sum(dim=1)
        extra_info = {
            "prefix_values": prefix_values.cpu().numpy(),
            "num_episodes": per_sample_episode_counts.cpu().numpy(),
        }

        # Add utility statistics for monitoring (aggregate stats for this batch)
        extra_info["utility_stats"] = utility_stats

        # Add length baseline statistics for Phase 2 or Anchor (as aggregate stats, not per-sample)
        if self.phase == 2:
            length_stats = self.length_tracker.get_statistics()
            # Use special key that ray_trainer.py extracts for aggregate logging
            extra_info["length_baseline_stats"] = length_stats
        elif self.phase == "anchor":
            anchor_stats = self.anchor_tracker.get_statistics()
            extra_info["anchor_baseline_stats"] = anchor_stats

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": extra_info}
        return reward_tensor
