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
)
from length_baseline import LengthBaselineTracker


@register("step_progress_reward")
class StepProgressRewardManager(AbstractRewardManager):
    """
    Reward manager for step progress rewards (overthinking mitigation).

    Computes rewards as:
        Phase 1: R = R_final + λ * Σ_i clip(U_i)
        Phase 2: R = R_final + λ * Σ_i U_i - Σ_i c_i

    where:
        U_i = V(prefix_i) - V(prefix_{i-1})  (marginal utility)
        V(prefix) = P(correct | stop at prefix and answer)

    This encourages the model to make consistent progress and avoid
    redundant verification loops after already solving the problem.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Callable,  # Base reward function (e.g., bigmath_reward)
        reward_fn_key: str = "data_source",
        actor_forward_fn: Optional[Callable] = None,  # For V(prefix) estimation
        step_progress_reward_config: Optional[dict] = None,
    ):
        """
        Initialize StepProgressRewardManager.

        Args:
            tokenizer: Tokenizer for text processing
            num_examine: Number of samples to print for debugging
            compute_score: Base reward function for outcome correctness
            reward_fn_key: Key to use for reward function selection (default: "data_source")
            actor_forward_fn: Function to compute log probs from actor model
                             Signature: (input_ids, attention_mask) -> dict with "log_probs"
                             Required for process rewards, falls back to base rewards if None
            step_progress_reward_config: Configuration dict with keys:
                - phase: 1 or 2 (default: 1)
                - lambda: Weight for process rewards (default: 0.1)
                - clip_min, clip_max: Clipping bounds for Phase 1 (default: -0.5, 0.5)
                - discourse_markers: List of markers for episode segmentation
                - beta_length: Weight for length penalty (Phase 2, default: 0.01)
                - waste_threshold: Threshold for wasteful episodes (Phase 2, default: 0.0)
                - use_solve_gating: Whether to only penalize post-solve episodes (Phase 2, default: False)
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.actor_forward_fn = actor_forward_fn

        # Parse configuration
        config = step_progress_reward_config or {}
        self.phase = config.get("phase", 1)
        self.lambda_process = config.get("lambda", 0.1)

        # Phase 1 parameters
        self.clip_min = config.get("clip_min", -0.5)
        self.clip_max = config.get("clip_max", 0.5)

        # Phase 2 parameters
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

        # Initialize length baseline tracker (Phase 2 only)
        if self.phase == 2:
            self.length_tracker = LengthBaselineTracker(
                quantile=config.get("length_quantile", 0.6),
                ema_alpha=config.get("length_ema_alpha", 0.1),
            )

        # For printing samples
        self.num_printed = 0

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

            # Print some samples for debugging
            if self.num_printed < self.num_examine:
                print(f"[StepProgressReward] Sample {self.num_printed}: score={scores[i]:.2f}")
                print(f"  Response: {sequences_str[i][:200]}...")
                self.num_printed += 1

        return reward_tensor

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """
        Compute step progress rewards for a batch.

        Steps:
        1. Compute base correctness rewards (R_final)
        2. If actor_forward_fn available:
           a. Segment responses into episodes
           b. Estimate V(prefix) for each episode
           c. Compute marginal utilities U_i
           d. Apply process reward formula (Phase 1 or 2)
        3. Return token-level rewards

        Args:
            data: DataProto containing batch data
            return_dict: If True, return dict with "reward_tensor" and "reward_extra_info"

        Returns:
            reward_tensor: [batch_size, seq_len] token-level rewards
            OR dict with {"reward_tensor": ..., "reward_extra_info": {...}}
        """
        # Check if rewards already computed (from reward loop)
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        # Step 1: Compute base correctness rewards
        base_reward_tensor = self._compute_base_rewards(data)

        # If no actor_forward_fn, fall back to base rewards only
        if self.actor_forward_fn is None:
            print("[StepProgressReward] Warning: actor_forward_fn not provided, using base rewards only")
            if return_dict:
                return {"reward_tensor": base_reward_tensor}
            return base_reward_tensor

        # Step 2: Segment into episodes
        # Need to construct input_ids from prompts + responses
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)

        # Response mask
        prompt_length = prompt_ids.shape[-1]
        response_mask = data.batch["attention_mask"][:, prompt_length:]

        episode_boundaries = self.segmenter.segment_batch(
            input_ids=input_ids,
            response_mask=response_mask,
        )

        # Step 3: Extract ground truths
        ground_truths = [
            item["ground_truth"]
            for item in data.non_tensor_batch["reward_model"]
        ]

        # Step 4: Estimate prefix values using actor model
        prefix_values = self.value_estimator.estimate_prefix_values_batch(
            batch=data,
            episode_boundaries=episode_boundaries,
            ground_truths=ground_truths,
            actor_model_fn=self.actor_forward_fn,
        )

        # Step 5: Compute process rewards
        if self.phase == 1:
            reward_tensor = compute_step_progress_reward_phase1(
                batch=data,
                base_reward_tensor=base_reward_tensor,
                prefix_values=prefix_values,
                episode_boundaries=episode_boundaries,
                lambda_process=self.lambda_process,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
            )
        else:  # phase == 2
            # Update length baselines from this batch
            problem_ids = data.non_tensor_batch.get("problem_id", [f"sample_{i}" for i in range(len(data))])
            response_lengths = response_mask.sum(dim=-1).tolist()
            is_correct = (base_reward_tensor.sum(dim=-1) > 0).tolist()

            self.length_tracker.update(
                problem_ids=problem_ids,
                lengths=response_lengths,
                is_correct=is_correct,
            )

            reward_tensor = compute_step_progress_reward_phase2(
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
            )

        # Prepare extra info for logging
        extra_info = {
            "prefix_values": prefix_values.cpu().numpy(),
            "num_episodes": int(episode_boundaries.shape[1]),
        }

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": extra_info}
        return reward_tensor
