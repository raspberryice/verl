"""
Prefix Log Prob Populator

Pre-computes prefix log probs during generation phase and stores in batch metadata.
This is a REQUIRED component (not an optional cache) - training will fail if values
are missing. Moves expensive computation (70% overhead) out of critical training path.
"""

from typing import Callable, Optional
import torch
from verl.protocol import DataProto
from transformers import PreTrainedTokenizer

# Import reward function components
import sys
from pathlib import Path
reward_functions_path = Path(__file__).parent.parent.parent.parent.parent / "reward_functions"
if str(reward_functions_path) not in sys.path:
    sys.path.insert(0, str(reward_functions_path))

from episode_segmenter import EpisodeSegmenter
from prefix_value_estimator import PrefixValueEstimator


class PrefixLogProbPopulator:
    """
    Pre-computes prefix log probs during generation phase.

    This is a REQUIRED component (not an optional cache) that moves 70% of reward
    computation overhead out of the critical training path. Training will fail if
    prefix values are not populated - there is no silent fallback.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        actor_forward_fn: Callable,
        step_progress_reward_config: Optional[dict] = None,
    ):
        """
        Initialize prefix log prob populator.

        Args:
            tokenizer: Tokenizer for text processing
            actor_forward_fn: Function to compute log probs from actor model
                Signature: (input_ids, attention_mask, responses) -> List[Tensor]
                where responses is List[Tensor] of response tokens per sequence
            step_progress_reward_config: Configuration dict (same as StepProgressRewardManager)
        """
        self.tokenizer = tokenizer
        self.actor_forward_fn = actor_forward_fn

        # Parse configuration
        config = step_progress_reward_config or {}

        # Initialize episode segmenter
        self.segmenter = EpisodeSegmenter(
            tokenizer=tokenizer,
            discourse_markers=config.get("discourse_markers", None),
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

    def populate_cache(self, batch: DataProto) -> DataProto:
        """
        Pre-compute prefix log probs and store in batch metadata.

        Called after generate_sequences() completes, before batch is passed
        to training loop. This moves expensive computation out of critical path.
        This is REQUIRED - training will fail if not called.

        Args:
            batch: DataProto with generated responses

        Returns:
            batch: Same batch with prefix_value_cache added to meta_info
        """
        # Check if ground truths are available
        if "reward_model" not in batch.non_tensor_batch:
            raise RuntimeError(
                "[PrefixLogProb] Missing reward_model data in batch - cannot pre-compute prefix values"
            )

        # Step 1: Segment into episodes
        prompt_ids = batch.batch["prompts"]
        response_ids = batch.batch["responses"]
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)

        prompt_length = prompt_ids.shape[-1]

        # Build response_mask that matches input_ids shape
        # segment_response expects response_mask to have same shape as input_ids
        # with 1s marking valid response tokens (not prompt tokens)
        attention_mask = batch.batch["attention_mask"]
        response_mask = torch.zeros_like(input_ids, dtype=attention_mask.dtype)
        response_mask[:, prompt_length:] = attention_mask[:, prompt_length:]

        episode_boundaries = self.segmenter.segment_batch(
            input_ids=input_ids,
            response_mask=response_mask,
        )

        # Step 2: Extract ground truths
        ground_truths = [
            item["ground_truth"]
            for item in batch.non_tensor_batch["reward_model"]
        ]

        # Step 3: Estimate prefix values (EXPENSIVE - but in generation phase)
        print(f"[PrefixLogProb] Pre-computing prefix values for {len(batch)} samples "
              f"with {episode_boundaries.shape[1]} episodes...")

        prefix_values = self.value_estimator.estimate_prefix_values_batch(
            batch=batch,
            episode_boundaries=episode_boundaries,
            ground_truths=ground_truths,
            actor_model_fn=self.actor_forward_fn,
        )

        # Step 4: Store in batch metadata
        batch.meta_info["prefix_value_cache"] = {
            "prefix_values": prefix_values,
            "episode_boundaries": episode_boundaries,
            "cache_timestamp": batch.meta_info.get("global_steps", 0),
        }

        print(f"[PrefixLogProb] Pre-computation complete "
              f"(prefix_values shape: {prefix_values.shape})")

        return batch
