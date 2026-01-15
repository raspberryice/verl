# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Step Progress Reward Manager for the reward loop architecture.

Computes process rewards based on marginal utility of reasoning episodes.
Uses vLLM with prompt_logprobs for on-demand prefix value estimation
(no dependency on prefix_value_cache from generation phase).

Phase 1: R = R_final + lambda * sum_i clip(U_i)
Phase 2: R = R_final + lambda * sum_i U_i - sum_i c_i (with length penalties)
"""

import asyncio
import inspect
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score

# Note: reward_functions imports are done lazily in init_class() to avoid
# import errors when the module is loaded on Ray workers before PYTHONPATH is set

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("step_progress")
class StepProgressRewardManager(RewardManagerBase):
    """
    Reward manager for step progress rewards (overthinking mitigation).

    Computes rewards as:
        Phase 1: R = R_final + lambda * sum_i clip(U_i)
        Phase 2: R = R_final + lambda * sum_i U_i - sum_i c_i

    Uses vLLM with prompt_logprobs to estimate prefix values V(prefix) on-demand.
    Does NOT require prefix_value_cache from generation phase.
    """

    # Class-level shared state
    _length_tracker: Optional[LengthBaselineTracker] = None
    _segmenter: Optional[EpisodeSegmenter] = None

    def __init__(
        self,
        config: DictConfig,
        tokenizer: AutoTokenizer,
        compute_score: Optional[Callable] = None,
        reward_router_address: Optional[str] = None,
        reward_model_tokenizer: Optional[AutoTokenizer] = None,
    ):
        super().__init__(config, tokenizer)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer: PreTrainedTokenizerBase = reward_model_tokenizer or tokenizer  # type: ignore

        # Parse reward config
        self.reward_kwargs = config.reward_model.get("reward_kwargs", {})
        self.phase = self.reward_kwargs.get("phase", 1)
        self.lambda_process = self.reward_kwargs.get("lambda_process", 0.1)
        self.clip_min = self.reward_kwargs.get("clip_min", -0.5)
        self.clip_max = self.reward_kwargs.get("clip_max", 0.5)

        # Phase 2 parameters
        self.beta_length = self.reward_kwargs.get("beta_length", 0.01)
        self.waste_threshold = self.reward_kwargs.get("waste_threshold", 0.0)
        self.use_solve_gating = self.reward_kwargs.get("use_solve_gating", False)
        self.solve_threshold = self.reward_kwargs.get("solve_threshold", 0.8)

        # Value estimation parameters
        self.force_answer_prompt = self.reward_kwargs.get(
            "force_answer_prompt",
            "I need to stop thinking. I think the final answer is \\boxed{"
        )
        self.ground_truth_max_tokens = self.reward_kwargs.get("ground_truth_max_tokens", 32)
        self.value_computation = self.reward_kwargs.get("value_computation", "log_mean")

        # Model name for vLLM requests
        self.model_name = config.reward_model.model.get("path", "default")

        # HTTP session for async requests
        self._session: Optional[aiohttp.ClientSession] = None

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer):
        """Initialize class-level shared state."""
        if cls._class_initialized:
            return

        # Lazy import reward_functions modules (may not be in PYTHONPATH at module load time)
        from reward_functions.episode_segmenter import EpisodeSegmenter
        from reward_functions.length_baseline import LengthBaselineTracker

        reward_kwargs = config.reward_model.get("reward_kwargs", {})

        # Initialize shared segmenter (cast tokenizer for type checker)
        cls._segmenter = EpisodeSegmenter(
            tokenizer=tokenizer,  # type: ignore
            discourse_markers=reward_kwargs.get("discourse_markers", None),
            pattern_markers=reward_kwargs.get("pattern_markers", None),
            min_episode_tokens=reward_kwargs.get("min_episode_tokens", 32),
        )

        # Initialize shared length tracker (Phase 2 only)
        if reward_kwargs.get("phase", 1) == 2:
            cls._length_tracker = LengthBaselineTracker(
                quantile=reward_kwargs.get("length_quantile", 0.6),
                ema_alpha=reward_kwargs.get("length_ema_alpha", 0.1),
                buffer_size=reward_kwargs.get("length_buffer_size", 100),
                min_samples=reward_kwargs.get("length_min_samples", 5),
                lenient_fallback=reward_kwargs.get("lenient_fallback", 16384.0),
            )

        cls._class_initialized = True
        logger.info(f"StepProgressRewardManager initialized: phase={reward_kwargs.get('phase', 1)}")

    async def run_single(self, data: DataProto) -> dict:
        """Main entry point for computing step progress reward."""
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        # Step 1: Compute base correctness reward
        base_reward = await self._compute_base_reward(data_item)

        # Step 2: Segment response into episodes
        episodes = self._segment_response(data_item)

        if not episodes or len(episodes) == 0:
            # No episodes found, return base reward only
            return {
                "reward_score": base_reward,
                "reward_extra_info": {
                    "base_reward": base_reward,
                    "process_reward": 0.0,
                    "num_episodes": 0,
                }
            }

        # Step 3-5: Compute prefix values via batched vLLM request
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        prefix_values = await self._estimate_prefix_values_batched(
            data_item, episodes, ground_truth
        )

        # Step 6-7: Compute process reward and combine
        process_reward, extra_info = self._compute_process_reward(
            data_item, episodes, prefix_values, base_reward  # episodes passed for future use
        )

        final_reward = base_reward + process_reward

        return {
            "reward_score": final_reward,
            "reward_extra_info": {
                "base_reward": base_reward,
                "process_reward": process_reward,
                "num_episodes": len(episodes),
                "prefix_values": prefix_values,
                **extra_info,
            }
        }

    async def _compute_base_reward(self, data_item: Any) -> float:
        """Compute base correctness reward using compute_score function."""
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum())
        valid_response_ids = response_ids[:valid_response_length]

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)  # type: ignore
        )

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )

        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        if isinstance(result, dict):
            return float(result.get("score", 0.0))
        return float(result)

    def _segment_response(self, data_item: Any) -> List[Tuple[int, int]]:
        """Segment response into episodes using EpisodeSegmenter."""
        if self._segmenter is None:
            raise RuntimeError("EpisodeSegmenter not initialized. Call init_class first.")

        input_ids = data_item.batch["input_ids"]
        attention_mask = data_item.batch["attention_mask"]

        # Construct response mask from prompts shape
        prompt_length = data_item.batch["prompts"].shape[-1]
        response_mask = attention_mask.clone()
        response_mask[:prompt_length] = 0

        return self._segmenter.segment_response(input_ids, response_mask)

    async def _estimate_prefix_values_batched(
        self,
        data_item: Any,
        episodes: List[Tuple[int, int]],
        ground_truth: str,
    ) -> List[float]:
        """
        Estimate V(prefix) for all episodes via single batched vLLM request.

        Leverages vLLM's automatic prefix caching for shared prefixes.
        """
        # Tokenize ground truth once
        gt_tokens = self.reward_model_tokenizer.encode(ground_truth, add_special_tokens=False)
        gt_tokens = gt_tokens[:self.ground_truth_max_tokens]
        gt_len = len(gt_tokens)

        if gt_len == 0:
            return [0.0] * len(episodes)

        # Build all prefix sequences
        prompts = []
        input_ids = data_item.batch["input_ids"]

        # Pre-tokenize force_answer_prompt and think close tag
        force_answer_tokens = self.reward_model_tokenizer.encode(
            self.force_answer_prompt, add_special_tokens=False
        )
        think_close_tokens = self.reward_model_tokenizer.encode("</think>", add_special_tokens=False)

        for _start_idx, end_idx in episodes:
            # Extract prefix tokens up to episode end (inclusive)
            prefix_tokens = input_ids[:end_idx + 1].tolist()

            # Check if prefix already ends with </think>
            prefix_ends_with_close = (
                len(prefix_tokens) >= len(think_close_tokens) and
                prefix_tokens[-len(think_close_tokens):] == think_close_tokens
            )

            # Build full sequence: prefix + [</think>] + force_answer + ground_truth
            if prefix_ends_with_close:
                full_tokens = prefix_tokens + force_answer_tokens + gt_tokens
            else:
                full_tokens = prefix_tokens + think_close_tokens + force_answer_tokens + gt_tokens

            # Decode to string for vLLM
            full_prompt = self.reward_model_tokenizer.decode(full_tokens, skip_special_tokens=False)
            prompts.append(full_prompt)

        # Single batched request to vLLM
        payload = {
            "model": self.model_name,
            "prompt": prompts,
            "max_tokens": 0,
            "prompt_logprobs": 1,
            "temperature": 1.0,
        }

        try:
            response = await self._post_request(payload, "v1/completions")
        except Exception as e:
            logger.warning(f"vLLM request failed: {e}. Returning zero prefix values.")
            return [0.0] * len(episodes)

        # Extract and aggregate prefix values
        values: List[float] = []
        choices = response.get("choices", [])

        for choice in choices:
            prompt_logprobs = choice.get("prompt_logprobs", [])
            if not prompt_logprobs or len(prompt_logprobs) < gt_len:
                values.append(0.0)
                continue

            # Extract logprobs for ground truth tokens (last gt_len positions)
            gt_logprobs: List[float] = []
            for j, token_id in enumerate(gt_tokens):
                pos = -(gt_len - j)
                if abs(pos) <= len(prompt_logprobs):
                    logprob_entry = prompt_logprobs[pos]
                    if logprob_entry and isinstance(logprob_entry, dict):
                        # vLLM returns {token_id: logprob} or {token_str: logprob}
                        # Try token_id first, then string representation
                        if token_id in logprob_entry:
                            gt_logprobs.append(logprob_entry[token_id])
                        elif str(token_id) in logprob_entry:
                            gt_logprobs.append(logprob_entry[str(token_id)])
                        else:
                            # Find the logprob for this position (top logprob)
                            gt_logprobs.append(-10.0)  # Low probability fallback
                    else:
                        gt_logprobs.append(-10.0)
                else:
                    gt_logprobs.append(-10.0)

            # Aggregate to single value
            value = self._aggregate_logprobs(gt_logprobs)
            values.append(value)

        # Apply baseline correction for log_mean (shift so first episode can have positive utility)
        if self.value_computation == "log_mean" and len(values) > 0:
            baseline = float(np.percentile(values, 10)) if len(values) > 1 else min(values)
            values = [v - baseline for v in values]

        return values

    def _aggregate_logprobs(self, logprobs: List[float]) -> float:
        """Aggregate log probabilities to a single value estimate."""
        if not logprobs:
            return 0.0

        logprobs_arr = np.array(logprobs)

        if self.value_computation == "log_mean":
            return float(np.mean(logprobs_arr))
        elif self.value_computation == "geometric_mean":
            return float(np.exp(np.mean(logprobs_arr)))
        elif self.value_computation == "arithmetic_mean":
            return float(np.mean(np.exp(logprobs_arr)))
        elif self.value_computation == "joint_prob":
            return float(np.exp(np.sum(logprobs_arr)))
        else:
            return float(np.mean(logprobs_arr))

    async def _post_request(self, payload: dict, endpoint: str, max_retries: int = 8) -> dict:
        """POST request to vLLM server with retry logic."""
        if self.reward_router_address is None:
            raise ValueError("reward_router_address is required for prefix value estimation")

        url = f"http://{self.reward_router_address}/{endpoint}"
        last_exception: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=120)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)
                    logger.debug(f"vLLM request failed (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Unknown error in _post_request")

    def _compute_process_reward(
        self,
        data_item: Any,
        _episodes: List[Tuple[int, int]],  # kept for future token-level reward assignment
        prefix_values: List[float],
        base_reward: float,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute process reward from prefix values.

        Returns (process_reward, extra_info_dict)
        """
        # Compute marginal utilities: U_i = V_i - V_{i-1}
        utilities: List[float] = []
        for i, v in enumerate(prefix_values):
            if i == 0:
                utilities.append(v)  # U_0 = V_0 - 0
            else:
                utilities.append(v - prefix_values[i - 1])

        extra_info: Dict[str, Any] = {"marginal_utilities": utilities}

        if self.phase == 1:
            # Phase 1: R = lambda * sum clip(U_i)
            clipped_utils = [
                max(self.clip_min, min(self.clip_max, u))
                for u in utilities
            ]
            process_reward = self.lambda_process * sum(clipped_utils)
            extra_info["clipped_utilities"] = clipped_utils

        else:  # Phase 2
            # Phase 2: R = lambda * sum U_i - sum c_i
            process_reward = self.lambda_process * sum(utilities)

            # Length penalty (if tracker available)
            if self._length_tracker is not None:
                problem_id = data_item.non_tensor_batch.get("problem_id", "unknown")
                if problem_id == "unknown":
                    # Try to extract from extra_info
                    extra_info_dict = data_item.non_tensor_batch.get("extra_info", {})
                    problem_id = extra_info_dict.get("problem_id", "unknown")

                # Compute response length
                response_length = int(data_item.batch["attention_mask"].sum())
                L_ref = self._length_tracker.get_baseline(problem_id)

                if L_ref > 0:
                    excess = max(0, (response_length - L_ref) / L_ref)
                    length_penalty = self.beta_length * excess
                    process_reward -= length_penalty
                    extra_info["length_penalty"] = length_penalty
                    extra_info["baseline_length"] = L_ref
                    extra_info["actual_length"] = response_length
                    extra_info["excess_ratio"] = excess

                # Update tracker if correct
                if base_reward > 0:
                    self._length_tracker.update(
                        problem_ids=[problem_id],
                        lengths=[response_length],
                        is_correct=[True],
                    )

            # Solve gating (optional): only penalize post-solve episodes
            if self.use_solve_gating:
                solve_point: Optional[int] = None
                for i, v in enumerate(prefix_values):
                    if v >= self.solve_threshold:
                        solve_point = i
                        break
                extra_info["solve_point"] = solve_point

        return process_reward, extra_info
