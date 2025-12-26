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
Utility functions for OPD teacher client-server communication.
"""

import pickle
import time
from types import SimpleNamespace

import torch

from verl import DataProto


def serialize(obj):
    """Serialize Python object for network transmission.

    Args:
        obj: Python object to serialize

    Returns:
        bytes: Serialized object
    """
    return pickle.dumps(obj)


def deserialize(data: bytes):
    """Deserialize bytes back to Python object.

    Args:
        data (bytes): Serialized data

    Returns:
        object: Deserialized Python object
    """
    return pickle.loads(data)


def get_teacher_logprobs(batch: DataProto, teacher_client, n_server_workers: int = 1, is_async: bool = False):
    """Retrieve teacher model's full log probabilities for OPD.

    This function extracts input_ids and attention_mask from a batch, sends them to
    the teacher server, and receives full log probability distributions for computing
    reverse KL divergence.

    Args:
        batch (DataProto): Input batch containing input_ids and attention_mask
        teacher_client: OPDTeacherClient instance for server communication
        n_server_workers (int): Number of parallel workers for teacher inference
        is_async (bool): Whether to use asynchronous processing

    Returns:
        If is_async=True: SimpleNamespace with get() method to process futures
        If is_async=False: Processed DataProto containing teacher log probabilities

    Example:
        >>> from verl.trainer.ppo.opd_teacher import OPDTeacherClient, get_teacher_logprobs
        >>> client = OPDTeacherClient(server_ip="10.0.0.1", server_port=15555)
        >>> teacher_batch = get_teacher_logprobs(batch, client, n_server_workers=1)
        >>> teacher_logprobs = teacher_batch.batch["teacher_log_probs"]  # [batch_size, seq_len, vocab_size]
    """

    # Extract input_ids and attention_mask
    input_ids = []
    attention_mask_bool = batch.batch["attention_mask"].to(torch.bool)

    for ids, mask in zip(batch.batch["input_ids"], attention_mask_bool, strict=False):
        input_ids.append(ids[mask].tolist())

    # Also get attention masks as lists
    attention_masks = []
    for mask in attention_mask_bool:
        attention_masks.append(mask.tolist())

    all_teacher_logprobs = []
    batch_size = len(input_ids)

    assert batch_size % n_server_workers == 0, f"Batch size {batch_size} must be divisible by n_server_workers {n_server_workers}"

    micro_batch_size = batch_size // n_server_workers
    futures = []

    tik1 = time.time()
    tok1 = tik1

    def cb(future):
        nonlocal tok1
        tok1 = max(tok1, time.time())

    # Submit requests to teacher server
    for i in range(0, batch_size, micro_batch_size):
        fut = teacher_client.submit(
            input_ids=input_ids[i : i + micro_batch_size],
            attention_mask=attention_masks[i : i + micro_batch_size],
        )
        fut.add_done_callback(cb)
        futures.append(fut)

    def handle_futures():
        """Process futures and assemble results into DataProto."""
        for future in futures:
            try:
                teacher_logprobs_batch = future.result()
            except Exception as e:
                raise RuntimeError(f"Teacher request failed: {e}") from e

            all_teacher_logprobs.extend(teacher_logprobs_batch)

        tik2 = time.time()

        # Pad logprobs to match input batch shape
        # Teacher returns [seq_len-1, vocab_size] per sequence (predicting next token)
        # We need to pad to [batch_size, max_seq_len, vocab_size] for batch processing

        max_seq_len = batch.batch["input_ids"].shape[1]
        vocab_size = all_teacher_logprobs[0].shape[1] if len(all_teacher_logprobs) > 0 else 50257  # fallback

        # Create padded tensor
        teacher_logprobs_padded = torch.zeros(
            batch_size, max_seq_len, vocab_size, dtype=torch.float32
        )

        for i in range(batch_size):
            logprobs = all_teacher_logprobs[i]  # [seq_len-1, vocab_size]
            seq_len = logprobs.shape[0]

            # Place logprobs in padded tensor
            # Note: logprobs[i] predicts token[i+1], so we offset by 1
            if seq_len > 0:
                mask = attention_mask_bool[i]
                # Find positions where mask is True
                valid_positions = torch.where(mask)[0]
                if len(valid_positions) > 1:
                    # Place logprobs starting from position 1 (predicting position 2 onwards)
                    end_pos = min(len(valid_positions), seq_len + 1)
                    teacher_logprobs_padded[i, valid_positions[1:end_pos]] = logprobs[:end_pos - 1]

        # Convert to log probabilities tensor (ensure it's log probs, not raw logits)
        # vLLM already returns log probabilities, so no need to apply log_softmax

        output_batch = DataProto.from_single_dict(
            data={"teacher_log_probs": teacher_logprobs_padded},
        )

        tok2 = time.time()
        output_batch.meta_info["timing"] = {
            "get_teacher_logprobs": (tok1 - tik1) + (tok2 - tik2)
        }

        return output_batch

    if is_async:
        return SimpleNamespace(get=handle_futures)
    else:
        return handle_futures()


if __name__ == "__main__":
    # Test serialization
    test_obj = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
    serialized = serialize(test_obj)
    deserialized = deserialize(serialized)
    assert test_obj == deserialized
    print("Serialization test passed")
