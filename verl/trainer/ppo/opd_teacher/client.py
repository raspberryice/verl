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
OPD Teacher Client for communicating with remote teacher model server.

This client sends student-generated trajectories to a teacher model server
and receives full log probabilities for knowledge distillation.
"""

import queue
import threading
from concurrent.futures import Future
from contextlib import nullcontext

import zmq

try:
    from .utils import deserialize, serialize
except ImportError:
    from utils import deserialize, serialize


class OPDTeacherClient:
    """Client for OPD teacher model server communication.

    This client communicates with a remote teacher model server via ZeroMQ to obtain
    full log probabilities for student-generated sequences. Unlike GKD which uses top-k
    logprobs, this returns complete log probability distributions for reverse KL computation.

    Args:
        server_ip (str): IP address of the teacher server
        server_port (int): Port number of the teacher server
        num_microbatches (int): Number of microbatches to process per request (default: 1)
        n_server_workers (int): Number of parallel server workers (default: 1)
        timeout_ms (int): Timeout for server responses in milliseconds (default: 600000 = 10min)

    Example:
        >>> client = OPDTeacherClient(server_ip="10.0.0.1", server_port=15555)
        >>> future = client.submit(input_ids_list, attention_mask_list)
        >>> teacher_logprobs = future.result()  # [batch_size, seq_len, vocab_size]
    """

    def __init__(
        self,
        server_ip: str,
        server_port: int,
        num_microbatches: int = 1,
        n_server_workers: int = 1,
        timeout_ms: int = 600000,
    ) -> None:
        self.server_ip = server_ip
        self.server_port = server_port
        self.num_microbatches = num_microbatches
        self.n_server_workers = n_server_workers
        self.timeout_ms = timeout_ms

        self.task_queue = queue.Queue()
        self.mutex = threading.Lock() if n_server_workers > 1 else nullcontext()
        self.context = zmq.Context()

        # Start background worker threads
        self._run()

    def bg_task(self):
        """Background worker thread that processes requests to teacher server.

        This thread:
        1. Collects microbatches from the task queue
        2. Sends batched requests to the teacher server
        3. Receives full log probabilities from the server
        4. Distributes results to corresponding futures
        """
        socket = self.context.socket(zmq.REQ)
        socket.connect(f"tcp://{self.server_ip}:{self.server_port}")
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)

        while True:
            futures = []
            batch_input_ids = []
            batch_attention_mask = []

            try:
                with self.mutex:
                    # Collect microbatches
                    for _ in range(self.num_microbatches):
                        future, input_ids, attention_mask = self.task_queue.get()
                        futures.append(future)
                        batch_input_ids.extend(input_ids)
                        batch_attention_mask.extend(attention_mask)

                # Prepare request
                request = {
                    "input_ids": batch_input_ids,
                    "attention_mask": batch_attention_mask,
                }

                # Send request to teacher server
                socket.send(serialize(request))
                raw = socket.recv()
                response = deserialize(raw)

                # Handle errors
                if isinstance(response, dict) and response.get("status") == "error":
                    reason = response.get("reason", "unknown")
                    err = RuntimeError(f"OPD Teacher server error: {reason}")
                    for f in futures:
                        f.set_exception(err)
                    continue

                # Validate response
                if "logprobs" not in response:
                    raise RuntimeError("Invalid response from teacher: missing 'logprobs' key")

                teacher_logprobs = response["logprobs"]
                total = len(teacher_logprobs)

                if self.num_microbatches <= 0 or total % self.num_microbatches != 0:
                    raise RuntimeError(
                        f"Size mismatch: total={total}, num_microbatches={self.num_microbatches}"
                    )

                # Distribute results to futures
                mbs = total // self.num_microbatches
                for i, future in enumerate(futures):
                    s, e = i * mbs, (i + 1) * mbs
                    logprobs_slice = teacher_logprobs[s:e]
                    future.set_result(logprobs_slice)

            except zmq.Again:
                err = TimeoutError(
                    f"Timeout waiting for OPD teacher server {self.server_ip}:{self.server_port}"
                )
                for f in futures:
                    f.set_exception(err)
                continue

            except Exception as e:
                for f in futures:
                    try:
                        f.set_exception(e)
                    except Exception:
                        pass
                continue

    def _run(self):
        """Start background worker threads."""
        for _ in range(self.n_server_workers):
            threading.Thread(target=self.bg_task, daemon=True).start()

    def submit(self, input_ids: list, attention_mask: list) -> Future:
        """Submit a request to the teacher server for log probability computation.

        Args:
            input_ids (list): List of input ID sequences (each is a list of ints)
            attention_mask (list): List of attention masks (each is a list of ints/bools)

        Returns:
            Future: Future object that will contain teacher log probabilities
                   Result will be a list of tensors [batch_size, seq_len, vocab_size]
        """
        future = Future()
        self.task_queue.put((future, input_ids, attention_mask))
        return future

    def __del__(self):
        """Cleanup ZeroMQ context on deletion."""
        self.context.destroy()


if __name__ == "__main__":
    # Example usage
    import torch

    client = OPDTeacherClient(
        server_ip="127.0.0.1",
        server_port=15555,
        num_microbatches=1,
        n_server_workers=1,
    )

    # Example batch
    batch_size = 4
    seq_len = 128
    input_ids = [[i for i in range(seq_len)] for _ in range(batch_size)]
    attention_mask = [[1] * seq_len for _ in range(batch_size)]

    # Submit request
    future = client.submit(input_ids, attention_mask)

    # Get result
    teacher_logprobs = future.result()
    print(f"Received {len(teacher_logprobs)} sequences")
    print(f"Shape of first sequence logprobs: {teacher_logprobs[0].shape}")
