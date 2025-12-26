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
OPD Teacher Module for On-Policy Distillation.

This module provides a client-server architecture for efficiently computing teacher
log probabilities on student-generated trajectories. The teacher model runs on separate
hardware with independent TP/PP settings, avoiding GPU memory conflicts with student training.

Key differences from GKD teacher:
- Returns full log probabilities (not top-k) for reverse KL computation
- Computes on student trajectories (not teacher samples)
- Supports selective processing based on OPD eligibility masks
"""

from .client import OPDTeacherClient
from .utils import get_teacher_logprobs

__all__ = ["OPDTeacherClient", "get_teacher_logprobs"]
