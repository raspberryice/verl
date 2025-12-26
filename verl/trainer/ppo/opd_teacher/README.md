# OPD Teacher Module

On-Policy Distillation (OPD) Teacher Server for computing log probabilities on student trajectories.

## Overview

This module provides a **client-server architecture** for efficiently obtaining teacher model guidance during OPD training. The teacher model runs on **separate hardware** with independent TP/PP settings, avoiding GPU memory conflicts with student training.

### Key Features

✅ **Separate Infrastructure**: Teacher runs on different GPUs/machines from student
✅ **Flexible Scaling**: Independent TP settings (e.g., teacher TP=4, student TP=2)
✅ **Full Logprobs**: Returns complete log probability distributions for reverse KL
✅ **Async Processing**: Non-blocking requests with concurrent futures
✅ **Large Teachers**: Supports teachers much larger than students (e.g., 70B teacher, 4B student)

### Differences from GKD Teacher

| Feature | GKD Teacher | OPD Teacher |
|---------|-------------|-------------|
| Output | Top-k logprobs | Full log probabilities |
| Input | Prompts (for generation) | Student trajectories (teacher-forcing) |
| Use Case | Knowledge distillation from samples | Reverse KL on student rollouts |
| Selectivity | All samples | OPD-eligible samples only |

## Architecture

```
┌─────────────────────┐         ZeroMQ          ┌──────────────────────┐
│  Student Training   │◄──────────────────────►│  Teacher Server      │
│  (Main GPUs)        │    TCP Connection       │  (Separate GPUs)     │
│                     │                         │                      │
│  - Student Model    │   Request:              │  - Teacher Model     │
│  - RL Training      │   - input_ids           │  - vLLM Engine       │
│  - OPD Loss         │   - attention_mask      │  - Log Prob Compute  │
│                     │                         │                      │
│                     │   Response:             │                      │
│                     │   - teacher_log_probs   │                      │
│                     │     [B, L, V]           │                      │
└─────────────────────┘                         └──────────────────────┘
```

## Installation

No additional dependencies beyond VeRL requirements. Ensure vLLM is installed:

```bash
pip install vllm
```

## Usage

### 1. Start Teacher Server

On a **separate machine** or set of GPUs:

```bash
# Set GPUs for teacher (different from student training)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Start teacher server
export TEACHER_MODEL_PATH=/path/to/teacher/checkpoint
export PORT=15555
export TP=4  # Tensor parallelism for teacher

./verl/verl/trainer/ppo/opd_teacher/start_server.sh
```

Or run directly:

```bash
python3 -m verl.trainer.ppo.opd_teacher.server \
    --model_path /path/to/teacher/checkpoint \
    --port 15555 \
    --tp 4 \
    --gpu_memory_utilization 0.9 \
    --dtype bfloat16 \
    --max_model_len 16384
```

**Server Arguments:**
- `--model_path`: Path to teacher model checkpoint (required)
- `--port`: Port to listen on (default: 15555)
- `--tp`: Tensor parallel size (default: 1)
- `--gpu_memory_utilization`: GPU memory fraction (default: 0.9)
- `--dtype`: Model dtype (default: "bfloat16")
- `--max_model_len`: Maximum sequence length (default: None)
- `--trust_remote_code`: Trust remote code flag

### 2. Configure Training to Use Teacher Server

In your training config or script, specify the teacher server:

```yaml
# In your training YAML config
algorithm:
  opd:
    enable: true
    teacher_server_ip: "10.0.0.1"  # IP of machine running teacher server
    teacher_server_port: 15555
    teacher_n_workers: 1  # Number of parallel workers
```

Or in your training script:

```python
from verl.trainer.ppo.opd_teacher import OPDTeacherClient, get_teacher_logprobs

# Initialize client
teacher_client = OPDTeacherClient(
    server_ip="10.0.0.1",
    server_port=15555,
    n_server_workers=1,
)

# During training, fetch teacher logprobs
teacher_batch = get_teacher_logprobs(
    batch=student_batch,
    teacher_client=teacher_client,
    n_server_workers=1,
    is_async=False,  # Set to True for async processing
)

# Extract teacher log probabilities
teacher_log_probs = teacher_batch.batch["teacher_log_probs"]  # [batch_size, seq_len, vocab_size]
```

### 3. Example: Multi-Machine Setup

**Machine 1 (Teacher Server)**:
- Hardware: 4x H100 80GB
- Model: Qwen-72B-Instruct (teacher)
- TP: 4

```bash
# On Machine 1 (10.0.0.1)
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m verl.trainer.ppo.opd_teacher.server \
    --model_path Qwen/Qwen2.5-72B-Instruct \
    --port 15555 \
    --tp 4 \
    --gpu_memory_utilization 0.95
```

**Machine 2 (Student Training)**:
- Hardware: 8x H200 141GB
- Model: Qwen-4B-Base (student)
- TP: 2

```bash
# On Machine 2 (10.0.0.2)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TEACHER_SERVER_IP=10.0.0.1
export TEACHER_SERVER_PORT=15555

# Run training with OPD
./configs/grpo_opd_qwen3-4b_bigmath_8xh200.sh
```

## API Reference

### OPDTeacherClient

```python
class OPDTeacherClient:
    def __init__(
        self,
        server_ip: str,
        server_port: int,
        num_microbatches: int = 1,
        n_server_workers: int = 1,
        timeout_ms: int = 600000,
    ):
        """Initialize OPD teacher client.

        Args:
            server_ip: IP address of teacher server
            server_port: Port number of teacher server
            num_microbatches: Number of microbatches per request
            n_server_workers: Number of parallel workers
            timeout_ms: Request timeout in milliseconds
        """

    def submit(self, input_ids: list, attention_mask: list) -> Future:
        """Submit async request for teacher logprobs.

        Args:
            input_ids: List of token ID sequences
            attention_mask: List of attention masks

        Returns:
            Future containing teacher log probabilities
        """
```

### get_teacher_logprobs

```python
def get_teacher_logprobs(
    batch: DataProto,
    teacher_client: OPDTeacherClient,
    n_server_workers: int = 1,
    is_async: bool = False,
) -> DataProto:
    """Retrieve teacher log probabilities for a batch.

    Args:
        batch: Input batch with input_ids and attention_mask
        teacher_client: OPDTeacherClient instance
        n_server_workers: Number of parallel workers
        is_async: Whether to process asynchronously

    Returns:
        DataProto with teacher_log_probs: [batch_size, seq_len, vocab_size]
    """
```

## Performance Considerations

### Network Latency
- Use gigabit/10G ethernet for low latency
- Consider async processing (`is_async=True`) to overlap communication
- Batch requests when possible

### Teacher Model Size
- Larger teachers need more TP (e.g., 70B → TP=4 or TP=8)
- Adjust `gpu_memory_utilization` based on available VRAM
- Use bf16/fp16 for memory efficiency

### Throughput
- Multiple `n_server_workers` for concurrent processing
- Teacher server can handle multiple training jobs
- Monitor teacher GPU utilization

## Troubleshooting

### Connection Errors

```python
TimeoutError: Timeout waiting for OPD teacher server
```
**Solution**: Check network connectivity, firewall rules, and that server is running.

### Memory Errors on Teacher

```
OutOfMemoryError: CUDA out of memory
```
**Solution**: Reduce `gpu_memory_utilization`, increase TP, or use smaller teacher.

### Shape Mismatches

```
RuntimeError: Size mismatch: total=16, num_microbatches=8
```
**Solution**: Ensure `batch_size % num_microbatches == 0` and `batch_size % n_server_workers == 0`.

## Limitations & Future Work

**Current Limitations:**
- vLLM's prompt_logprobs API is used as a workaround (may not be most efficient)
- Requires separate hardware for teacher

**Future Improvements:**
- [ ] Direct model forward pass for logprob computation (more efficient than vLLM generate)
- [ ] Support for multi-node teacher deployment
- [ ] Caching frequently requested sequences
- [ ] Compression of logprob tensors for network transfer
- [ ] Quantized teacher models (fp8/int8) for memory efficiency

## References

- Design Doc: `design/OPD+RL.md`
- VeRL GKD Recipe: `verl/recipe/gkd/`
- Configuration: `verl/verl/trainer/config/algorithm.py` (OPDConfig)
