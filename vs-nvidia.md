# Apple M5 Pro vs NVIDIA GPUs

## Our Benchmark Results

| Metric | M5 Pro |
|---|---|
| GPU matmul (float32, 4096×4096) | 1.43 TFLOPS |
| CPU matmul (float32, 4096×4096) | 844 GFLOPS |
| Memory bandwidth (measured) | 7–12 GB/s |
| Disk read / write | 16 / 7 GB/s |
| Total power draw (full load) | ~30W |

---

## GPU Compute: Raw TFLOPS Comparison

| GPU | float32 TFLOPS | vs M5 Pro |
|---|---|---|
| RTX 4090 | ~40–80 TFLOPS | ~30–55x faster |
| RTX 4080 Super | ~25–40 TFLOPS | ~18–28x faster |
| RTX 3080 | ~15–20 TFLOPS | ~10–14x faster |
| RTX 3060 | ~8–12 TFLOPS | ~6–8x faster |
| RTX 3050 (laptop) | ~4–6 TFLOPS | ~3–4x faster |
| **Apple M5 Pro** | **~1.4 TFLOPS** | — |

> Note: TFLOPS figures are real-world PyTorch matmul measurements, not theoretical peaks. Theoretical peaks are higher for all cards.

---

## Where NVIDIA Wins

**Raw compute.** An RTX 4090 is 30–55x faster at float32 matrix math. For training large models or running batch inference at scale, there is no comparison.

**Ecosystem.** CUDA has been the standard for over a decade. cuDNN, cuBLAS, Triton, FlashAttention, and most ML frameworks are built and optimized for CUDA first. MPS (Apple's GPU backend) is still catching up — some operations fall back to CPU, and not all PyTorch features are supported.

**Multi-GPU scaling.** NVIDIA cards support NVLink for multi-GPU setups. Apple Silicon has no equivalent.

**Memory for large models.** An RTX 4090 has 24 GB GDDR6X. A single H100 has 80 GB HBM3. For training or fine-tuning large models, dedicated VRAM wins.

---

## Where Apple M5 Pro Wins

**Unified memory.** The GPU and CPU share the same 24 GB pool — there is no separate VRAM. An RTX 3080 has only 10 GB VRAM, meaning models larger than ~7B parameters at float16 won't fit. On the M5 Pro, the full 24 GB is available to the GPU. This is a meaningful advantage for local LLM inference.

**Power efficiency.** The entire M5 Pro system draws ~30W under load. An RTX 3080 alone has a 320W TDP. Per TFLOP, Apple Silicon is significantly more power efficient — closer to NVIDIA's data center chips (H100, A100) in FLOPS/Watt than consumer CUDA cards.

**No PCIe bottleneck.** On a discrete GPU setup, data has to travel over PCIe between CPU RAM and VRAM. On M5 Pro, the CPU and GPU access the same memory directly — no transfer overhead.

**Form factor.** This performance runs silently in a laptop with no external power brick.

---

## Practical Takeaway

| Use case | Better option |
|---|---|
| Training large models | NVIDIA (by a large margin) |
| Fine-tuning (LoRA, QLoRA) | NVIDIA (more VRAM options) |
| Local LLM inference (7B–13B) | M5 Pro (unified memory, no VRAM limit) |
| ML research / experimentation | NVIDIA (mature ecosystem) |
| On-the-go / battery-powered ML | M5 Pro (unmatched efficiency) |
| Production inference at scale | NVIDIA (cloud, multi-GPU) |

The M5 Pro is not trying to compete with NVIDIA on raw compute. It is a power-efficient, memory-flexible chip that punches above its weight for **local inference and development** — especially for models that would exhaust the VRAM on a mid-range NVIDIA card.
