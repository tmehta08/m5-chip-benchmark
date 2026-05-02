import os
import platform
import multiprocessing
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
import torch


def run(label, fn):
    print(f"  {label} ... ", end="", flush=True)
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    print(f"{elapsed:.3f}s")
    return elapsed, result


# ── CPU: single core ────────────────────────────────────────────────────────

def bench_cpu_single():
    n = 10_000_000
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return sum(sieve)


# ── CPU: all cores ───────────────────────────────────────────────────────────

def _worker(n):
    total = 0
    for i in range(n):
        total += i * i
    return total


def bench_cpu_multicore():
    cores = multiprocessing.cpu_count()
    n_per_core = 5_000_000

    # single-core baseline: same task, one worker
    t0 = time.perf_counter()
    _worker(n_per_core)
    t_single = time.perf_counter() - t0

    with multiprocessing.Pool(cores) as pool:
        results = pool.map(_worker, [n_per_core] * cores)

    return sum(results), cores, t_single


# ── NumPy: matrix multiply (CPU / Accelerate) ────────────────────────────────

def bench_matmul():
    size = 4096
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)
    C = A @ B
    return float(C.sum()), size


# ── Torch: matrix multiply (MPS / GPU) ──────────────────────────────────────

def bench_torch_mps():
    device = torch.device("mps")
    size = 4096
    A = torch.rand(size, size, dtype=torch.float32, device=device)
    B = torch.rand(size, size, dtype=torch.float32, device=device)
    # warmup
    _ = A @ B
    torch.mps.synchronize()

    start = time.perf_counter()
    C = A @ B
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed, size


# ── Memory bandwidth ─────────────────────────────────────────────────────────

def bench_memory_bandwidth():
    n = 250_000_000  # 1 GB of float32
    a = np.ones(n, dtype=np.float32)
    b = np.empty_like(a)
    np.copyto(b, a)
    return float(b.sum()), n


# ── Disk I/O ─────────────────────────────────────────────────────────────────

def bench_disk_io():
    size = 512 * 1024 * 1024  # 512 MB
    data = np.random.bytes(size)

    tmp = tempfile.NamedTemporaryFile(delete=False)
    path = tmp.name
    tmp.close()

    try:
        # write
        start = time.perf_counter()
        with open(path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        write_time = time.perf_counter() - start

        # read
        start = time.perf_counter()
        with open(path, "rb") as f:
            _ = f.read()
        read_time = time.perf_counter() - start
    finally:
        os.unlink(path)

    return write_time, read_time, size


# ── Pandas ───────────────────────────────────────────────────────────────────

def bench_pandas():
    n = 10_000_000
    df = pd.DataFrame({
        "sensor_id": np.random.randint(0, 200, n),
        "temperature": np.random.uniform(20.0, 80.0, n),
        "humidity": np.random.uniform(30.0, 90.0, n),
        "pressure": np.random.uniform(950.0, 1050.0, n),
    })
    result = df.groupby("sensor_id").agg(["mean", "min", "max", "std"])
    return result.shape


# ── System info ──────────────────────────────────────────────────────────────

def get_system_info():
    chip, ram = "Apple Silicon", 0
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True,
        ).stdout.strip() or chip
    except Exception:
        pass
    try:
        ram = int(subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True,
        ).stdout.strip()) // 1024 ** 3
    except Exception:
        pass
    return chip, ram


# ── Results ──────────────────────────────────────────────────────────────────

def save_results(chip, ram, cores, results):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = "\n".join(
        f"| {label} | {time:.3f}s | {score} |"
        for label, time, score in results
    )

    content = f"""# M5 Chip Benchmark Results

**Run:** {timestamp}
**Chip:** {chip}
**RAM:** {ram} GB
**CPU cores:** {cores} logical

## Results

| Benchmark | Time | Score |
|---|---|---|
{rows}
"""
    with open("results.md", "w") as f:
        f.write(content)
    print("  Results saved to results.md")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    chip, ram = get_system_info()
    cores = multiprocessing.cpu_count()

    print("=" * 60)
    print("SYSTEM")
    print("=" * 60)
    print(f"  Python:    {platform.python_version()}")
    print(f"  Machine:   {platform.machine()}")
    print(f"  Chip:      {chip}")
    print(f"  RAM:       {ram} GB")
    print(f"  CPU cores: {cores} logical")
    print(f"  PyTorch:   {torch.__version__}  (MPS: {torch.backends.mps.is_available()})")
    print()

    print("=" * 60)
    print("BENCHMARKS")
    print("=" * 60)

    t_cpu,  _              = run("CPU single-core  (prime sieve, n=10M)     ", bench_cpu_single)
    t_multi, (_, c, t_single) = run("CPU multi-core   (sum-of-squares, 15 cores)", bench_cpu_multicore)
    t_mm,   (_, mm_size)   = run("NumPy matmul     (float32, 4096×4096)     ", bench_matmul)
    t_mps,  (_, mps_size)  = run("Torch MPS matmul (float32, 4096×4096)     ", bench_torch_mps)
    t_bw,   (_, bw_n)      = run("Memory bandwidth (float32, 1 GB copy)     ", bench_memory_bandwidth)
    print(f"  {'Disk I/O         (512 MB write + read)      '} ... ", end="", flush=True)
    t_write, t_read, d_sz = bench_disk_io()
    print(f"{t_write + t_read:.3f}s")
    t_pd,   _              = run("Pandas groupby   (10M rows, 200 groups)   ", bench_pandas)

    cpu_gflops  = 2 * mm_size  ** 3 / t_mm  / 1e9
    mps_gflops  = 2 * mps_size ** 3 / t_mps / 1e9
    gb_s        = (bw_n * 4 * 2) / t_bw / 1e9
    write_gb_s  = d_sz / t_write / 1e9
    read_gb_s   = d_sz / t_read  / 1e9
    speedup     = t_single * c / t_multi

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  CPU single-core:   {t_cpu:.3f}s")
    print(f"  CPU multi-core:    {t_multi:.3f}s   →  {speedup:.1f}x speedup across {c} cores")
    print(f"  NumPy matmul:      {t_mm:.3f}s   →  {cpu_gflops:.1f} GFLOPS  (CPU/Accelerate)")
    print(f"  Torch MPS matmul:  {t_mps:.3f}s   →  {mps_gflops:.1f} GFLOPS  (GPU)")
    print(f"  Memory bandwidth:  {t_bw:.3f}s   →  {gb_s:.1f} GB/s")
    print(f"  Disk write:        {t_write:.3f}s  →  {write_gb_s:.2f} GB/s")
    print(f"  Disk read:         {t_read:.3f}s  →  {read_gb_s:.2f} GB/s")
    print(f"  Pandas groupby:    {t_pd:.3f}s")
    print()

    save_results(chip, ram, cores, [
        ("CPU single-core (prime sieve, n=10M)",      t_cpu,   "—"),
        (f"CPU multi-core (sum-of-squares, {c} cores)", t_multi, f"{speedup:.1f}x speedup"),
        ("NumPy matmul (float32, 4096×4096)",          t_mm,    f"{cpu_gflops:.1f} GFLOPS (CPU)"),
        ("Torch MPS matmul (float32, 4096×4096)",      t_mps,   f"{mps_gflops:.1f} GFLOPS (GPU)"),
        ("Memory bandwidth (float32, 1 GB copy)",      t_bw,    f"{gb_s:.1f} GB/s"),
        ("Disk write (512 MB)",                        t_write, f"{write_gb_s:.2f} GB/s"),
        ("Disk read (512 MB)",                         t_read,  f"{read_gb_s:.2f} GB/s"),
        ("Pandas groupby (10M rows, 200 groups)",      t_pd,    "—"),
    ])


if __name__ == "__main__":
    main()
