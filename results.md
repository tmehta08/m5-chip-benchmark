# M5 Chip Benchmark Results

**Run:** 2026-05-02 00:26:11
**Chip:** Apple M5 Pro
**RAM:** 24 GB
**CPU cores:** 15 logical

## Results

| Benchmark | Time | Score |
|---|---|---|
| CPU single-core (prime sieve, n=10M) | 0.032s | — |
| CPU multi-core (sum-of-squares, 15 cores) | 1.105s | 1.5x speedup |
| NumPy matmul (float32, 4096×4096) | 0.163s | 844.2 GFLOPS (CPU) |
| Torch MPS matmul (float32, 4096×4096) | 0.096s | 1430.8 GFLOPS (GPU) |
| Memory bandwidth (float32, 1 GB copy) | 0.262s | 7.6 GB/s |
| Disk write (512 MB) | 0.075s | 7.15 GB/s |
| Disk read (512 MB) | 0.034s | 15.99 GB/s |
| Pandas groupby (10M rows, 200 groups) | 0.316s | — |
