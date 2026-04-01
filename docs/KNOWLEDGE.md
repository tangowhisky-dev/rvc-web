
---

## GPU Memory Management

**Rule: F0 extraction must never fork multiple GPU-bound processes — it exhausts VRAM with zero throughput gain.**

`extract_f0_print.py` spawns `n_p` child processes via `multiprocessing.Process`.
On Linux with `fork`, each child inherits the parent's CUDA context — all children share copies of the GPU-allocated RMVPE model (~600 MiB each).

**Before fix:** `n_p = multiprocessing.cpu_count()` on an 80-core server → 80 processes × 600 MiB = ~48 GiB of VRAM exhausted before training even starts, all contending on the same GPU with zero parallelism gain (GPU-bound workload).

**After fix:**
- GPU devices (cuda / mps): `n_p = 1` — single process, no forking, one RMVPE instance.
- CPU device: `n_p = min(cpu_count(), 8)` — CPU-bound, genuine parallelism benefit.

Files changed:
- `backend/rvc/infer/modules/train/extract_f0_print.py`: Check `device` and set `effective_n_p = 1` for GPU, skip forking.
- `backend/app/training.py`: Pass `_f0_n_proc = 1` for GPU instead of `n_cpu`.

**Preprocess is unaffected** — it's pure CPU (audio loading + resampling), parallelism up to 8 is beneficial.

---
