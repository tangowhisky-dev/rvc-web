# GSD State

**Active Milestone:** M001 — COMPLETE + INFERENCE HARDENED
**Status:** All 4 slices done, all 8 requirements validated, realtime inference fixed (MPS + warmup)

## Milestone M001 Final Status

**COMPLETE.** All 4 slices done. All 8 requirements validated (R001–R008). M001-SUMMARY.md written.

- S01: 17/17 contract tests ✅
- S02: 8/8 contract tests + 1-epoch integration (55MB .pth in 24.5s) ✅
- S03: 10/10 contract tests; 35 passed / 2 skipped (integration skip-guarded without hardware) ✅
- S04: TypeScript clean; script syntax verified; browser UI confirmed ✅

## Realtime Inference Fixes (Post-M001)

**Segfault root causes fixed:**

1. **hubert_base.pt relative path**: fairseq resolved relative to uvicorn cwd, not rvc_root
   - Fix: set `hubert_path` env var; rtrvc.py reads `os.environ.get("hubert_path")`

2. **faiss + fairseq OpenMP collision**: both link different libomp on macOS
   - Fix: `faiss.omp_set_num_threads(1)` + `OMP_NUM_THREADS=1` before any imports

3. **Native code segfaulting uvicorn**: RVC/fairseq/sounddevice moved to isolated subprocess
   - Isolated worker: `_realtime_worker.py` (spawn context)
   - Parent holds queues + proc ref; native code never touches uvicorn address space

**MPS inference backend:**

- Device selection: prefers Apple Silicon MPS, falls back to CPU
- Model load: 1.4s; warmup: 2s; steady-state: 50ms/block (real-time compatible)
- `is_half=False` for MPS stability
- Warmup happens before "ready" signal — client knows inference is stable
- Structured JSON logging: `worker_warming_up`, `worker_ready` with device type

**Tests:** 35 passed, 2 skipped (integration skip-guarded)

## Blockers

None.

## Next Action

Ready for M002 planning. Run `./scripts/start.sh` for hardware UAT with MPS inference.


## RVC Realtime Inference Crash Fix
- Discovered that the root cause of the immediate worker crash on audio input was an OpenMP `libomp.dylib` clash combined with FAISS requiring contiguous memory on some environments.
- Applied `np.ascontiguousarray` monkey-patch to `rvc.index.search(npy)` to prevent segfaults when querying the RVC index.
- Restored missing `faiss` `import` in `_realtime_worker.py` and `omp_set_num_threads(1)` call to prevent macOS multithreading crash.
- Realtime WS now streams audio properly (tested on backend endpoint).
