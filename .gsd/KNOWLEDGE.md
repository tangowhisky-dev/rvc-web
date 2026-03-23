# KNOWLEDGE — rvc-web

Append-only register of project-specific rules, patterns, and lessons learned.
Read at the start of every unit. Never remove or edit existing entries.

---

## Service Management

**Rule: Always use `scripts/start.sh` and `scripts/stop.sh` to start and stop services.**

Never start the backend or frontend manually (no raw `uvicorn`, `conda run`, `pnpm dev`, or `bg_shell` commands).
Never kill services manually with `lsof | kill` or `pkill` — use `scripts/stop.sh` instead.

Reasons:
- `start.sh` exports required env vars (`RVC_ROOT`, `KMP_DUPLICATE_LIB_OK`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`) that the backend and training pipeline depend on.
- `start.sh` kills stale ports and dangling training/inference processes before starting.
- `stop.sh` shuts down worker subprocesses (`spawn_main`, `_realtime_worker`) before killing the backend, preventing orphaned GPU processes.
- `start.sh` writes PIDs to `.pids/` so `stop.sh` can do a clean shutdown.

Usage:
```bash
scripts/start.sh   # start backend + frontend
scripts/stop.sh    # stop everything cleanly
```

---

## Path Handling

**Rule: Always convert `profile_dir` and `sample_dir` to absolute paths before passing to subprocesses.**

RVC training subprocesses (`preprocess.py`, `extract_f0_print.py`, `extract_feature_print.py`, `train.py`) run with `cwd=rvc_root`. Any relative path passed to them resolves against `rvc_root`, not the project root, causing `FileNotFoundError`.

Fix: call `os.path.abspath()` on `profile_dir` and `audio_dir` in `routers/training.py` immediately after construction.

---

## Audio Preprocessing

**Rule: Always pass `format="WAV"` explicitly to `soundfile.write()` when the output path has a non-standard extension.**

`soundfile` infers format from the file extension. Temp files like `moeed.wav.cleaning` have an unrecognised extension and cause `TypeError: No format specified`. Always pass `format="WAV"` (or the appropriate format) explicitly.

---

## Training Workspace

**Rule: The shared `logs/rvc_finetune_active/` directory is NOT profile-scoped — it must be treated as a temp workspace.**

When a profile is deleted, `routers/profiles.py` wipes `exp_dir` (`logs/rvc_finetune_active/`) entirely.
When training starts for a new profile, `training.py` checks a sentinel file (`.profile_id`) and wipes the workspace if the profile ID changed.
A fresh profile (no own checkpoints) must never inherit `G_latest.pth` / `D_latest.pth` from a prior profile — `_KEEP_FILES` is empty when `seeded_checkpoint=False`.

---

## Epoch Loss Tracking

**Rule: `epoch_losses` rows must be deleted explicitly on profile delete** — `aiosqlite` does not enforce `ON DELETE CASCADE` by default.

The delete handler in `routers/profiles.py` explicitly runs:
```sql
DELETE FROM epoch_losses WHERE profile_id = ?
DELETE FROM audio_files WHERE profile_id = ?
DELETE FROM profiles WHERE id = ?
```
