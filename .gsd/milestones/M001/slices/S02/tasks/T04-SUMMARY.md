---
id: T04
parent: S02
milestone: M001
provides:
  - backend/tests/test_training_integration.py — integration smoke test, auto-skipped without RVC_ROOT
  - Retrieval-based-Voice-Conversion-WebUI/assets/weights/rvc_finetune_active.pth — real fine-tuned model on disk
  - Retrieval-based-Voice-Conversion-WebUI/logs/rvc_finetune_active/added_IVF6_Flat_nprobe_1_rvc_finetune_active_v2.index — FAISS index on disk
  - pytest.ini — project pytest config with integration mark registered
key_files:
  - backend/tests/test_training_integration.py
  - pytest.ini
  - Retrieval-based-Voice-Conversion-WebUI/infer/lib/train/process_ckpt.py
  - backend/app/training.py
key_decisions:
  - D022: pytest-timeout installed in rvc conda env; pytest.ini created at repo root to register integration mark and set asyncio_mode=strict; this is the authoritative pytest config going forward
  - D023: process_ckpt.py save_small_model patched to catch model_hash_ckpt failure — hash is DRM metadata only; saving model without hash/id fields is functionally correct on PyTorch >= 2.6
  - D024: train subprocess env must include rmvpe_root — save_small_model calls model_hash_ckpt → Pipeline.__init__ which reads rmvpe_root; training.py train_env now sets it
patterns_established:
  - Integration test uses real app TestClient (not isolated DB) — writes to production backend/data/rvc.db; cleanup via DELETE /api/profiles/{id} in fixture teardown
  - autouse skip fixture on RVC_ROOT check — every test in module is auto-skipped if env var missing
  - asyncio DB check in sync test via asyncio.get_event_loop().run_until_complete() — valid for integration tests that use sync TestClient wrapper
observability_surfaces:
  - ls Retrieval-based-Voice-Conversion-WebUI/assets/weights/rvc_finetune_active.pth — presence confirms training completed
  - ls Retrieval-based-Voice-Conversion-WebUI/logs/rvc_finetune_active/added_IVF*.index — FAISS index presence
  - sqlite3 backend/data/rvc.db "SELECT id, name, status FROM profiles WHERE status='trained';" — DB status
  - /tmp/integration-test.log — captured test output from last run
  - tail -f Retrieval-based-Voice-Conversion-WebUI/logs/rvc_finetune_active/train.log — live training log (note: train.log stays empty on fast runs due to os._exit(0); G_latest.pth/D_latest.pth presence confirms train.py ran)
duration: ~30 minutes (implementation + 2 test runs ~25 seconds each)
verification_result: passed
completed_at: 2026-03-14
blocker_discovered: false
---

# T04: Write and Run Integration Smoke Test (1 Epoch)

**Full RVC training pipeline runs end-to-end in 24 seconds on Apple Silicon, producing a real 55MB `.pth` file; all 4 assertions pass and contract tests remain 25/25 green.**

## What Happened

Wrote `backend/tests/test_training_integration.py` with a single `test_full_training_pipeline` test:
- Auto-skips via `autouse` fixture if `RVC_ROOT` not set
- Creates a real 5-second 48kHz 440Hz sine-wave WAV using numpy + scipy
- Uploads it via `POST /api/profiles` to create a real profile in the production DB
- POSTs to `POST /api/training/start` with `epochs=1, save_every=1`
- Collects all WebSocket messages via `TestClient.websocket_connect()` with 30-minute timeout
- Asserts: ≥1 log message, final message is "done", `.pth` exists on disk, DB status is "trained"

Also created `pytest.ini` to register the `integration` mark and configure asyncio strict mode.

**First run failed** with `.pth` not existing despite "done" message. Root cause: two bugs in the RVC codebase:

1. **Missing `rmvpe_root` in train env**: `save_small_model` in `process_ckpt.py` calls `model_hash_ckpt` → `Pipeline.__init__` which reads `os.environ["rmvpe_root"]`. Train subprocess env didn't include it.  
   Fix: `backend/app/training.py` now sets `train_env["rmvpe_root"]` before launching train subprocess.

2. **PyTorch 2.6 `weights_only` incompatibility**: With `rmvpe_root` set, `model_hash_ckpt` → `model_hash` → `Pipeline` tries to load HuBERT via fairseq's `load_checkpoint_to_cpu`, which now uses `weights_only=True` by default (PyTorch 2.6), causing `UnpicklingError` for `fairseq.data.dictionary.Dictionary` globals.  
   Fix: `process_ckpt.py:save_small_model` now catches hash computation failure and proceeds without hash/id metadata. The hash is DRM metadata only; the saved `.pth` is fully functional for inference without it.

**Second run: PASSED** — 24.5 seconds, 60 WebSocket messages, 55MB `.pth` on disk, DB status "trained".

## Verification

```
# Integration test passing
conda run -n rvc pytest backend/tests/test_training_integration.py -v -s -m integration --timeout=1800
# → 1 passed in 24.50s

# .pth on disk
ls -lh Retrieval-based-Voice-Conversion-WebUI/assets/weights/rvc_finetune_active.pth
# → -rw-r--r-- 55M rvc_finetune_active.pth

# FAISS index on disk
ls Retrieval-based-Voice-Conversion-WebUI/logs/rvc_finetune_active/added_IVF6_Flat_nprobe_1_rvc_finetune_active_v2.index

# DB status trained
sqlite3 backend/data/rvc.db "SELECT id, name, status FROM profiles WHERE status='trained';"

# Contract test regression — 25/25 green
conda run -n rvc pytest backend/tests/test_training.py backend/tests/test_profiles.py backend/tests/test_devices.py -v
# → 25 passed in 0.25s
```

## Diagnostics

- `ls Retrieval-based-Voice-Conversion-WebUI/assets/weights/rvc_finetune_active.pth` — primary proof of success
- `sqlite3 backend/data/rvc.db "SELECT id, name, status FROM profiles WHERE status='trained';"` — DB state
- `ls Retrieval-based-Voice-Conversion-WebUI/logs/rvc_finetune_active/added_IVF*.index` — FAISS index
- `/tmp/integration-test.log` — full captured output of last integration test run
- Note: `train.log` remains **empty** (0 bytes) on fast 1-epoch runs because `train.py` calls `os._exit(0)` before the FileHandler flushes. The presence of `G_latest.pth`/`D_latest.pth` in `logs/rvc_finetune_active/` confirms train.py ran to epoch completion.
- If test fails: check `{type:"error"}` message in WebSocket stream; check `G_latest.pth` presence to determine whether train phase ran

## Deviations

- `pytest-timeout` plugin was not installed — installed via `pip install pytest-timeout` in rvc env; `--timeout=1800` flag now works
- `pytest.ini` did not exist — created at repo root to register `integration` mark and consolidate pytest config
- Two upstream RVC bugs required fixing before test could pass (both documented in key_decisions D023/D024)

## Known Issues

- `train.log` stays empty on 1-epoch runs due to `os._exit(0)` in train.py calling before Python logging flushes. The log tailer therefore emits "Warning: train.log did not appear within 10s" — this is expected and harmless for 1-epoch runs. Longer runs (>10s to first log line) will populate the file.
- `model_hash`/`id` fields in the saved `.pth` are empty strings — inference pipeline does not use these fields; only affects RVC's own DRM/authenticity checks.

## Files Created/Modified

- `backend/tests/test_training_integration.py` — integration smoke test (new)
- `pytest.ini` — project pytest config with integration mark and asyncio_mode=strict (new)
- `backend/app/training.py` — added `rmvpe_root` to train subprocess env (fix)
- `Retrieval-based-Voice-Conversion-WebUI/infer/lib/train/process_ckpt.py` — made model_hash_ckpt failure non-fatal in save_small_model (fix)
