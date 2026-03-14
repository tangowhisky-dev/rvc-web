# S02: Training Pipeline with Live Log Streaming — UAT

**Milestone:** M001
**Written:** 2026-03-14

## UAT Type

- UAT mode: artifact-driven
- Why this mode is sufficient: All acceptance criteria are machine-verifiable — file existence, DB state, WebSocket message shape, and HTTP response codes. No human audio judgment is required for the training pipeline itself. The integration smoke test exercises the complete real subprocess chain and asserts all outcomes programmatically.

## Preconditions

- `rvc` conda environment installed with all dependencies (aiosqlite, aiofiles, faiss-cpu, scipy, pytest-timeout)
- `Retrieval-based-Voice-Conversion-WebUI/` present at repo root with pretrained models in `assets/pretrained_v2/`
- `RVC_ROOT` env var set to the absolute path of `Retrieval-based-Voice-Conversion-WebUI/`
- Backend not running a training job (in-memory state is fresh on server start)
- For contract tests only: no additional setup needed (mocked subprocesses)

## Smoke Test

```bash
# Confirm .pth on disk (produced by the integration run)
ls -lh Retrieval-based-Voice-Conversion-WebUI/assets/weights/rvc_finetune_active.pth
# Expected: file ~55MB, present

# Confirm all contract tests pass
conda run -n rvc pytest backend/tests/test_training.py -v
# Expected: 8 passed
```

## Test Cases

### 1. Contract tests — full training API surface

```bash
conda run -n rvc pytest backend/tests/test_training.py -v
```

1. Run the above command from `rvc-web/`
2. **Expected:** 8 passed, 0 failed, in < 1 second. Tests cover: POST /start (200+job_id), POST /start unknown (404), POST /start no sample (400), POST /start double (409), POST /cancel (200), GET /status hit (200+phase/progress/status), GET /status miss (404), WS connect receives {type,message,phase} messages ending with type=="done".

### 2. Full test suite regression

```bash
conda run -n rvc pytest backend/tests/ -v
```

1. Run from `rvc-web/`
2. **Expected:** 25 passed, 1 skipped (integration test auto-skips without RVC_ROOT if not set), 0 failed

### 3. Integration smoke test — real 1-epoch training run

```bash
RVC_ROOT=$(pwd)/Retrieval-based-Voice-Conversion-WebUI conda run -n rvc pytest backend/tests/test_training_integration.py -v -s -m integration --timeout=1800
```

1. Run from `rvc-web/`
2. Watch stdout for phase transitions: preprocess → extract_f0 → extract_feature → train → index
3. **Expected:** 1 passed; elapsed ~20–60 seconds; WebSocket collected ≥1 log message; final message type=="done"

### 4. Artifact verification after integration run

```bash
# .pth file
ls -lh Retrieval-based-Voice-Conversion-WebUI/assets/weights/rvc_finetune_active.pth

# FAISS index
ls Retrieval-based-Voice-Conversion-WebUI/logs/rvc_finetune_active/added_IVF*_Flat*.index

# DB status
sqlite3 backend/data/rvc.db "SELECT id, name, status FROM profiles WHERE status='trained';"
```

1. Run all three commands after integration test completes
2. **Expected:** `.pth` ~55MB present; FAISS index present; at least one row returned from SQLite query

### 5. Status endpoint inspection

```bash
# Requires server running: conda run -n rvc uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
curl -s http://localhost:8000/api/training/status/<profile_id> | python3 -m json.tool
```

1. Start backend server
2. Note a profile_id from SQLite (use the `trained` profile from test case 4)
3. Hit the status endpoint
4. **Expected:** JSON `{"phase":"index","progress_pct":100,"status":"done","error":null}` (or similar post-completion state)

### 6. Health check includes RVC_ROOT

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

1. **Expected:** `{"status":"ok","db_path":"...","rvc_root":"..."}` — `rvc_root` field present

## Edge Cases

### Double-start rejection

```bash
# Simulate: POST /api/training/start twice rapidly
# Contract test test_start_training_conflict covers this
```

1. Run `conda run -n rvc pytest backend/tests/test_training.py::test_start_training_conflict -v`
2. **Expected:** PASSED — second POST returns 409 with {"detail":"job already running"}

### Unknown profile start

```bash
curl -s -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{"profile_id":"nonexistent-id"}' | python3 -m json.tool
```

1. **Expected:** 404 `{"detail":"Profile not found"}`

### Cancel with no active job

```bash
curl -s -X POST http://localhost:8000/api/training/cancel \
  -H "Content-Type: application/json" \
  -d '{"profile_id":"some-id"}' | python3 -m json.tool
```

1. **Expected:** 404 `{"detail":"No active training job for this profile"}`

### Status for unknown profile

```bash
curl -s http://localhost:8000/api/training/status/unknown-id
```

1. **Expected:** 404 `{"detail":"No training job found"}`

## Failure Signals

- Contract tests fail → check router registration in `main.py`; confirm `backend.app.routers.training` module imports cleanly
- Integration test: `{type:"error"}` message in WebSocket stream → inspect message field for subprocess error; check `RVC_ROOT` is set; check pretrained models exist in `assets/pretrained_v2/`
- `.pth` not produced after "done" message → check `rmvpe_root` is in train subprocess env (D024); check `process_ckpt.py` patch is in place (D025)
- `train.log` empty after run → expected on 1-epoch runs (os._exit(0) races Python logging flush); confirm `G_latest.pth` exists in `logs/rvc_finetune_active/` as the real completion signal
- DB status stuck at `training` after server restart → in-memory job state lost; status was not updated to `failed` before restart; manually reset: `sqlite3 backend/data/rvc.db "UPDATE profiles SET status='failed' WHERE status='training';"`

## Requirements Proved By This UAT

- R002 (Training / Fine-tuning Pipeline) — proved by integration smoke test: real preprocess → feature extract → train → FAISS pipeline runs end-to-end; `.pth` produced (55MB); WebSocket streams live logs; DB status updated to `trained`; cancel endpoint exists and is contract-tested; phase/status inspection surface live at GET /api/training/status/{profile_id}

## Not Proven By This UAT

- R003 (Realtime Voice Conversion) — not tested; requires S03
- R004 (Live Waveform Visualization) — not tested; requires S03
- R005 (Audio Device Selection) — verified at API level in S01; not tested end-to-end with training output
- R001 (Voice Sample Library) — already validated in S01; integration test exercises upload path but not exhaustively
- Human audio quality judgment — whether the fine-tuned model produces audible voice cloning is not verified by this UAT; that is the domain of S03's realtime VC session

## Notes for Tester

- The integration test creates a real profile in `backend/data/rvc.db` and cleans it up via `DELETE /api/profiles/{id}` in fixture teardown. If the test crashes mid-run, a dangling profile may remain — safe to delete manually.
- `train.log` being empty (0 bytes) after a 1-epoch run is expected and harmless. Check `G_latest.pth` and `D_latest.pth` in `logs/rvc_finetune_active/` as the real "train.py ran to completion" signal.
- The FAISS index filename includes the IVF size which depends on sample count. For a 5-second sine wave the filename is `added_IVF6_Flat_nprobe_1_rvc_finetune_active_v2.index` — for a real voice sample the IVF size will differ.
- The `model_hash` and `id` fields in the saved `.pth` are empty strings (PyTorch 2.6 / fairseq incompatibility workaround, D025). This does not affect inference quality.
