---
estimated_steps: 5
estimated_files: 2
---

# T04: Write and Run Integration Smoke Test (1 Epoch)

**Slice:** S02 — Training Pipeline with Live Log Streaming
**Milestone:** M001

## Description

Write `backend/tests/test_training_integration.py` and run it against a real RVC training pipeline with `epochs=1, save_every=1`. This proves:
1. The full subprocess chain executes without crashing on macOS Apple Silicon.
2. The log streaming architecture works end-to-end (messages flow through WebSocket).
3. `assets/weights/rvc_finetune_active.pth` is produced on disk.
4. Profile status is updated to `"trained"` in SQLite.

This test retires the "MPS training stability" risk from the M001 roadmap. It is marked `@pytest.mark.integration` so it is skipped in normal CI runs and only executed when explicitly invoked.

If the test fails due to subprocess errors, this task includes the fix-and-rerun loop: diagnose the error from the WebSocket stream or log file, fix the relevant code in `training.py`, and re-run until it passes.

## Steps

1. Write `backend/tests/test_training_integration.py`:
   - Mark entire module with `pytestmark = pytest.mark.integration`
   - `autouse` skip fixture: `pytest.skip` if `RVC_ROOT` not set in environment
   - Fixture `real_wav_path`: generates a 5-second mono 48kHz WAV file using `numpy` + `scipy.io.wavfile` (440Hz sine wave), saves to a temp file, returns the path. (Alternatively, search for an existing wav in the test assets first.)
   - Fixture `trained_profile`: creates a real profile via HTTP POST to the running app with the real WAV. Uses `TestClient` synchronously (faster for integration setup). Returns `profile_id`.
   - Fixture `start_url` / `ws_url`: derives endpoint URLs from app base.

2. Test `test_full_training_pipeline`:
   - POST `/api/training/start` with `{profile_id, epochs: 1, save_every: 1}`
   - Assert 200 response with `job_id`
   - Connect to `ws://localhost:8000/ws/training/<profile_id>` using `TestClient.websocket_connect()`
   - Loop receiving messages with a 30-minute overall timeout (`time.time()` check on each iteration)
   - Collect all messages; break on `type=="done"` or `type=="error"`
   - Assertions after loop:
     - `assert any(m["type"] == "log" for m in messages)` — at least one log message received
     - `assert messages[-1]["type"] == "done"` — final message is done (not error)
     - `assert (Path(os.environ["RVC_ROOT"]) / "assets/weights/rvc_finetune_active.pth").exists()`
   - DB assertion: open aiosqlite directly and check `SELECT status FROM profiles WHERE id = ?` returns `"trained"`

3. Run the test and observe:
   ```bash
   RVC_ROOT=/Users/tango16/code/rvc-web/Retrieval-based-Voice-Conversion-WebUI \
   conda run -n rvc pytest backend/tests/test_training_integration.py \
     -v -s -m integration --timeout=1800 2>&1 | tee /tmp/integration-test.log
   ```

4. If any subprocess phase fails, diagnose via the error message in the WebSocket stream and `train.log`:
   - Phase 1/2/3 failures: stderr captured in WebSocket stream — check the `{type:"error"}` message text
   - Phase 4 failures: check `Retrieval-based-Voice-Conversion-WebUI/logs/rvc_finetune_active/train.log`
   - Fix the relevant subprocess command or env in `training.py`, then re-run

5. After test passes, run the full suite to confirm no regressions:
   ```bash
   conda run -n rvc pytest backend/tests/test_training.py backend/tests/test_profiles.py backend/tests/test_devices.py -v
   ```

## Must-Haves

- [ ] Integration test file created with `pytestmark = pytest.mark.integration`
- [ ] Test is auto-skipped if `RVC_ROOT` env var is not set
- [ ] Real 48kHz WAV file generated (or found) for the test profile
- [ ] Test collects all WebSocket messages until `done` or `error`
- [ ] Asserts: at least one log message, final message is `done`, `.pth` file exists, DB status is `trained`
- [ ] Test passes with the real RVC subprocess chain (fix any errors found)
- [ ] `Retrieval-based-Voice-Conversion-WebUI/assets/weights/rvc_finetune_active.pth` verified on disk after run

## Verification

```bash
# Confirm .pth exists
ls -lh Retrieval-based-Voice-Conversion-WebUI/assets/weights/rvc_finetune_active.pth

# Confirm DB status
sqlite3 backend/data/rvc.db "SELECT id, name, status FROM profiles WHERE status='trained';"

# Confirm FAISS index exists
ls Retrieval-based-Voice-Conversion-WebUI/logs/rvc_finetune_active/added_IVF*.index

# Confirm no regressions in contract tests
conda run -n rvc pytest backend/tests/test_training.py backend/tests/test_profiles.py backend/tests/test_devices.py -v
```

## Observability Impact

- Signals added/changed:
  - `Retrieval-based-Voice-Conversion-WebUI/logs/rvc_finetune_active/train.log` — persists after test; inspectable with `tail`
  - `Retrieval-based-Voice-Conversion-WebUI/assets/weights/rvc_finetune_active.pth` — presence = proof
  - `sqlite3 backend/data/rvc.db "SELECT status FROM profiles"` — DB status = durable state
  - `/tmp/integration-test.log` — captured test output for post-mortem if test is re-run
- How a future agent inspects this: `ls assets/weights/rvc_finetune_active.pth` + DB query above; `GET /api/training/status/<profile_id>` if server is running
- Failure state exposed: If test fails, `{type:"error"}` message in WebSocket stream contains last subprocess error; `train.log` contains full training crash traceback

## Inputs

- `backend/app/training.py` (T02) — full pipeline implementation
- `backend/app/routers/training.py` (T03) — REST + WebSocket endpoints
- `S02-RESEARCH.md` — subprocess command reference for diagnosing failures
- `Retrieval-based-Voice-Conversion-WebUI/web.py` — reference implementation for comparison when debugging

## Expected Output

- `backend/tests/test_training_integration.py` — integration test, auto-skipped when `RVC_ROOT` unset
- `Retrieval-based-Voice-Conversion-WebUI/assets/weights/rvc_finetune_active.pth` — real fine-tuned model file on disk
- `Retrieval-based-Voice-Conversion-WebUI/logs/rvc_finetune_active/added_IVF*.index` — FAISS index file on disk
- All contract tests still passing (25/25 green)
