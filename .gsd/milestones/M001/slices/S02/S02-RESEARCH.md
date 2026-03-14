# S02: Training Pipeline with Live Log Streaming — Research

**Date:** 2026-03-14

## Summary

S02 builds a multi-phase training pipeline that wraps four RVC subprocesses (preprocess → F0 extraction → feature extraction → train) and streams their log output in real time to a browser WebSocket client. All prerequisite Python packages (`asyncio`, `aiosqlite`, `websockets`, FastAPI WebSocket) are already installed and confirmed working in the `rvc` conda env. The training subprocess chain must run from `RVC_ROOT` as CWD because all internal paths in the RVC scripts are relative to that directory.

The trickiest architectural problem is log streaming: preprocess and feature extraction scripts print to stdout (easy to capture via asyncio subprocess pipes), but `train.py` writes exclusively to a log file (`logs/<exp>/train.log`) via a Python `FileHandler` — it has no stdout output. This means the streaming implementation needs two different strategies: stdout capture for phases 1–2, and async log-file tailing for phase 3. A single `asyncio.Queue` per training job bridges both strategies to the WebSocket sender.

The training device picture is clearer than expected: on Apple Silicon with no CUDA, `train.py` runs with `gloo` DDP backend and never calls `.cuda()` on models — they stay on CPU. `fp16_run` is already set to `false` in `configs/inuse/v2/48k.json`, so `GradScaler(enabled=False)` is a no-op. This means training runs on CPU and will be slow (20–60 min for a short sample) but should be stable. The D010 constraint (try MPS, fallback to CPU) effectively resolves to CPU-only for training because the upstream code doesn't move models to MPS.

## Recommendation

Use a single `TrainingManager` singleton (in-memory dict, one active job at a time) that chains four async steps. For phases 1–3 use `asyncio.create_subprocess_exec` with `stdout=PIPE, stderr=STDOUT` and stream lines into an `asyncio.Queue`. For phase 4 (train.py) start the subprocess but tail the log file asynchronously using a polling loop (read new bytes every 200ms). After training completes, run the FAISS index build in-process (Python, no subprocess) since `train_index()` in `web.py` is Python logic that calls `faiss` directly. Use `exp_name = "rvc_finetune_active"` as the fixed experiment directory name so the model lands at `assets/weights/rvc_finetune_active.pth` automatically via `save_small_model()`.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Subprocess log streaming | `asyncio.create_subprocess_exec` with `stdout=PIPE` | Built-in, non-blocking, integrates cleanly with asyncio queue |
| Log file tailing | `asyncio` + `aiofiles` polling loop | No external dep; simple and reliable for single-file tail |
| FAISS index build | `train_index()` logic from `web.py` (adapt in-process) | Already written, tested, correct; just needs async wrapper |
| WebSocket endpoint | FastAPI `WebSocket` (already in starlette) | Already installed; no extra dep |
| Test WebSocket | `starlette.testclient.TestClient.websocket_connect()` | Already in env; synchronous API works cleanly in tests |

## Existing Code and Patterns

- `Retrieval-based-Voice-Conversion-WebUI/web.py` — `preprocess_dataset()`, `extract_f0_feature()`, `click_train()`, `train_index()` are the authoritative reference for all subprocess commands, arguments, and log-file paths. Read before building. Don't copy these functions wholesale — adapt the subprocess commands.
- `Retrieval-based-Voice-Conversion-WebUI/infer/modules/train/preprocess.py` — Takes `inp_root sr n_p exp_dir noparallel per` as positional args. Prints to **stdout** AND writes `logs/<exp>/preprocess.log`. CWD must be `RVC_ROOT`.
- `Retrieval-based-Voice-Conversion-WebUI/infer/modules/train/extract_f0_print.py` — Takes `exp_dir_abs n_p f0method device is_half`. Prints to **stdout** AND writes `logs/<exp>/extract_f0_feature.log`. Needs `rmvpe_root` env var pointing to `assets/rmvpe`. CWD = `RVC_ROOT`.
- `Retrieval-based-Voice-Conversion-WebUI/infer/modules/train/extract_feature_print.py` — Takes `device n_part i_part i_gpu exp_dir_abs version is_half`. Prints to **stdout** AND writes to `logs/<exp>/extract_f0_feature.log`. Sets `PYTORCH_ENABLE_MPS_FALLBACK=1`. CWD = `RVC_ROOT`.
- `Retrieval-based-Voice-Conversion-WebUI/infer/modules/train/train.py` — Takes 13 CLI args (see `get_hparams()`). Writes logs **only to file** (`logs/<exp>/train.log`) via `FileHandler`. **No stdout output.** Must run CWD = `RVC_ROOT`. Saves final model to `assets/weights/<exp_name>.pth` via `save_small_model()`.
- `Retrieval-based-Voice-Conversion-WebUI/infer/lib/train/utils.py:312` — `get_logger()` writes to `logs/<exp>/train.log` only.
- `Retrieval-based-Voice-Conversion-WebUI/infer/lib/train/process_ckpt.py:15` — `save_small_model()` saves to `assets/weights/<name>.pth`. The `name` arg = `hps.name` = `experiment_dir` argument passed to `train.py` via `-e`.
- `backend/app/db.py` — `get_db()` context manager pattern to follow for all DB updates in training router.
- `backend/tests/conftest.py` — `client` fixture with `monkeypatch` isolation; WebSocket tests use `starlette.testclient.TestClient` (sync) alongside the async `httpx` client.

## Subprocess Command Reference

All commands run with `cwd=RVC_ROOT` and `PYTHONPATH` implicitly set via `cwd`.

**Phase 1 — Preprocess:**
```
python3 infer/modules/train/preprocess.py \
  "<inp_root>" 48000 <n_cpu> "logs/rvc_finetune_active" False 3.0
```
- `inp_root` = `os.path.dirname(profile.sample_path)` (absolute path, e.g. `<rvc_web>/data/samples/<id>`)
- `n_cpu` = `multiprocessing.cpu_count()` (16 on this machine)
- Outputs to: `logs/rvc_finetune_active/preprocess.log` + stdout

**Phase 2a — F0 Extraction:**
```
python3 infer/modules/train/extract_f0_print.py \
  "<RVC_ROOT>/logs/rvc_finetune_active" <n_cpu> rmvpe mps False
```
- `exp_dir` must be absolute path
- Env: `rmvpe_root=<RVC_ROOT>/assets/rmvpe`
- Outputs to: `logs/rvc_finetune_active/extract_f0_feature.log` + stdout

**Phase 2b — Feature Extraction (HuBERT):**
```
python3 infer/modules/train/extract_feature_print.py \
  mps 1 0 "" "<RVC_ROOT>/logs/rvc_finetune_active" v2 False
```
- 8-arg form (matches web.py pattern); `i_gpu=""` with `CUDA_VISIBLE_DEVICES`
- Env: `PYTORCH_ENABLE_MPS_FALLBACK=1` (script sets this itself)
- Outputs to: `logs/rvc_finetune_active/extract_f0_feature.log` + stdout

**Phase 3 — Train (pre-step: write files):**
Before launching `train.py`, the backend must:
1. Create `logs/rvc_finetune_active/` (already done by preprocess)
2. Write `logs/rvc_finetune_active/config.json` from `configs/inuse/v2/48k.json` (if not already present)
3. Build `logs/rvc_finetune_active/filelist.txt` from the feature directories (see `click_train()` in `web.py` — exactly the same logic)

```
python3 infer/modules/train/train.py \
  -e "rvc_finetune_active" -sr 48k -f0 1 -bs 4 -te 200 -se 10 \
  -pg "assets/pretrained_v2/f0G48k.pth" -pd "assets/pretrained_v2/f0D48k.pth" \
  -l 1 -c 0 -sw 0 -v v2 -a ""
```
- `-te 200` (200 epochs) = reasonable starting point; expose as configurable
- `-bs 4` = default from config; safe for CPU training
- `-l 1` = save only latest checkpoint (smaller disk footprint)
- Logs to file only: `logs/rvc_finetune_active/train.log`
- Final model: `assets/weights/rvc_finetune_active.pth`

**Phase 4 — FAISS Index (in-process Python):**
Run synchronously in a `asyncio.get_event_loop().run_in_executor(None, build_index)` call after train completes. Adapt `train_index()` logic from `web.py`: load npy files from `logs/rvc_finetune_active/3_feature768/`, build IVF index, write to `logs/rvc_finetune_active/added_IVF*_Flat_nprobe_1_rvc_finetune_active_v2.index`.

## Log Streaming Architecture

```
Training job
  phase 1,2,3: asyncio.create_subprocess_exec(stdout=PIPE, stderr=STDOUT)
               → read line by line → put to asyncio.Queue
  phase 4 (train): launch subprocess, then poll train.log file every 200ms
               → when new bytes appear → split lines → put to asyncio.Queue

WebSocket handler (/ws/training/:profile_id)
  → await queue.get() → send JSON {type, message, phase}
  → on "done" message: close WebSocket
```

Message schema:
```json
{"type": "log",   "message": "...",  "phase": "preprocess"}
{"type": "phase", "message": "extract_f0", "phase": "extract_f0"}
{"type": "done",  "message": "Training complete", "phase": "train"}
{"type": "error", "message": "...",  "phase": "train"}
```

State is held in-memory as `Dict[str, TrainingJob]` keyed by `profile_id`. This is safe for the single-user local tool.

## Constraints

- All subprocess commands must run with `cwd=RVC_ROOT` — all internal paths in RVC scripts are relative to the repo root.
- `rmvpe_root` env var must be set to `<RVC_ROOT>/assets/rmvpe` before extract_f0_print.py subprocess runs.
- `configs/inuse/v2/48k.json` must exist before `train.py` runs — it does (pre-baked in the repo).
- Mute data at `logs/mute/` must exist for filelist.txt padding — it does (pre-baked).
- `fp16_run=false` is already set in `configs/inuse/v2/48k.json` — no need to override.
- Training runs on CPU (not MPS) because `train.py` only calls `.cuda()` when CUDA is available; no `.mps()` call exists in the training loop.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` should be set for extract_feature_print.py (the script sets it itself, but also safe to pass in env).
- Profile `sample_path` in DB is relative to the `rvc-web/` CWD (e.g. `data/samples/<id>/file.wav`). Make absolute before passing to subprocess.
- WebSocket test requires `starlette.testclient.TestClient` (sync), not the `httpx` async client — `httpx` doesn't support WebSocket.

## Common Pitfalls

- **train.py produces no stdout** — polling its stdout pipe will block forever. Tail `train.log` via file polling instead. The log is written by `utils.get_logger()` with `FileHandler` only; there is no `StreamHandler` added.
- **extract_f0_print.py needs rmvpe_root** — The `Generator` class in `rvc/f0.py` reads `os.environ["rmvpe_root"]` at import time. If missing, the subprocess crashes immediately with `KeyError`. Pass `rmvpe_root` in the `env` dict to `create_subprocess_exec`.
- **filelist.txt must be built before train.py** — `train.py` reads `hps.data.training_files` (= `logs/<exp>/filelist.txt`) at startup. If it's missing, training crashes with a FileNotFoundError before logging anything useful.
- **config.json must exist before train.py** — `get_hparams()` does `open(config_save_path, "r")` immediately at module import. If missing, train.py crashes before `main()` is called.
- **inp_root for preprocess is a directory** — `preprocess.py` iterates all files in `inp_root` via `glob`. Pass the directory containing the wav, not the wav file itself. Derive as `os.path.dirname(os.path.abspath(profile.sample_path))`.
- **exp_dir for preprocess is the short name** — `preprocess.py` expects `exp_dir` as a path that it prepends with nothing (it uses `exp_dir` directly as an absolute path: `"%s/0_gt_wavs" % exp_dir`). Pass the full absolute path: `<RVC_ROOT>/logs/rvc_finetune_active`.
- **exp_dir for extract scripts is absolute** — same as preprocess. Confirm: `"%s/logs/%s" % (now_dir, exp_dir)` in web.py confirms this.
- **Cancellation leaves partial state** — killing the subprocess tree is best-effort. The `logs/rvc_finetune_active/` directory will have partial artifacts. This is acceptable per the roadmap's open question resolution.
- **Single active training job** — only one profile can train at a time. Return HTTP 409 if a job is already running.
- **Profile status updates race** — after subprocess completes, update DB status via a fresh `get_db()` context (S01 pattern). Don't pass connections across task boundaries.
- **Process group kill** — use `os.killpg(os.getpgid(proc.pid), signal.SIGTERM)` or `proc.kill()` for cancellation; a single `SIGTERM` to the process may not kill child processes spawned by `train.py` (which uses `mp.Process`).

## Open Risks

- **MPS fallback ops in extract_feature_print.py** — The script sets `PYTORCH_ENABLE_MPS_FALLBACK=1` which allows unsupported MPS ops to fall back to CPU. If specific ops are not handled by the fallback, the process may crash silently. Mitigation: capture stderr alongside stdout and surface any error lines.
- **train.py DDP with gloo on single CPU** — `dist.init_process_group(backend="gloo")` needs `MASTER_ADDR` and `MASTER_PORT` set. `train.py` sets these itself (`os.environ["MASTER_ADDR"] = "localhost"`, random port). This should work but has not been empirically verified in this environment.
- **DataLoader num_workers=4 + persistent_workers on macOS** — macOS has known issues with `multiprocessing` spawn + certain memory configurations. If training crashes early with DataLoader worker errors, fallback is to patch `num_workers=0`. This is unlikely but possible.
- **`torch.cuda.amp.GradScaler` with `enabled=False` on CPU** — Works at Python 3.11 + torch 2.10. Verified in research session. Not a risk.
- **train.log file creation race** — The log file is created by `get_logger()` inside `run()` which is called from a child `mp.Process`. There's a small window between subprocess start and log file creation. The file tailer must wait for the file to appear before tailing.
- **Long training time for MPS stability proof** — The M001 roadmap requires a real `.pth` to be produced. Even a 1-epoch test run (set `-te 1 -se 1`) should suffice to prove the pipeline works. Recommend a smoke-test mode in the S02 verification plan.

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| FastAPI | none searched | N/A — standard usage; docs not needed |
| RVC training pipeline | none | No installable skill exists for this proprietary system |

## Sources

- Subprocess command signatures extracted from `Retrieval-based-Voice-Conversion-WebUI/web.py` functions `preprocess_dataset()`, `extract_f0_feature()`, `click_train()`, `train_index()` (source: codebase)
- Argument parsing: `Retrieval-based-Voice-Conversion-WebUI/infer/lib/train/utils.py:get_hparams()` (source: codebase)
- Log file behavior: `FileHandler` only in `get_logger()` at `utils.py:312`, no `StreamHandler` (source: codebase)
- Model save path: `save_small_model()` in `process_ckpt.py:15` saves to `assets/weights/<name>.pth` (source: codebase)
- Device behavior on Apple Silicon: `train.py` uses `n_gpus=1` on MPS but never calls `.cuda()` on models → CPU training (source: codebase + runtime verification)
- `rmvpe_root` env var required: `extract_f0_print.py` → `rvc/f0.py` imports `os.environ["rmvpe_root"]` (source: codebase)
- `fp16_run=false` in `configs/inuse/v2/48k.json` confirmed at research time (source: live file read)
- All packages confirmed available: `asyncio`, `aiosqlite`, `websockets 16.0`, `starlette 0.52.1`, FastAPI WebSocket (source: `conda run -n rvc pip list`)
