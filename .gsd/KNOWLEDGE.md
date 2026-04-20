# KNOWLEDGE — rvc-web

Append-only register of project-specific rules, patterns, and lessons learned.
Read at the start of every unit. Never remove or edit existing entries.

---

## Service Management

**Rule: Always use `start.sh` and `stop.sh` (project root) to start/stop services.**

Never use raw `uvicorn`, `conda run`, `pnpm dev`, `lsof | kill`, or `pkill`.

- `start.sh` exports required env vars (`PROJECT_ROOT`, `KMP_DUPLICATE_LIB_OK`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`), kills stale ports and dangling processes, writes PIDs to `.pids/`.
- `stop.sh` shuts down worker subprocesses (`spawn_main`, `_realtime_worker`) before killing the backend.

---

## Directory Layout

```
rvc-web/
  assets/              ← pretrained weights, hubert, rmvpe, beatrice2
  backend/app/         ← FastAPI app
  backend/beatrice2/   ← Beatrice 2 pipeline + trainer source
  backend/rvc/         ← RVC code; assets/ and logs/ are symlinks to project root
  data/profiles/       ← per-profile audio, checkpoints, model files
  data/rvc.db          ← SQLite DB
  logs/                ← RVC training workspace (rvc_finetune_active/)
```

`PROJECT_ROOT` env var is the single source of truth (not `RVC_ROOT` — that's gone).
Training subprocess cwd = `backend/rvc/`. The symlinks inside it make relative paths resolve correctly.

---

## Path Handling

**Rule: Always convert `profile_dir` and `sample_dir` to absolute paths before passing to subprocesses.**

RVC training subprocesses run with `cwd=rvc_root`. Relative paths resolve against `rvc_root`, not the project root → `FileNotFoundError`.

Fix: call `os.path.abspath()` on `profile_dir` and `audio_dir` in `routers/training.py` immediately after construction.

---

## Audio Preprocessing

**Rule: Always pass `format="WAV"` explicitly to `soundfile.write()` when the output path has a non-standard extension.**

`soundfile` infers format from extension. Temp files like `moeed.wav.cleaning` cause `TypeError: No format specified`. Pass `format="WAV"` explicitly.

---

## RVC Training Workspace

**Rule: The shared `logs/rvc_finetune_active/` directory is NOT profile-scoped.**

- Profile delete wipes `exp_dir` entirely.
- New training checks a sentinel file (`.profile_id`) and wipes if profile changed.
- A fresh profile must never inherit `G_latest.pth`/`D_latest.pth` from a prior run.

---

## RVC: Epoch Loss Cleanup on Profile Delete

**Rule: `epoch_losses` rows must be deleted explicitly on profile delete** — aiosqlite does not enforce `ON DELETE CASCADE` by default.

```sql
DELETE FROM epoch_losses WHERE profile_id = ?
DELETE FROM audio_files WHERE profile_id = ?
DELETE FROM profiles WHERE id = ?
```

---

## RVC: Best Epoch — Use `loss_mel`, Skip First 50 Epochs

**Rule: `G_best.pth` tracks lowest epoch-average `loss_mel`, not `loss_gen`. Skip epochs 1–50.**

- `loss_gen` plateaus at ~epoch 12 and never improves. Using it freezes `G_best` forever.
- `loss_mel` decreases monotonically — the direct proxy for output quality.
- First 50 epochs are warmup; `G_best` is not updated until epoch 51.
- In `train.py`: `lowest_mel` / `avg_mel` / `avg_mel=X` log line. DB column `best_avg_gen_loss` stores `avg_mel` (name unchanged for backward compat).

---

## MPS: 4D Constant Padding → Use `torch.cat`

**Rule: Never use `F.pad(..., mode='constant')` on tensors with `ndim > 3` on MPS.**

MPS silently falls back to CPU for 4D+ constant padding — fires every attention step.

Fix in `backend/rvc/rvc/layers/attentions.py`:
```python
# Before:
x = F.pad(x, [0, 1, 0, 0, 0, 0, 0, 0])
# After (native MPS):
x = torch.cat([x, torch.zeros(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)], dim=-1)
```

3D `F.pad` calls are fine — MPS handles ≤3D natively.

---

## MPS: Batch Size → LR Scaling (sqrt rule)

**Rule: Scale learning rate with `LR = 1e-4 × sqrt(batch_size / 32)` on MPS.**

Reference: `batch_size=32, LR=1e-4`. At `batch_size=8`, use `LR=5e-5`.

`_write_config` in `backend/app/training.py` auto-scales on every training start.

---

## MPS: fp16 AMP Must Be Disabled

**Rule: Never train with `fp16_run=True` on MPS. Use fp32.**

MPS unified memory means no off-chip bandwidth to save. fp16 causes weight_norm underflow → NaN gradients → GradScaler collapse loop.

`_write_config` sets: `cfg["train"]["fp16_run"] = torch.cuda.is_available()`

---

## Beatrice 2: DB Writer — `is_best` Default

**Rule: Always use `losses.get("is_best", 0)` — never `losses.get("is_best")`.**

`beatrice_steps.is_best` is `NOT NULL DEFAULT 0`. Passing an explicit `None` via `losses.get("is_best")` overrides the column default → constraint violation → every regular step insert silently fails → all losses show NULL in chart.

This was the root cause of "all losses showing 0" while only UTMOS appeared.

---

## Beatrice 2: ON CONFLICT — COALESCE All Loss Columns

**Rule: Use `COALESCE(excluded.col, col)` for ALL loss columns in the ON CONFLICT clause.**

Regular step inserts (every 10 steps) have `utmos=NULL`. UTMOS-only inserts (every 500 steps) have `loss_mel=NULL` etc. Without COALESCE, one insert type overwrites the other's data with NULLs.

See `backend/beatrice2/db_writer.py` `insert_step_loss()` — all 8 columns use COALESCE.

---

## Beatrice 2: Best Checkpoint — Skip First 1000 Steps (Warmup)

**Rule: Do not designate a best checkpoint for the first 1000 steps. During warmup, `checkpoint_latest` = `checkpoint_best`.**

Same rationale as RVC's 50-epoch skip: early UTMOS scores are unstable. After step 1000, best is tracked by highest UTMOS.

In `loop.py`:
```python
_warmup = (iteration + 1) <= 1000
_is_new_best = (not _warmup) and (_utmos_now > _best_utmos)
```

During warmup, every `save_interval` checkpoint is also copied to `checkpoint_best.pt.gz` so inference always has a loadable model.

---

## Beatrice 2: Training Data Flow

```
loop.py (subprocess)
  every 10 steps → BeatriceDBWriter.insert_step_loss(step, {loss_mel, loss_loud, ...})
  every 500 steps → BeatriceDBWriter.insert_step_loss(step, {utmos, is_best})
                  → BeatriceDBWriter.update_best_utmos(step, utmos)

training.py (backend, async)
  _poll_beatrice_steps() polls beatrice_steps every 2s
  → emits step_done WS events with losses dict

frontend training/page.tsx
  step_done handler → setStepPoints() → BeatriceStepChart renders
```

Poller query: `SELECT ... FROM beatrice_steps WHERE profile_id = ? AND step > ? ORDER BY step ASC`

---

## Beatrice 2: Subprocess Invocation

```python
cmd = [python, "-m", "backend.beatrice2.beatrice_trainer",
       "-d", data_dir, "-o", out_dir, "-c", config_path,
       "--db", DB_PATH, "--profile-id", profile_id]
# cwd = project_root
```

`DB_PATH` is passed so the subprocess's `BeatriceDBWriter` writes to the same DB the backend reads. If `--db` or `--profile-id` is missing, `_enabled=False` and all writes are silently skipped.

---

## Beatrice 2: GradBalancer `return {}` on Non-finite Gradient

**Rule: When `grad_balancer.backward()` returns `{}`, the training step continues normally** — losses are still appended to `dict_scalars` and the regular insert still fires. The empty dict just means `generator_stats = {}` for that step. This is NOT a cause of missing DB rows.

The unfixed version (`loss.py` raising on non-finite and returning `{}` after `input.backward(grad)`) exists in `backend/beatrice2/beatrice_trainer/train/loss.py` — the NaN propagation is handled by GradScaler, not a crash.

---

## SQLite: Never Query `gsd.db` Directly

Use `gsd_*` tools (gsd_milestone_status, gsd_journal_query, etc.). Direct `sqlite3`/`better-sqlite3` access causes WAL conflicts.

---

## Diagnosing "Chart Shows 0 / Waiting"

Checklist in order:
1. `sqlite3 data/rvc.db "SELECT step, loss_mel, utmos FROM beatrice_steps ORDER BY step LIMIT 10;"` — are rows there? Are values non-NULL?
2. If rows have NULLs → DB writer inserting NULLs → check `insert_step_loss` exception log in training stdout
3. If no rows → `_enabled=False` or subprocess not receiving `--db`/`--profile-id`
4. If rows are fine → poller silent exception (`except Exception: rows = []`) — add logging temporarily
5. If poller fires → WS not delivering to client — check browser WS connection
