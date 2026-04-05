# gdlevelai

Python toolkit for:

1. Downloading a Geometry Dash `.gmd` dataset from GDHistory with careful rate limiting.
2. Training/sampling a plain autoregressive baseline (default) that produces `.gmd` files.
3. Experimental diffusion generator support.

The downloader is the primary component and is designed for low-impact crawling:

- Uses API search filters (`cache_featured >= 1`) and sorts oldest-to-newest (`online_id:asc`).
- Supports stricter styles via `--min-epic-tier` (for epic+ buckets).
- Enforces request spacing + jitter + retries + periodic batch pauses.
- Applies a politeness profile with hard request budgets and cooldown rollover.
- Uses a SQLite global rate limiter so multiple shards still share one safe request budget.
- Saves progress in SQLite with per-level metadata + download status.
- Resumes automatically from the last downloaded `online_id`.
- Uses DB-first duplicate detection (no filesystem scan required).
- Uses a multi-worker downloader and paged record lookup to speed up downloads.
- Supports adaptive auto-tuning to raise/lower throughput dynamically.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Scripts (Colab-friendly)

You can use helper scripts from `scripts/`:

```bash
python scripts/prep_dataset.py --max-levels 200
python scripts/train_model.py --dataset-dir dataset/gmd --model-out models/autoregressive.pt --epochs 10
python scripts/sample_level.py --model-path models/autoregressive.pt --out-path generated/level_01.gmd
python scripts/colab_bootstrap.py --dataset-dir dataset/gmd_overfit --epochs 1
```

- `scripts/prep_dataset.py` fetches data (unless `--skip-fetch`) and runs `dataset verify`.
- `scripts/train_model.py` wraps common training args and forwards extra args.
- `scripts/sample_level.py` wraps sampling args and forwards extra args.
- `scripts/colab_bootstrap.py` installs deps (optional), runs a tiny train, then samples.

## 1) Fetch featured dataset from GDHistory

```bash
python -m gdlevelai fetch \
  --output-dir dataset/gmd \
  --state-db dataset/state/fetch_state.sqlite3 \
  --metadata-jsonl dataset/metadata/levels.jsonl \
  --max-levels 500 \
  --politeness-profile careful \
  --workers 3
```

Notes:

- Default behavior is conservative to avoid stressing GDHistory.
- Files are saved as `<online_id>_<record_id>.gmd`.
- Metadata is appended to JSONL and state is tracked in SQLite.
- The fetcher resumes from the last downloaded level automatically.

### Resume controls

```bash
python -m gdlevelai fetch --start-online-id 1000000
python -m gdlevelai fetch --no-resume-from-last-downloaded
```

- `--start-online-id` forces a custom lower bound (`online_id > value`).
- `--resume-from-last-downloaded` is on by default and uses DB history.

### Politeness controls

```bash
python -m gdlevelai fetch \
  --max-levels 500 \
  --politeness-profile very_careful \
  --max-requests-per-hour 500 \
  --cooldown-seconds-on-budget 300
```

### Speed tuning (still respectful)

```bash
python -m gdlevelai fetch \
  --workers 4 \
  --limit-per-page 300 \
  --level-record-page-size 60 \
  --level-record-max-pages 8 \
  --politeness-profile normal
```

- `--workers` controls concurrent level downloads.
- `--limit-per-page` reduces search-page overhead.
- `--level-record-page-size` / `--level-record-max-pages` speed up record discovery.
- Keep politeness/rate caps enabled so total server load remains bounded.

### Adaptive auto-tune

```bash
python -m gdlevelai fetch \
  --auto-tune \
  --auto-min-workers 1 \
  --auto-max-workers 6 \
  --auto-target-seconds-per-level 8
```

Auto-tune adjusts workers/page sizes between batches using live telemetry,
while still respecting the global request budget.

### Sharded crawling (parallel-safe)

Run each shard in a separate process/terminal:

```bash
python -m gdlevelai fetch --shard-index 0 --shard-count 4 --global-limiter-db dataset/state/global_limiter.sqlite3
python -m gdlevelai fetch --shard-index 1 --shard-count 4 --global-limiter-db dataset/state/global_limiter.sqlite3
python -m gdlevelai fetch --shard-index 2 --shard-count 4 --global-limiter-db dataset/state/global_limiter.sqlite3
python -m gdlevelai fetch --shard-index 3 --shard-count 4 --global-limiter-db dataset/state/global_limiter.sqlite3
```

All shards share one limiter DB, so total request rate remains capped globally.

## 2) Train model

```bash
python -m gdlevelai train \
  --dataset-dir dataset/gmd \
  --model-out models/autoregressive.pt \
  --epochs 10
```

Training automatically writes Colab-resilient outputs under `models/`:

- `models/checkpoints/epoch_XXXX.pt` (per-epoch checkpoints)
- `models/artifacts/` (vocab, token mappings, processed sequences, schema metadata)
- `models/samples/` (periodic sample `.gmd` outputs + JSON summaries)

Resume from a checkpoint:

```bash
python -m gdlevelai train \
  --dataset-dir dataset/gmd \
  --model-out models/autoregressive.pt \
  --resume-checkpoint models/checkpoints/epoch_0008.pt
```

Device selection (`--device`) supports: `auto`, `cpu`, `cuda`, `rocm`, `directml`.
Default is `auto`.

If training looks slow/stuck on very large corpora, use bounded epoch steps and denser logs:

```bash
python -m gdlevelai train \
  --model-type autoregressive \
  --max-steps-per-epoch 1500 \
  --sample-stride 4 \
  --log-every-steps 50 \
  --num-threads 8
```

The baseline now trains on object-level tokens (not raw characters), which is much faster.
If you intentionally want clipping, set `--max-objects-per-file N` (default `0` = disabled).

For CPU, try increasing `--num-threads` and keep `--torch-compile` enabled (default).

### ROCm and DirectML support

- **ROCm (AMD on Linux / ROCm-enabled builds):** install a ROCm PyTorch build, then run with `--device rocm` (or `--device auto`).
- **DirectML (Windows):** install `torch-directml`, then run with `--device directml` (or `--device auto`).

DirectML install example:

```bash
pip install torch-directml
```

Notes from upstream docs:

- `torch-directml` uses the DirectML backend on Windows.
- DirectML package support is tied to specific PyTorch versions (Microsoft currently documents support up to PyTorch 2.3.1).

Default training now uses the autoregressive baseline.
By default it uses full object strings (no per-file clipping).
Use diffusion explicitly (experimental):

```bash
python -m gdlevelai train \
  --model-type diffusion \
  --model-out models/diffusion.pt
```

## 3) Sample a generated level

```bash
python -m gdlevelai sample \
  --model-path models/autoregressive.pt \
  --out-path generated/level_01.gmd \
  --level-name "Autoregressive Trial" \
  --device auto
```

Use diffusion sampling explicitly (experimental):

```bash
python -m gdlevelai sample \
  --model-type diffusion \
  --model-path models/diffusion.pt \
  --out-path generated/diffusion_level_01.gmd
```

## 4) Dataset tools (view size, records, integrity)

```bash
python -m gdlevelai dataset summary
python -m gdlevelai dataset recent --limit 25
python -m gdlevelai dataset top-creators --limit 15
python -m gdlevelai dataset find --query "wave" --limit 50
python -m gdlevelai dataset verify --limit 200
python -m gdlevelai dataset id-report --limit 100
python -m gdlevelai dataset version-report --limit 100
python -m gdlevelai dataset reset-limiter
python -m gdlevelai dataset migrate-state-db
python -m gdlevelai dataset reset-limiter --global-limiter-db dataset/state/global_limiter.sqlite3
python -m gdlevelai dataset verify --json-only
python -m gdlevelai dataset verify --no-strict
python -m gdlevelai dataset archive-untracked --dry-run
python -m gdlevelai dataset archive-untracked
```

Useful outputs include:

- total downloaded/failed/skipped/indexed counts
- dataset disk usage (`gmd_total_bytes_human`)
- last downloaded online ID
- creator leaderboard
- quick file existence checks for recent downloads

`dataset verify` compares:

1. `.gmd` files in `dataset/gmd`
2. metadata entries in `dataset/metadata/levels.jsonl`
3. downloaded state (`status='ok'`) in `dataset/state/fetch_state.sqlite3`

It reports invalid filenames, duplicate IDs, metadata parsing issues, and cross-source mismatches.
It prints a human summary first, then JSON. Exit code is non-zero if any issues are found.

Options for `dataset verify`:

- `--limit N`: truncates long issue/mismatch lists in output.
- `--json-only`: prints only the final JSON object.
- `--no-strict`: only fails on cross-source mismatches (ignores quality-only issues for exit status).

`dataset archive-untracked` moves `.gmd` files that are not tracked as downloaded in state
(`gmd_without_state`) into a timestamped archive folder. Use `--dry-run` to preview first.

## GDHistory endpoints used

- `GET /api/v1/search/level/advanced/`
- `GET /api/v1/level/{online_id}/`
- `GET /level/{online_id}/{record_id}/download/`

All dataset requests are made to `https://history.geometrydash.eu`.
