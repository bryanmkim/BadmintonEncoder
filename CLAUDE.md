# CLAUDE.md

This file guides Claude Code (claude.ai/code) when working in this repository.

## What this is

Two parts that together build training data for a separate project, **BadmintonShotPredictor** (`~/Desktop/Personal Projects/BadmintonShotPredictor`, which has its own `.venv`):

1. **Annotator** (repo root): a single-file React app where a human confirms or corrects Claude's shot-type label for each detected contact, then exports the predictor's `train.csv` format.
2. **Phase 1 pipeline** (`pipeline/`): Python scripts that turn BWF broadcast videos on YouTube into the contact events the annotator reviews.

The plan, split into sub-phases 1A–1I, each with a pass/fail gate, is in [PHASE_1_README.md](PHASE_1_README.md). [pipeline/README.md](pipeline/README.md) documents every stage in detail: algorithms, measured numbers, outputs and known limits. Read the relevant section before changing a stage.

**Status:** 1A (video prep) and 1B (court homography) are done. 1C (TrackNetV3 shuttle tracking) is built. 1D (contact detection, `contact_detect.py`) is in progress. 1E onward (player detection, feature assembly, Claude vision labelling) isn't built yet, so the annotator still runs on mock events.

## Commands

Annotator (Node, Vite 8, React 19):

```bash
npm install
npm run dev        # dev server; also auto-saves annotations.csv to disk
npm run build
```

Pipeline: run from `pipeline/` with its venv. The scripts import each other by bare module name (`import court`, `from video_prep import ...`), so the working directory has to be `pipeline/`.

```bash
cd pipeline
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # also needs ffmpeg, and node on PATH for yt-dlp
.venv/bin/python video_prep.py [--match <id>] [--force] [--keep-raw] [--resegment] [--spot-check N]   # 1A
.venv/bin/python court_calibrate.py [--match <id>] [--click]                                         # 1B
.venv/bin/python shuttle_track.py pack|ingest <zip>|process|overlay --match <id> --segment <n>       # 1C
.venv/bin/python contact_detect.py labels|detect|evaluate|plot --match <id> --segment <n>            # 1D
```

1C is split in two. `pack` zips 512×288 clips into `data/colab/`, the user uploads the zip to Google Drive and runs [pipeline/colab/tracknet_colab.ipynb](pipeline/colab/tracknet_colab.ipynb) on a T4 GPU, then `ingest` and `process` the result locally. The notebook pins TrackNetV3 to commit `6eda442` and deliberately skips its `requirements.txt`.

There's no test suite or linter. Checking works through each phase's gate: diagnostic images (`timeline.png`, `court_overlay.jpg`, `court_gate.jpg`, `spot_check.png`, overlay videos), `data/tracks_report.csv`, and `contact_detect.py evaluate` (recall and precision at ±1/2/3/5 frames against ShuttleSet).

## Annotator architecture

- `index.html` → `main.jsx` → `badminton-annotator.jsx`. The whole app is one file, originally written as a Claude artifact. It uses inline styles only (no CSS files) and has no other dependencies.
- **Storage:** the app calls `window.storage.get/set`, which Claude artifacts provide. `main.jsx` shims it onto `localStorage`. Events persist under `STORAGE_KEY` (`annotator:events:v2`). Bump the version when the event shape or label set changes incompatibly.
- **Disk save:** a plugin in `vite.config.js` handles `PUT /api/annotations.csv` and overwrites `./annotations.csv` (or `$ANNOTATIONS_CSV`). The app sends a PUT on every change, queued so writes land in order. This only works under `npm run dev` or `preview`; otherwise the Download button is the fallback. The dev watcher ignores `*.csv`, `pipeline/` and `data/`.
- **Event shape.** Phase 1F must produce exactly this. The mock events are in `generateMockEvents`:
  `{ id, match, players: [P1, P2], rally, shot_num, frame_time, cv: { player_xy, opponent_xy, landing_xy, speed, trajectory_angle }, claude_label: { shot_type, hitting_player (1|2), confidence, reasoning }, annotation: null | { shot_type, hitting_player, confirmed, corrected, timestamp } }`.
  The `*_xy` fields are normalised court fractions in [0,1] (see coordinates below).
- `Court3D` is a hand-rolled SVG perspective renderer. It draws painter's-order layers split at the net. Each shot's arc comes from the heuristic `SHOT_PROFILES`, not measured data.
- `FrameStrip` is still a placeholder: 5 empty frames around contact.

## Export contract (don't break it)

`annotations.csv` must stay loadable by the predictor's `main.py`, which uses `pd.factorize` over train types and its `RallyDataset`. Validate changes by loading the CSV with the predictor's own code.

- It has 7 columns, in this order: `rally_id, ball_round, player, type, landing_x, landing_y, rally_length`.
- `SHOT_TYPES` are the predictor's 10 class strings exactly as written (e.g. `"push/rush"`, `"defensive shot"`). Keys 1–9 then 0 select them in order.
- Only complete rallies are exported, because a gap would teach the model a false shot-to-shot transition.
- `rally_id` = `RALLY_ID_OFFSET` (10000) + index, which keeps it clear of the predictor's 0–4938.
- Player ids: the predictor's are 0–34 and anonymised. Unknown players get ids from 35 up, which overflow the predictor's `Embedding(36)`. This is still open: someone has to fill in `PLAYER_IDS` or grow the embedding.
- Landing coordinates are in ShuttleSet's court template: `((px − 175)/82, (py − 467)/192)`, with the doubles court at template x 27.5–327.5 and y 150–810 and the net at 480. ShuttleSet's "landing" for a returned shot is where the next player hits it, not where it bounces.

## Coordinate frames

There are three; don't mix them up.

- **Court metres** (`pipeline/court.py`, and `toWorld` in the annotator): X runs across the court (±3.05, + right as seen from the main camera). Y runs along it (−6.7 far baseline, 0 net, +6.7 near baseline). Z points up.
- **Normalised** (the annotator's `*_xy`): fractions in [0,1], converted with `court.to_normalized`. Here y=0 is the far baseline.
- **Predictor template** (export only): `toPredictorXY` in the annotator.

A homography only maps floor points. `floor_x_m/floor_y_m` for a shuttle in flight is where the line of sight meets the floor, not the point under the shuttle. Use players' feet, and the shuttle only when it touches the floor.

## Pipeline conventions

- **Shared helpers.** `video_prep.py` owns `DATA_DIR` (`$BADMINTON_DATA_DIR`, default `../data`), `SEG_DIR`, `log`, `label`, `sheet` and `read_matches`. `shuttle_track.py` owns `match_dirs` and `read_rows`. `court.py` holds the geometry and camera maths. Later stages import from these instead of redefining them.
- **Config** is module-level UPPER_CASE constants at the top of each script. Each one has a comment giving its unit and why it has that value. Frame counts assume 30 fps; pixel values assume 1280×720.
- **Matches** are rows in `pipeline/matches.csv` (`match_id, youtube_id, event, round, discipline, player_a, player_b, shuttleset_id`). A match is identified by its `match_id` slug everywhere.
- **Per-match data** lives in `data/segments/<match_id>/`: `seg_NNNN.mp4` clips, `segments.csv`, `match.json`, `analysis.npz`, `court.json`, plus `tracks/raw/`, `tracks/` and `contacts/`. The full download is deleted after segmenting so data can scale to 100+ matches, and `--resegment` works from the cached `analysis.npz`. New stages should read local clips under `DATA_DIR`, never stream from YouTube.
- **Docs style.** Docstrings, comments and the README explain *why*, with the measured number behind each threshold. When you change behaviour or a threshold, update the matching `pipeline/README.md` section and the module docstring in the same change.

## 1D decisions already made by the user

- Keep the geometric flight-splitting approach: piecewise drag-model curve fits, dynamic programming over cuts, velocity-jump rule. **Don't propose a learned detector** or a sequence model trained on ShuttleSet labels.
- Player detection and proximity filtering belong to 1E. Don't pull a person detector into 1D.
- ShuttleSet ground truth: tune on `yto2021-ms-f-axelsen-vs-ng` only, and report on the held-out `tto2021-ws-sf-marin-vs-an` and `wtf2020-ms-f-antonsen-vs-axelsen`. Labels are cached in `data/shuttleset/`.

## Repo hygiene

- There's no `.gitignore` yet. `node_modules/`, `.DS_Store`, `__pycache__/` and large files under `data/` (clips, and Colab zips of up to ~600 MB) show up as tracked or staged. GitHub rejects files over 100 MB. Don't stage `data/`, `node_modules/` or `pipeline/.venv/` unless the user asks.
- The clips come from copyrighted BWF broadcasts and are for personal research only. Never publish or redistribute them.
