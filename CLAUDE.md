# CLAUDE.md

This file guides Claude Code (claude.ai/code) when working in this repository.

## What this is

Two parts that together build training data for a separate project, **BadmintonShotPredictor** (`~/Desktop/Personal Projects/BadmintonShotPredictor`, which has its own `.venv`):

1. **Annotator** (repo root): a single-file React app where a human confirms or corrects a suggested shot-type label for each detected contact, then exports the predictor's `train.csv` format.
2. **Phase 1 pipeline** (`pipeline/`): Python scripts that turn BWF broadcast videos on YouTube into the contact events the annotator reviews.

The plan, split into sub-phases 1A–1I, each with a pass/fail gate, is in [PHASE_1_README.md](PHASE_1_README.md). [pipeline/README.md](pipeline/README.md) documents every stage in detail: algorithms, measured numbers, outputs and known limits. Read the relevant section before changing a stage.

**Status (2026-09-12):** 1A (video prep), 1B (court homography) and 1C (TrackNetV3 shuttle tracking) are done. 1D (contact detection, `contact_detect.py`) is built but short of its gate on the ShuttleSet test matches (recall 79-84%, precision 83-84% at ±2 frames); the user chose to move on. 1E (players, `player_detect.py`) passed its gate (right hitter at 98.1%). 1F (`feature_assemble.py`) wrote `data/events.json`: 5,999 events in 433 rallies from the five 2025 matches, validated; the other half of its gate (the user checking the court diagrams in the annotator) is pending. 1G is built as `shot_classify.py`, a classifier trained on ShuttleSet rather than Claude vision labelling: on the pipeline's events for the ShuttleSet test matches it's right at 69-74% (top-3 92-94%), short of the 75% gate, and its suggestions are in every event's `model_label`. Claude vision labelling isn't built (`claude_label: null`); it's the fallback for the types the numbers can't separate. Next is 1H (annotating).

## Commands

Annotator (Node, Vite 8, React 19):

```bash
npm install
npm run dev        # dev server at http://localhost:5173; also auto-saves annotations.csv to disk.
                   # Keep port 5173: the review lives in that origin's localStorage, so another port looks empty
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
.venv/bin/python contact_review.py queue [--add]|review|summary                                      # 1D hand review (OpenCV window)
.venv/bin/python player_detect.py detect|evaluate|sheet --match <id> --segment <n>                   # 1E
.venv/bin/python feature_assemble.py assemble|validate|evaluate|swap --match <id>                     # 1F
.venv/bin/python shot_classify.py train [--no-cv]|evaluate|label                                     # 1G; label again after every assemble
```

1E and 1F run YOLO11n-pose through `ultralytics` on the Mac's GPU (MPS).

1C is split in two. `pack` zips 512×288 clips into `data/colab/`, the user uploads the zip to Google Drive and runs [pipeline/colab/tracknet_colab.ipynb](pipeline/colab/tracknet_colab.ipynb) on a T4 GPU, then `ingest` and `process` the result locally. The notebook pins TrackNetV3 to commit `6eda442` and deliberately skips its `requirements.txt`.

There's no test suite or linter. Checking works through each phase's gate: diagnostic images (`timeline.png`, `court_overlay.jpg`, `court_gate.jpg`, `spot_check.png`, overlay videos, `players/sheet_*.jpg`, `identity_check.jpg`), `data/tracks_report.csv`, `contact_detect.py evaluate` (recall and precision at ±1/2/3/5 frames against ShuttleSet), `contact_review.py summary` (against the user's hand verdicts in `data/contact_review.csv`), `player_detect.py evaluate`, `feature_assemble.py validate`/`evaluate`, and `shot_classify.py evaluate` (shot-type accuracy against ShuttleSet, from its own positions and from 1F's events). The annotator's geometry and export code (everything from `const COURT =` up to `// With editLanding`, and the `// === BADMINTONSHOTPREDICTOR CSV ===` section) is plain JS: test it in Node by slicing those parts of the source into `new Function(...)`, then run `npm run build` to check the JSX compiles.

## Annotator architecture

- `index.html` → `main.jsx` → `badminton-annotator.jsx`. The whole app is one file, originally written as a Claude artifact. It uses inline styles only (no CSS files) and has no other dependencies.
- **Events and storage:** events come from `/data/events.json` (written by `feature_assemble.py`; the dev server serves the repo root) and fall back to `generateMockEvents` when it's missing. Only the review persists, as `{event id: { annotation, gap_before, landing_fix }}` (`landing_fix` is a landing moved on the court with L, normalised; the export prefers it), through `window.storage.get/set`, which Claude artifacts provide and `main.jsx` shims onto `localStorage`. `STORAGE_KEY` is `annotator:review:v4` (v3's bare annotations are read once as a fallback). An annotation is either a label `{ shot_type, hitting_player, confirmed, corrected, timestamp }` or `{ not_shot: true, timestamp }`. Bump the version when the review shape or label set changes incompatibly. `POSITION_KEY` separately remembers the event on screen and the filter, so a reload continues there (or at the first unreviewed event if that one is gone).
- **Disk save:** a plugin in `vite.config.js` handles `PUT /api/annotations.csv` and overwrites `./annotations.csv` (or `$ANNOTATIONS_CSV`). The app sends a PUT on every change, queued so writes land in order. This only works under `npm run dev` or `preview`; otherwise the Download button is the fallback. The dev watcher ignores `*.csv`, `pipeline/` and `data/`.
- **Event shape.** The mock events in `generateMockEvents` show the core:
  `{ id, match, players: [P1, P2], rally, shot_num, frame_time, cv: { player_xy, opponent_xy, landing_xy, speed, trajectory_angle }, claude_label: { shot_type, hitting_player (1|2), confidence, reasoning }, annotation: null | { shot_type, hitting_player, confirmed, corrected, timestamp } }`.
  The `*_xy` fields are normalised court fractions (see coordinates below); a landing out of court falls outside [0,1]. Pipeline events (1F) add `cv.hitting_player` (1|2, from 1E's side and 1F's shirt identity), `cv.hitter_side`, `cv.landing_source` (`next_hit`, `floor` or `track_end`), `cv.flight_s` (seconds from the hit to its landing) and `cv.since_prev_s` (from the previous hit in the clip, null for the first), `frames` (URL of the strip image), `frames_after` on a clip's last event (the court 0.5, 1 and 1.5 s later, shown where the next shot's strip would be) and `source` (`match_id`, `segment`, `clip`, `frame`), and have `claude_label: null`. `shot_classify.py label` adds `model_label: { shot_type, p, top: [[type, p] × 3], source }`. `cv.speed` is the average floor speed from hitter to landing in km/h; `cv.trajectory_angle` is the launch direction on screen, degrees above horizontal (-90..90). `hitterOf` in the app picks Claude's hitter when there is one, else the pipeline's. `suggestionOf` picks `claude_label`, else `model_label`: it's what Enter confirms and what the 3D court previews.
- **Review keys:** 1–0 shot type, Enter confirm the suggestion, X not a shot (a 1D false hit), G shot missing before this one (splits the rally), L move the landing (click or drag on the court, from the top view), U undo, ← → navigate. The same actions have buttons.
- `Court3D` is a hand-rolled SVG perspective renderer. It draws painter's-order layers split at the net. Each shot's arc comes from the heuristic `SHOT_PROFILES`, not measured data. In landing mode, clicks go through `screenToFloor`: the floor's projection is a homography fitted on the court corners and inverted.
- `FrameStrip` shows the middle three frames (f-1, contact, f+1) of the event's strip image (1F writes f-2..f+2, cropped on the hitter and shuttle), or placeholders for mock events. The layout is the 3D court on the left and, on the right, the strips for this shot and the next one in the rally (its contact is where this shot went), then the label and controls.

## Export contract (don't break it)

`annotations.csv` must stay loadable by the predictor's `main.py`, which uses `pd.factorize` over train types and its `RallyDataset`. Validate changes by loading the CSV with the predictor's own code.

- It has 7 columns, in this order: `rally_id, ball_round, player, type, landing_x, landing_y, rally_length`.
- `SHOT_TYPES` are the predictor's 10 class strings exactly as written (e.g. `"push/rush"`, `"defensive shot"`). Keys 1–9 then 0 select them in order.
- Only fully reviewed rallies are exported, and each is split into pieces (`rallyPieces`). The predictor's `RallyDataset` reads a rally's rows in order and learns shot → next shot, ignoring `ball_round` and `rally_length`, and its loss skips the first 3 positions. So a piece may start mid-rally, but a gap would teach a false transition. A shot marked "not a shot" (X, a 1D false hit) is dropped, and the shot before it takes its landing. A shot marked "shot missing before" (G, `gap_before`) starts a new piece, and the shot before the gap is dropped (its landing is really the shot after the gap). Pieces under `MIN_PIECE_SHOTS` (5) are left out; `ball_round` and `rally_length` are renumbered per piece.
- A returned shot whose landing is on its hitter's own half (`onOwnHalf`) is left out and the rally split there, until the review fixes it with L or X. The annotator flags it too. A rally's last shot may land on its own half (into the net), so there it's only flagged.
- `rally_id` = `RALLY_ID_OFFSET` (10000) + rally index × 100 + piece, which keeps ids stable and clear of the predictor's 0–4938.
- Player ids: the predictor's 0–34 are anonymised, but they were recovered into `PLAYER_IDS`. ShuttleSet22's `preprocess_data.py` numbers players by first appearance in its `set/match.csv` (`pd.unique` over winner, loser), and player A is always the match winner; rebuilt that way, the ids matched all 58 train/val/test matches and 30,162 of 30,172 `og_train.csv` shots. Names match ignoring case, spaces and punctuation (`playerKey`). A rally with a player outside the 35 is left out of the export (the export panel names them): the predictor's `Embedding(36)` has no row for a new id (ids are shifted +1 for padding). Of the current matches that's only Christo Popov, so `wtf2025-ms-f-popov-vs-shi` exports nothing; the user chose that over growing the embedding.
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
- **Landings (the user's rule).** A returned shot's landing on its hitter's own half is never accepted as real: either the landing is wrong or a hit is false. Flag or drop it, count it, and never draw conclusions from it. A rally's last shot is the exception (into the net: 687 of ShuttleSet's 708 last shots landing on their own side).
- **Docs style.** Docstrings, comments and the README explain *why*, with the measured number behind each threshold. When you change behaviour or a threshold, update the matching `pipeline/README.md` section and the module docstring in the same change.

## 1D decisions already made by the user

- Keep the geometric flight-splitting approach: piecewise drag-model curve fits, dynamic programming over cuts, velocity-jump rule. **Don't propose a learned detector** or a sequence model trained on ShuttleSet labels.
- Player detection and proximity filtering belong to 1E. Don't pull a person detector into 1D.
- ShuttleSet ground truth: tune on `yto2021-ms-f-axelsen-vs-ng` only, and report on the held-out `tto2021-ws-sf-marin-vs-an` and `wtf2020-ms-f-antonsen-vs-axelsen`. Labels are cached in `data/shuttleset/`. The test matches were looked at while fixing the floor line, so a fresh pair of ShuttleSet matches is the clean check for the next 1D change.
- The user's 887 hand verdicts on the 2025 matches (`contact_review.py`) drove the current parameters (cut penalty 1000 px², minimum flight 11 frames, rally ends at the first floor contact with a 4 m line). Timing within a few frames is fine to the user; false hits after the rally (ground contacts, pick-ups) were their main complaint. Audio was tried as a hit cue and dropped (unstable lag, weak separation).
- The ShuttleSet matches are left out of `events.json`: they're the predictor's training data and hold its 1I test set.

## 1G decisions

- On 2026-09-12 the user chose a classifier trained on ShuttleSet (`shot_classify.py`) over Claude vision labelling for the suggestions. Every exported shot is reviewed by hand, so a suggestion only has to make review faster.
- It trains on the predictor's `og_train.csv` and drops any match that is one of our ShuttleSet matches (`wtf2020` is og_train's match 7). Tune on `yto2021` and report the two test matches, as in 1D.
- Don't use BadmintonShotPredictor itself to suggest labels: it would be labelling its own training data.

## Repo hygiene

- `.gitignore` excludes all of `data/` (clips, Colab zips of up to ~600 MB, derived outputs), `node_modules/`, `pipeline/.venv/`, `__pycache__/` and `.DS_Store`. Never force-add anything from `data/`: GitHub rejects files over 100 MB, and the remote (`bryanmkim/BadmintonEncoder`) is public.
- The clips come from copyrighted BWF broadcasts and are for personal research only. Never publish or redistribute them.
