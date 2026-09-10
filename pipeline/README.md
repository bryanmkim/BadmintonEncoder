# Phase 1 pipeline

Turns broadcast match videos into data for the annotator. The plan and its gates are in [../PHASE_1_README.md](../PHASE_1_README.md).

## Setup

```bash
brew install ffmpeg                     # decoding and clip cutting
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

yt-dlp needs a JavaScript runtime for YouTube. The scripts pass `--js-runtimes node`, so Node must be on your `PATH`.

## 1A — Video prep (`video_prep.py`)

```bash
.venv/bin/python video_prep.py                   # every match in matches.csv not processed yet
.venv/bin/python video_prep.py --match <id>      # one match; --force to redo, --keep-raw to keep the download
.venv/bin/python video_prep.py --resegment       # re-apply segment rules from the cached analysis, no download
.venv/bin/python video_prep.py --spot-check 10   # gate check: contact sheet of 10 random kept segments
```

Add a match by appending a row to [matches.csv](matches.csv) (`match_id`, YouTube id, event, round, discipline, players).

For each match it:

1. Downloads the highest-bitrate 720p H.264 version plus audio. Audio is kept because racket-hit sounds may help contact detection in 1D.
2. Decodes every frame at 160×90 and records the colour-histogram jump from the previous frame (hard cuts) and a 32×18 thumbnail.
3. Finds the main camera view. The fixed wide-angle camera is on screen more than any other shot, so its frames form the densest cluster. Every frame is scored by its distance to that view, and an Otsu threshold splits main view from everything else. This also catches dissolves and logo wipes, which the histogram alone misses.
4. Keeps runs of main-view frames, split at hard cuts, with 0.2 s trimmed off each edge and anything under 4 s dropped. A run whose median distance is above 18 is also dropped: on the first 5 matches the main view stayed at 15 or below even under score graphics or changed LED boards, while other wide cameras (a low corner camera, a low baseline camera) sat just under the per-frame threshold at 24 or above. Each remaining run is cut into its own clip.
5. Deletes the full download (unless `--keep-raw`).

If you tighten a rule later, `--resegment` re-applies the rules from the cached per-frame analysis and removes and renumbers clips without downloading anything. It refuses, and asks for `--force`, if the new rules would need clips that don't exist yet.

Output, under `$BADMINTON_DATA_DIR` (default `../data`):

```
segments/<match_id>/seg_0001.mp4 ...   kept clips, H.264 + AAC
segments/<match_id>/segments.csv       each clip's start/end frame and time in the source video
segments/<match_id>/match.json         source id, format, fps, thresholds, totals
segments/<match_id>/analysis.npz       per-frame features, used by --resegment
segments/<match_id>/timeline.png       per-frame diagnostics with kept spans in green
segments/<match_id>/rejected.png       sample of dropped footage, to check nothing live was lost
spot_check.png                         output of --spot-check
```

To keep the data on an external drive, set `BADMINTON_DATA_DIR=/Volumes/<drive>/badminton-data`.

### Known limits

- The main view must fill at least a fifth of the broadcast. That held easily on BWF broadcasts, where it's about 25% of the full video including the intro and ceremony.
- Clips keep whatever the director shows on the main camera, so a clip can include a few seconds of players walking back before or after a rally.
- Main-view stretches under 4 s (after trimming) are dropped, which can lose a serve-fault rally.

These are copyrighted BWF broadcasts: keep the downloads and clips for personal research and don't redistribute them.

## 1B — Court homography (`court_calibrate.py`)

```bash
.venv/bin/python court_calibrate.py                          # every match with clips
.venv/bin/python court_calibrate.py --match <id>
.venv/bin/python court_calibrate.py --match <id> --click     # seed from 4 clicked corners if the automatic search fails
```

For each match it:

1. Builds a clean reference image: the per-pixel median of 41 frames spread across the match's clips. The players and shuttle move, so they vanish; the camera doesn't, so the court stays sharp.
2. Masks the painted lines: thin structures brighter than their surroundings and nearly colourless. Brightness is luminance rather than HSV value, because a red court is almost as high in value as white paint.
3. Seeds the homography automatically instead of clicking. Detected near-horizontal and steep lines are paired with every pair of model lines, and each set of four intersections gives a candidate. Candidates that don't look like a court seen from behind a baseline, or that no real camera could produce, are discarded, and the one whose projected court lands best on painted pixels wins. `--click` replaces this step with four clicks: far-left, far-right, near-right, near-left.
4. Refines: fits each painted line's centre near its projection and re-solves, narrowing the band each round. A band never reaches more than 45% of the way to the next parallel line, because at the far end the baseline and long service line are only about 10 px apart.
5. Checks the result three ways. It measures each line's distance from its projection to the fitted painted line. It recovers the camera (focal length and position) from the homography and projects the 3D net. And it re-measures the same homography on references built from each quarter of the match, to catch a camera that moved.

The gate: at least 10 of the 12 lines found, and every found line within 10 px, on the full-match reference and on every quarter.

Output per match, next to the clips:

```
court.json            homographies court<->image, camera K/R/t, per-line errors, stability, pass/fail
court_reference.jpg   the median frame
court_mask.png        detected line pixels
court_overlay.jpg     projected lines (green) and net (yellow), with 2x zooms of the corners and net
```

plus `court_gate.jpg` with every match's overlay in one image.

Court coordinates are metres in the annotator's frame: X across (+ right as seen from the main camera), Y along (-6.7 far baseline to +6.7 near baseline, net at 0), Z up. [court.py](court.py) has the geometry and helpers: `apply_h`, `to_normalized` (the annotator's 0-1 fractions), `camera_from_homography` and `project`.

### Known limits

- The homography only maps points on the floor. A shuttle in flight or a player's hand maps to wherever the line of sight meets the floor, not to the point below it. Use players' feet for their court position, and the shuttle only where it touches the floor (1F's landing points).
- Camera recovery assumes square pixels, the principal point at the image centre and no lens distortion. Line errors of 1-4 px on these broadcasts suggest that holds.
- One homography per match assumes a fixed main camera. The per-quarter check flags a match where it moved.
