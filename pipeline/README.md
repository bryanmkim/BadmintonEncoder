# Phase 1 pipeline

Turns broadcast match videos into data for the annotator. The project overview is in [../README.md](../README.md).

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

## 1C — Shuttle tracking (`shuttle_track.py` + `colab/tracknet_colab.ipynb`)

TrackNetV3 runs on a Colab GPU; everything else runs locally.

1. `.venv/bin/python shuttle_track.py pack` (add `--match <id>`, repeatable, to pack only some matches) writes `data/colab/tracknet_input_512.zip`: every clip shrunk to 512×288, plus each match's `segments.csv`, `court_reference.jpg` and `pack.json` (clip and source sizes). TrackNetV3 shrinks every frame to 512×288 itself, eight times over on one CPU core, so doing it once here made its preparation step about 5× faster (11 → 58 frames/s). On a test rally, visibility matched the full-size run on every frame and positions were a median 0.5 px apart at 720p. The exception was four consecutive fast frames (16–26 px of movement per frame), where the two differed by about 20 px, roughly one frame's worth of motion along the flight path. The notebook scales results back to 1280×720. `pack --full-size` packs the original clips instead.
2. Upload it to a Google Drive folder named `badminton-tracknet` in My Drive.
3. Open [colab/tracknet_colab.ipynb](colab/tracknet_colab.ipynb) in Colab (File → Upload notebook), set the runtime to a T4 GPU and run all. Set `LIMIT = 3` first for a trial run to see how long a clip takes. Each clip's result is saved to `badminton-tracknet/output/` as it finishes, so a disconnected session resumes where it stopped. The last cell zips everything to `tracknet_output.zip`.
4. `.venv/bin/python shuttle_track.py ingest ~/Downloads/tracknet_output.zip`, then `.venv/bin/python shuttle_track.py process`.
5. `.venv/bin/python shuttle_track.py overlay --match <id> --segment <n>` writes the clip with its track drawn on it, plus a sheet of stills, for the eye check. Every frame is stamped with its number, and once 1D has run each detected contact is marked on its own frame (red ring and `HIT n`), so stepping frame by frame (QuickTime: pause, then ← / →) checks each one.
6. `.venv/bin/python shuttle_track.py plot --match <id> --segment <n>` writes `tracks/plot_seg_NNNN.png`, a chart of the clip's track over time: TrackNetV3's raw detections, the ones cleaning dropped (stuck points and spikes), and the cleaned track. It has no video frames, so unlike the overlay it can be shared; the root README's tracking figure comes from it.

The notebook runs TrackNetV3's own `predict.py` (TrackNet + InpaintNet, temporal-ensemble mode, streaming frames with `--large_video`) at pinned commit `6eda442`, with three workarounds:

- Its `requirements.txt` is skipped. It pins torch 1.10 and numpy 1.22, which don't install on current Colab; the preinstalled versions work. Only `parse`, `pycocotools` (imported by `test.py` but not listed) and `gdown` are added.
- The background image is the match median from 1B rather than one built per clip, which would hold up to 1,800 full-size frames in RAM. On a test clip both gave identical positions.
- Data loading runs in-process. `predict.py` starts as many worker processes as the batch size (16), which crash on macOS and are pure overhead for InpaintNet's coordinate data.

Speed on a free Colab T4: 2.7 frames/s on full-size clips and 7.1 on the 512×288 ones, so all 5 matches (225,080 frames) take about 9 hours, split across sessions. The limit is TrackNetV3's CPU-side work, not the GPU. Its `--eval_mode nonoverlap` would be about 7× less work, but on the preview rally it went wrong a whole 8-frame window at a time during fast shots: after cleaning and smoothing, 26 of 149 frames were more than 10 px off the default mode (up to 118 px), against 5 of 150 for the 512×288 clips in the default mode. That kind of jitter is what 1D is most sensitive to, so the notebook keeps the default.

`process` cleans each raw track in order:

1. Drops TrackNetV3's stuck point. When it has nothing to track, it often reports the shuttle at one fixed position, (607.5, 177.5) px on all of the first 5 matches, sliding into it through the same few positions, sometimes for hundreds of frames. Any exact position found in at least 30% of a match's clips is treated as this artifact (on those matches the stuck positions were in 37-90% of clips, every other position in 20% or fewer; matches with under 10 clips are left alone), along with anything within 4 px of one: clips tracked from full-size video put the same artifact at (609, 178). Before this step it was 5-26% of each match's detections and pushed many clips to 100% visibility.
2. Drops a detection that jumps away and straight back (far from both neighbours, which are close to each other). A fast shot survives because its neighbours are far apart too.
3. Fills gaps of up to 5 frames (0.17 s) by linear interpolation.
4. Smooths each continuous run with a Savitzky-Golay filter (window 7, order 2).
5. Projects each point onto the floor with the match's `court.json`.

Output:

```
segments/<match_id>/tracks/raw/seg_0001_ball.csv   TrackNetV3's output, untouched
segments/<match_id>/tracks/seg_0001.csv            frame, detected, filled, x_px, y_px (smoothed),
                                                   raw_x_px, raw_y_px, floor_x_m, floor_y_m
tracks_report.csv                                  per segment: visibility raw (less stuck points) / after cleaning /
                                                   after gap-fill, stuck and spike frames dropped
```

The gate: TrackNetV3 visibility of at least 70% per segment, and the tracked position looking right over a full rally. A match below 60% overall should be dropped. Visibility is measured from a segment's first detection to its last, because clips run on past the rally (players walking back, the shuttle in hand). The whole-clip figure is in the report too: on the preview rally it was 60%, against 79% while the shuttle was in play.

### Known limits

- `floor_x_m` / `floor_y_m` are where the camera's line of sight meets the floor. That's the shuttle's position only when it's on the floor (see 1B).
- The smoothing window (0.23 s) rounds off the sharp change of direction at a hit. Raw positions are kept so 1D can tune it.
- A high shot that leaves the top of the frame, or a shuttle over the busy advertising boards behind the far player, can drop out for a second or more mid-rally. On the preview rally that was one 1.4 s gap during a net exchange. Gaps that long are left empty rather than guessed.

## 1D — Contact frames (`contact_detect.py`)

```bash
.venv/bin/python contact_detect.py labels                            # fetch ShuttleSet's hit frames (matches with a shuttleset_id)
.venv/bin/python contact_detect.py detect                            # every processed track -> segments/<match>/contacts/
.venv/bin/python contact_detect.py evaluate                          # recall and precision against ShuttleSet
.venv/bin/python contact_detect.py plot --match <id> --segment <n>   # one clip: track, fitted flights, contacts, labels
```

Between two hits the shuttle follows one smooth curve on screen, and a hit starts a new one. So each clip's track is split into the flights that explain it best, and every cut where the velocity jumps is a contact. For each clip it:

1. Takes TrackNetV3's raw positions after 1C's stuck-point and spike removal. Smoothed positions are no use here: smoothing rounds off exactly the turn a contact makes.
2. Splits the track wherever the shuttle is lost for more than 20 frames. A flight isn't bridged across a longer gap.
3. Fits flights. On each image axis a flight is `a + b·ln(1 + t/decay) + c·t + d·t²`, with t counted from the flight's first frame. The log term is how far a shuttle travels under air drag, which slows it fastest right after a hit; decay is shorter the harder it was hit, and each flight tries 2, 4, 8, 16 and 32 frames. The other terms take up gravity, the slow end of the flight and perspective. A cubic, tried first, couldn't bend that sharply and cut hard-hit flights in two.
4. Chooses the cuts by dynamic programming: the least total squared residual, plus a fixed penalty (1,000 px²) per flight. Flights are at least 11 frames long (ShuttleSet's 1st-percentile gap between hits) and at most 150. Both were raised from first guesses of 400 px² and 8 frames after the hand review: most false hits were one flight the model couldn't fit cut into minimum-length pieces, and those cuts removed a median 1,900 px² of residual against 13,700 for real hits. On the reviewed rallies this took precision on the random ones from 70% to 87% and recall from 99% to 93%, the loss mostly in fast net exchanges. Neighbouring flights share the detection at their cut, so the path stays continuous through a hit.
5. Keeps a cut only if the velocity jumps there by at least 3 px/frame; a smaller jump is the model running out of shape, not a hit. A cut into a flight whose top speed is over 2 px/frame is a hit; a cut from a moving flight into a still one is a landing.
6. Ends the rally at the first floor contact after a hit. The camera looks down from behind the near baseline, so anything in the air maps onto the floor further from the camera than it really is: a racket contact maps toward the net or past it, and only a shuttle on the floor maps to where it is. A hit that maps more than 4 m into the near half, or whose next 7 detections stay within 0.5 m of it on the floor inside the court, is the shuttle landing. (The line was first 3 m; on ShuttleSet, near-player shots hit low around mid-court mapped to 3.2-3.8 m and ended rallies early, each costing the rest of its rally's hits.) It becomes the rally's `landing`, as does an ordinary landing, and every hit after it (the bounce, pick-ups, the shuttle knocked back to the server) becomes `after_rally`. On the 280 hits checked by hand with `contact_review.py`, this removed 46 of 135 false hits and 1 of 145 real ones; before it, 74 of the false hits came after the rally's last real hit.

Output: `segments/<match_id>/contacts/seg_0001.csv` with `frame, kind (hit, landing or after_rally), x_px, y_px, floor_x_m, floor_y_m, speed_in, speed_out, kink` (speeds in px/frame at the cut; the floor position is only real for a landing).

### Evaluation against ShuttleSet

[ShuttleSet](https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet) labels the frame of every hit, with the hitter's position, on BWF broadcasts that are still on YouTube. Three of its 30 fps matches are in [matches.csv](matches.csv), with a `shuttleset_id` column naming their label folder: `yto2021-ms-f-axelsen-vs-ng` for tuning, `tto2021-ws-sf-marin-vs-an` and `wtf2020-ms-f-antonsen-vs-axelsen` held back for testing. Together they have 2,391 labelled hits inside kept clips, split evenly between the near and far player. Labels are in frames of the full broadcast, so `segments.csv` places each in its clip. A first contact sheet of 8 hits showed no offset, but over all three matches detected minus labelled peaked at -1 and -2 frames, and on a filmstrip of 6 hits the racket met the shuttle at or just after 1D's frame while the labelled frame was the follow-through. `read_labels` therefore moves every label one frame earlier (`LABEL_LAG`, chosen on the dev match); the detections are left where the contact is.

`evaluate` counts detected hits inside each labelled rally (15 frames before its first hit to 45 after its last, so hits made while picking the shuttle up between rallies don't count against precision). Labelled hits within 5 frames of a clip's start or end are left out. It prints recall and precision at ±1, 2, 3 and 5 frames (±2 is the CoachAI challenge's rule), recall for the near and far player, for hits TrackNet saw or missed, and per stroke type, and the median timing error. Every labelled hit's result goes to `data/contacts_eval.csv`.

The gate: recall above 90% and precision above 85% on the test matches, plus 5 hand-counted rallies from the 2025 matches to check it carries over.

Results at ±2 frames, with the parameters above:

| Match | Recall | Precision |
|---|---|---|
| yto2021 (dev) | 90.1% | 86.4% |
| tto2021 (test) | 79.4% | 83.1% |
| wtf2020 (test) | 83.6% | 84.0% |
| 2025 hand review, 35 random rallies | 92.8% | 84.3% |

Not passed on the test matches, mostly on recall. Near-player hits are found at 90%, far-player hits at 80%, and serves at 37% (usually shown on a close-up before the clip starts). Nearly every missed hit was tracked by TrackNet, so the misses are cuts the flight fit didn't make, plus rallies still ended early on tto2021 by the "stays within 0.5 m" floor test. The test matches were looked at while fixing the 3 m floor line, so they're no longer strictly unseen; a fresh pair of ShuttleSet matches is the clean check for the next change.

### Hand review on the 2025 matches (`contact_review.py`)

```bash
.venv/bin/python contact_review.py queue      # rallies most likely to be wrong, plus 1 random per match
.venv/bin/python contact_review.py review     # keyboard review, saved on every key, resumes where you stopped
.venv/bin/python contact_review.py summary    # precision / recall / timing, and the false rate per flag
.venv/bin/python contact_review.py queue --add --per-match 2 --random 3   # append a new batch, none already queued
```

ShuttleSet doesn't label the 2025 matches, so their hits are checked by eye. Each detected hit is flagged when something about it is doubtful: a velocity jump under 8 px/frame (`weak`), another contact within 12 frames (`close`), the shuttle leaving at under 4 px/frame (`slow`), TrackNet missing 3 of the 7 frames around it (`gap`), or within 5 frames of the clip's edge (`edge`). `queue` takes, per match, the 4 clips of 4-30 hits with the largest share of flagged hits, plus 1 clip at random, into `data/contact_review_queue.csv`. The random ones are the unbiased estimate for the gate; the suspects show where the detector fails and whether the flags predict it.

`summary` scores `contact_detect`'s current output against your verdicts, so after changing the detector, rerun `detect` and `summary` to see the effect without reviewing again. Hits are matched to your verdicts within ±2 frames, since a change to the detector can move a cut slightly: a hit at one you confirmed or marked missed is right, one at a hit you rejected is false, and a real hit the detector no longer reports is a miss. Any hit a change adds somewhere new is "not judged yet", and `review` goes back to just those.

`review` opens a window on each hit: the frame (red ring at the detected position), a 2x zoom there, and a strip of the 3 frames either side, all cropped on the same spot so the change of direction shows. `y` confirms a hit at the frame shown (step with the arrow keys first if it's a frame or two off; the offset is saved), `n` rejects it, `u` is unsure, `m` marks a hit the detector missed, `p` plays from the previous hit to the next, `b` undoes, `]` skips the rest of the clip. The next clip loads in the background. Verdicts go to `data/contact_review.csv`.

### Known limits

- The serve is often not in the clip. In `yto2021-ms-f-axelsen-vs-ng`, 35 of 69 rallies had their serve (sometimes with the next shot or two) 1-48 frames before the clip starts: the broadcast shows the serve on a close-up and cuts to the main camera just after. A clip can therefore begin mid-rally, and 1F's shot numbering has to allow for it.
- The parameters are tuned on the same hand-checked rallies they're scored on, so the ShuttleSet test matches are the honest measure. On the random reviewed rallies: precision 87%, recall 93%, 89% of hits on the exact frame and 95% within one. What's left is mostly one flight cut in two mid-air (a smash's steep, fast-slowing descent is the usual case) and far-side landings, where a shuttle on the floor and one on a racket map to the same place on the court. Fitting flights in 3D with the camera from 1B would address both. The broadcast audio was tried as a hit cue and isn't usable: the audio lag wasn't steady within a match, and at the best lag hit sounds barely told real hits from false ones.
- The landing can come a few frames early. When a smash's descent is itself cut in two, the second piece starts low enough to map deep into the near half, and that cut is taken as the landing rather than the true floor contact a moment later (on one clip 21 frames early, 2 m short). 1F should take the landing position from where the shuttle comes to rest when there is one.

## 1E — Players at each contact (`player_detect.py`)

```bash
.venv/bin/python player_detect.py detect                             # every match with contacts -> segments/<match>/players/
.venv/bin/python player_detect.py evaluate                           # hitter and feet against ShuttleSet's labels
.venv/bin/python player_detect.py sheet --match <id> --segment <n>   # stills of each contact with the players marked
```

Pose runs only on each contact frame and one frame either side, not the whole match. For each contact (hits and landings) it:

1. Runs YOLO11n-pose at 1280 px (the far player is only ~60 px tall at 720p). On an M1's GPU that's 100 ms a frame, and it found both players on 16 of 16 test contacts; yolo11s-pose took 890 ms.
2. Takes each person's feet (the midpoint of their ankles, or the bottom of their box) onto the court with `court.json`. People whose feet land more than 1.2 m outside the sidelines or 1.5 m behind a baseline are officials, coaches or crew. Of the rest, the largest on each half is that half's player.
3. Names the hitter: the player whose nearer wrist is closest to the shuttle, measured in the player's own heights so the small far player and the large near one compare fairly.
4. Takes each player's court position as the median over the three frames.

Output: `segments/<match_id>/players/seg_0001.csv` with `frame, kind, hitter (near/far), near_reach, far_reach` (shuttle-to-wrist distance in player heights) and each player's feet in court metres and pixels.

On the 2025 matches (5,896 hits, 34 minutes on an M1) both players were found at 95.7-99.7% of hits and a hitter named at all of them. `ultralytics`' NMS time limit is raised at load: on the Mac's GPU its clock also counts waiting for the model, and before that it gave up mid-batch about 50 times, returning nobody (hitter named at only 97.6%).

Consecutive hitters alternated at only 78-88% of hits, because 1D's false hits break the alternation. Using the players to filter 1D's hits was scored against the hand review: dropping hits with no wrist within 1.5 player heights of the shuttle took the random rallies from 84.3% precision / 92.5% recall to 85.1% / 90.3%, and tighter limits trade recall for precision about one for one. False hits often have a player close by (pick-ups after the rally, a flight split as it passes a player), so none of these filters is applied yet.

`evaluate` matches detected hits to ShuttleSet's within ±2 frames and checks the hitter against the labelled one (whose half their labelled feet are on) and our feet against the labelled feet of hitter and opponent. The gate: the right hitter at more than 95% of contacts.

**Passed.** On the three ShuttleSet matches the hitter was right at 1,864 of 1,901 matched hits (98.1%): 97.8% and 97.7% on the test matches and 98.6% on the dev match. Both players were found at 98.9-100% of hits. Our feet are a median 22 px from ShuttleSet's labelled opponent and 35 px from the labelled hitter, who is usually lunging or in the air at the contact, where "the feet" is loosely defined.

## 1F — Events for the annotator (`feature_assemble.py`)

```bash
.venv/bin/python feature_assemble.py assemble [--match <id>]   # -> data/events.json, strips, identity check sheets
.venv/bin/python feature_assemble.py validate                  # events.json against the annotator's event shape
.venv/bin/python feature_assemble.py evaluate                  # identity, landing and shot numbers vs ShuttleSet
.venv/bin/python feature_assemble.py swap --match <id>         # the match's two names are the wrong way round
```

Each clip with at least 2 hits is a rally; each hit becomes one event in the shape `badminton-annotator.jsx` reads (see the root `CLAUDE.md`), with `claude_label: null` (1G's suggestions go in `model_label`, from `shot_classify.py label`). Run `npm run dev` at the repo root and the annotator loads `/data/events.json`.

- **Who hit it.** 1E gives near or far, but players change ends between games. Their shirts don't change, so each clip's near and far shirt colour (median Lab inside the shoulders-hips quadrilateral, over 4 posed hits) is clustered into two players per match. Which cluster is on the near half is then chosen for the whole match at once, not clip by clip. Players change ends only between games and at 11 in the third, so at most 3 times, and on the 7 matches whose calls ran cleanly every change came at a gap of 53-134 s between rally clips (the median gap is 16-26 s). Dynamic programming finds the runs that best fit the clips' colours, allowing a change only at a gap of 45 s or more, each change costing 2 clips' worth of evidence. Taken clip by clip, Shi Yu Qi's red and Vitidsarn's pink at the 2025 Worlds had flipped the names 17 times in that match. The colours are kept in `identity.json`, so changing these rules needs no posing. The names go to the clusters in the match's `identity.json`, defaulting to `player_a`, `player_b` from `matches.csv`. **Check `identity_check.jpg` for each match**: a row of crops per cluster with the name it gets. If they're the wrong way round, run `swap` (on the first run, `wtf2025-ms-f-popov-vs-shi` needed it; the other four were right, judged by names printed on the backs of shirts). On the ShuttleSet matches the hitter's name was right at 97.8-98.6% of matched hits, the same as 1E's near/far, so the clustering adds no errors there.
- **Positions.** `player_xy` and `opponent_xy` are 1E's feet at the contact.
- **Landing.** For a returned shot, ShuttleSet's landing is where the next player hits it, so `landing_xy` is the floor point below the shuttle at the next contact: straight down the screen from the shuttle to the next hitter's feet level, then onto the court. Against ShuttleSet's labelled landings that was a median 1.0-1.1 m off, against 1.4-1.5 m for the next hitter's feet and 4.2-4.8 m for the shuttle's image position taken straight onto the court (ShuttleSet's landing points are on the floor, about 100 px from the airborne shuttle). The last shot's landing is 1D's floor contact (median 0.26-0.47 m off), else the last detection that maps onto the court (a shuttle leaving the top of the frame maps up to 100 m past the far baseline, so the track is walked back from its end; still in the air, so rough). `landing_source` says which.
- **Speed** is the average over the floor from hitter to landing, in km/h (median about 29). The launch speed of a shot moving toward or away from the camera can't be read from one view. It's left empty when the landing comes less than 0.2 s after the hit or the average would exceed 300 km/h: both are 1D pairing a false hit with a landing.
- **Angle** is the launch direction on screen from 1D's outgoing velocity, degrees above horizontal.
- **Timing.** `flight_s` is the seconds from the hit to its landing (the next hit, or the floor), and `since_prev_s` the seconds since the previous hit in the clip (null for the first). 1G's classifier uses them, since ShuttleSet times every hit.
- **Frames.** `frames` is one JPEG per event: frames f-2 to f+2, each 320x180, cropped 16:9 on the hitter (feet to 1.8 m above them, from the 1B camera) and the shuttle, with the shuttle ringed on the contact frame. They're in `segments/<match_id>/strips/`, and kept between runs. A clip's last event has no next hit to show where the shot went, so it also gets `frames_after`: the court 0.5, 1 and 1.5 s later (a 16:9 box holding the court floor and 3 m above it, from the 1B camera; sized from the floor alone, it cut the far player off at the waist), with the shuttle ringed where 1C tracked it, or "clip has ended" once the broadcast cuts away.

The ShuttleSet matches are left out of `events.json`: ShuttleSet is the predictor's training data and holds its 1I test set, so labelling them again would leak into that test. `evaluate` builds their events in memory instead.

### Known limits

- **Shot numbers are often off.** `shot_num` counts 1D's hits in the clip, which matched ShuttleSet's `ball_round` at only 20-39% of hits. The serve is usually missing (shown on a close-up before the clip starts), which pulls numbers down, and 1D's missed and extra hits shift the rest. The predictor trains on whole rallies from the serve, so rallies should be checked before export, or only those whose first event is a serve kept.
- About 18% of returned shots land on the hitter's own half, which a returned shot can't. It's 1D's false hits (two hits in a row on one side) or a wrong landing, and it matches the rate at which consecutive hitters fail to alternate. Such a landing is never used as real: `shot_classify.py` ignores it, and the annotator flags the shot and leaves it out of the export until the review fixes it (X on the false hit, or L). Since 2026-09-15 the same goes for a rally's last shot, though in ShuttleSet 687 of the 708 last shots landing on their own half are net errors.
- The annotator finds the strips through the dev server, which serves the repo root. With `BADMINTON_DATA_DIR` pointing outside the repo, the strip URLs won't resolve.

## 1G — Shot-type suggestions (`shot_classify.py`)

```bash
.venv/bin/python shot_classify.py train [--no-cv]   # og_train.csv -> data/shot_classifier.pkl (~3 min with the cross-validation)
.venv/bin/python shot_classify.py evaluate          # accuracy on the ShuttleSet matches
.venv/bin/python shot_classify.py label             # model_label on every event in data/events.json; rerun after assemble
```

The plan was Claude vision labelling. But every exported shot is reviewed by hand, so a suggestion only has to make the review faster, and a shot's type follows mostly from numbers 1D-1F already measure. ShuttleSet labels those numbers for 30,000 shots in exactly the predictor's 10 types. So this trains gradient-boosted trees (scikit-learn's `HistGradientBoostingClassifier`) on the predictor's own `og_train.csv`. It learns ShuttleSet's conventions for the types directly, costs nothing per match, and can be graded on the ShuttleSet matches without labelling anything.

- **Training data.** `og_train.csv` is ShuttleSet22's training split, with its 18 types already merged into the predictor's 10. Any og_train match sharing half its hit frames with one of our ShuttleSet matches is left out: `wtf2020` is og_train's match 7 (676 of 676 frames; the next largest overlap is 14). That leaves 29,494 shots from 43 matches. Its template coordinates convert to court metres within a median 0.07 m of ShuttleSet's raw pixels mapped through our `court.json`. For the evaluation, ShuttleSet's 18 raw types are merged as ShuttleSet22's `preprocess_data.py` does (return net, defensive lob and defensive drive become "defensive shot", and so on).
- **Features** are the same from ShuttleSet and from 1F: the hitter's, opponent's and landing position in metres, turned 180° when the hitter is on the far half (the same shot seen from the other end); where the previous hitter stood; the seconds since the previous hit and to the next; the outgoing average speed, distance and sideways change; and the incoming speed from the previous hitter, meant to separate a net shot from a blocked smash (it left top-1 on the dev match unchanged and top-3 up from 91.6% to 92.3%). ShuttleSet has no shuttle height or launch angle, so 1D's angle isn't used. A landing where the track ends (still in the air) is left out, as is the flight time to a floor landing: ShuttleSet times only hit to hit.
- **No landings on the hitter's own half.** A returned shot can't land there, so such a landing is a wrong landing or a false hit, never data. It's left out along with the flight time to that hit: in training (393 of og_train's 27,904 returned shots, label noise) and on 1F's events (18% of returned shots, mostly 1D's false hits). Since 2026-09-15 a rally's last shot loses its own-half landing too, though in ShuttleSet 687 of 708 of those are nets. That moved each match by under 1.5 points either way. The rough copies hide a returned shot's landing 18% of the time to match. On the dev match's events that raised top-3 from 92.3% to 94.0% and the confident suggestions from 80.5% to 82.3% right, with top-1 about the same (68.8% to 69.4%).
- **Rough copies.** Trained on ShuttleSet's clean numbers alone, the model leaned on precision the pipeline doesn't have. It also called every rally's first detected hit a serve, but in 1F a clip often starts after the serve. So each shot is trained twice: once clean, and once with 1D-1F's measured errors added. That's 0.4 m on the feet, 0.85 m per axis on the landing (1F's landings are a median 1.0-1.1 m off), 0.05 s on the timing, and the previous hit hidden 30% of the time. On the dev match's events, that took top-1 from 61.7% to 68.9% and short-service precision from 24% to 70%. The test matches moved the same way (68.5% to 73.3%, 61.5% to 69.3%).
- **Output.** `label` writes `model_label: { shot_type, p, top: [[type, p] × 3], source }` into each event. The annotator shows it with its probability (green at 70% and above) and the next two alternatives, and Enter confirms it. `assemble` rewrites `events.json` without it, so run `label` after.

`evaluate` gives two results per match. With ShuttleSet's own positions and timings as input, it shows how good the model can be. With 1F's events matched to ShuttleSet's hits within ±2 frames, it shows how good the model is on our measurements:

| Match | From ShuttleSet's numbers | 1F's events: top-1 | top-3 | Suggestions at p ≥ 0.7 |
|---|---|---|---|---|
| og_train, 5-fold by match | 85.7% (top-3 98.4%) | | | |
| yto2021 (dev) | 82.7% | 68.5% | 93.5% | 64% of events, 83.3% right |
| tto2021 (test) | 88.0% | 73.6% | 91.5% | 72% of events, 86.0% right |
| wtf2020 (test) | 80.6% | 70.1% | 92.1% | 66% of events, 81.7% right |

**Not passed:** the gate was 75% agreement, and the test matches reach 69-74%. The gap between the two columns is 1D-1F's measurement error, not the model. On the 2025 events, 60% of suggestions come with p ≥ 0.7.

### Known limits

- **Weak types.** Drive (45% recall even on ShuttleSet's own numbers) and push/rush (50%) look like smashes and lobs from positions and timing alone. On 1F's events, a net shot is often taken for a blocked smash (a "defensive shot" in ShuttleSet's merge). These are the cases where Claude looking at the frames could add something.
- **No previous shot type.** The previous shot's label is known during review, and the sequence would help (a block usually follows a smash), but `events.json` is labelled before review starts.
- **1D's errors carry through.** After a missed hit, the "next hit" is two shots on, so the landing and flight time describe the wrong shot.
