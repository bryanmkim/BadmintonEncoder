# Phase 1, Divided Into Sub-Phases

Each sub-phase has a stop-and-check gate. If a gate fails, you fix it before moving on rather than discovering the problem three steps later.

---

## Phase 1A — Video Prep(COMPLETED)

**Build:** download 5 singles matches at 720p, detect camera cuts via frame histogram distance, keep only wide-angle live-rally segments.

**Effort:** 1-2 days.

**Gate:** you have a folder of segments, and spot-checking 10 of them shows they're all the main court angle with a rally in progress. No replays, no crowd shots.

**Fails if:** cut detection misses gradual transitions. Fix by also flagging segments where court corners aren't detectable.

---

## Phase 1B — Court Homography(COMPLETED)

**Build:** a click-four-corners script that saves a homography matrix per match. Project net and service lines back onto the frame to verify.

**Effort:** half a day.

**Gate:** projected lines land within ~10px of the painted lines on all 5 matches.

**Why here:** every downstream step wants court coordinates, not pixels. Doing this early means you never write pixel-space logic you'll have to rewrite.

---

## Phase 1C — Shuttle Tracking

**Build:** run TrackNetV3 per segment, interpolate short gaps, apply Savitzky-Golay smoothing, convert to court coordinates.

**Effort:** 2-3 days, mostly environment wrangling.

**Gate:** shuttle visibility above 70% per segment, and overlaying the tracked position on video looks correct to your eye across a full rally.

**Fails if:** visibility is below 60% on a match. That match's camera angle or lighting isn't workable — drop it, pick another.

---

## Phase 1D — Contact-Frame Detection

This is the one that decides whether the whole approach works. Do it on **one rally** before scaling.

**Build:** velocity vector per frame, angle change between consecutive vectors, local maxima above threshold as contact candidates. Filter by minimum inter-contact gap and proximity to a player.

**Effort:** 3-5 days. Most of your Phase 1 time goes here.

**Gate:** on 5 hand-counted rallies, recall >90% and precision >85%.

**Fails if:** tracking jitter creates false spikes. Increase smoothing. If real contacts get smoothed away too, you have a fundamental resolution problem — try a higher frame rate source or a learned detector instead of the geometric heuristic.

**Stop here if this gate fails badly.** Everything downstream assumes you can segment shots. No point building the rest.

---

## Phase 1E — Player Detection

**Build:** YOLOv8-pose per frame, assign detections to near/far player by court y-coordinate, extract player position at each contact frame.

**Effort:** 1-2 days. This is mostly off-the-shelf.

**Gate:** correct player attributed to >95% of contacts.

**Why after 1D:** you only need player positions at contact frames, which you don't know until 1D works. Running pose on every frame of every match wastes hours of compute.

---

## Phase 1F — Feature Assembly

**Build:** for each contact, assemble the event object the annotator expects — match, rally, shot number, player xy, landing xy, speed, trajectory angle, plus a 5-frame image strip around contact.

Landing position = the shuttle's court position at the *next* contact frame, or where it hits the floor if the rally ends.

**Effort:** 1-2 days.

**Gate:** the JSON validates against your annotator's expected shape, and loading it into the UI renders sensible court diagrams.

---

## Phase 1G — Claude Vision Labeling

**Build:** send each contact's frame strip plus trajectory summary to the Claude Batch API. Prompt for shot type, hitting player, confidence, and reasoning as JSON.

**Effort:** 1-2 days including prompt iteration.

**Gate:** on 100 hand-labeled events, agreement above 75%. If below, iterate the prompt — usually the fix is giving Claude the court diagram and trajectory numbers alongside the frames, not frames alone.

---

## Phase 1H — Annotate

**Build:** nothing. Swap the annotator's mock data for real events.

**Effort:** 3-5 hours of review for ~2,000 shots from 5 matches.

**Gate:** confirm-rate above 80%. Note which shot types you override most — that's your Phase 2 prompt-improvement list.

---

## Phase 1I — The Real Test

**Build:** train your existing decoder on the CV-extracted dataset. Evaluate on a ShuttleSet22 holdout set.

**Effort:** 1 day.

**Gate:** accuracy within 5 points of your current 48.5%. So roughly 44% or better.

**This is the decision point.** Pass, and Phase 2 (scale to 100+ matches) is justified. Fail, and you go back to whichever sub-phase produced the most noise — usually 1D or 1G.

---

## Sequencing Note

1A → 1B → 1C are independent enough to parallelize if you want, but 1D depends on 1C, and everything after depends on 1D.

**Total realistic estimate: 3-4 weeks part-time**, with 1D consuming a third of it.

One shortcut worth considering: do 1D on a single rally by hand-cropping the video before building 1A and 1C properly. If contact detection is going to fail, you want to know in day 3, not week 2. You can hand-extract one rally's shuttle track from TrackNetV3 in an hour without any of the segmentation infrastructure.
