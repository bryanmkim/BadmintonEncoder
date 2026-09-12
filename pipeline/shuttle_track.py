"""Phase 1C — shuttle tracking, local half.

TrackNetV3 runs on a Colab GPU (colab/tracknet_colab.ipynb). This script packs the clips for it and
turns its raw per-frame detections into cleaned, court-referenced tracks.

  .venv/bin/python shuttle_track.py pack                          # -> data/colab/tracknet_input_512.zip, for Google Drive
                                                                  #    (--full-size: original clips, tracknet_input.zip;
                                                                  #     --match <id>, repeatable: only those matches)
  .venv/bin/python shuttle_track.py ingest <tracknet_output.zip>   # raw tracks -> segments/<match>/tracks/raw/
  .venv/bin/python shuttle_track.py process                        # clean every raw track and write the gate report
  .venv/bin/python shuttle_track.py overlay --match <id> --segment <n>   # clip with the track drawn, plus a still sheet
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter

import court
from video_prep import DATA_DIR, SEG_DIR, label, log, sheet

COLAB_DIR = DATA_DIR / "colab"
PACK_SIZE = (512, 288)       # TrackNetV3's own input size: it shrinks every frame to this anyway
STUCK_SHARE = 0.3            # an exact position detected in this share of a match's clips is a TrackNetV3 artifact
STUCK_MIN_CLIPS = 10         # too few clips to tell an artifact from chance; no stuck filter below this
STUCK_RADIUS_PX = 4          # a detection this close to a stuck position is the artifact too (full-size runs land ~1.5 px off)
SPIKE_PX = 60                # a lone detection this far from both neighbours, which agree with each other, is dropped
MAX_GAP = 5                  # frames; gaps up to this long are filled by linear interpolation (0.17 s at 30 fps)
SG_WINDOW, SG_ORDER = 7, 2   # Savitzky-Golay smoothing; 1D may need to retune this around contacts
GATE_VISIBILITY = 0.70       # 1C gate, per segment
DROP_VISIBILITY = 0.60       # below this for a whole match, its camera or lighting isn't workable

TRACK_FIELDS = ["frame", "detected", "filled", "x_px", "y_px", "raw_x_px", "raw_y_px", "floor_x_m", "floor_y_m"]


def match_dirs(match=None):
    return [d for d in sorted(SEG_DIR.iterdir())
            if (d / "segments.csv").exists() and (d / "court.json").exists() and match in (None, d.name)]


def read_rows(match_dir):
    with open(match_dir / "segments.csv", newline="") as f:
        return list(csv.DictReader(f))


# === COLAB HAND-OFF ===
def shrink(src, dst):
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", f"scale={PACK_SIZE[0]}:{PACK_SIZE[1]}:flags=bicubic",
        "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "12", str(dst),
    ], check=True)


def pack(full_size=False, only=None):
    """One zip for Google Drive: Colab reads a single large file far faster than hundreds of small ones.

    By default the clips are shrunk to TrackNetV3's input size. It resizes every frame to that itself, eight
    times over on one CPU core; doing it once here made its preparation step 5x faster, and on a test rally
    visibility matched the full-size run on every frame with positions a median 0.5 px apart at 720p.
    Each match's pack.json records both sizes so the notebook can scale results back to the source video.
    `only` limits the zip to those match ids.
    """
    COLAB_DIR.mkdir(parents=True, exist_ok=True)
    out = COLAB_DIR / ("tracknet_input.zip" if full_size else "tracknet_input_512.zip")
    tmp = COLAB_DIR / "pack_tmp"
    dirs = [d for d in match_dirs() if not only or d.name in only]
    jobs = []
    for d in dirs:
        meta = json.loads((d / "match.json").read_text())
        source = [meta["width"], meta["height"]]
        (tmp / d.name).mkdir(parents=True, exist_ok=True)
        (tmp / d.name / "pack.json").write_text(json.dumps({"clip_size": source if full_size else list(PACK_SIZE), "source_size": source}))
        jobs += [(clip, tmp / d.name / clip.name) for clip in sorted(d.glob("seg_*.mp4"))]
    if not full_size:
        log(f"shrinking {len(jobs)} clips to {PACK_SIZE[0]}x{PACK_SIZE[1]}")
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda job: shrink(*job), jobs))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:  # mp4s don't compress
        for d in dirs:
            for name in ("segments.csv", "court_reference.jpg"):
                z.write(d / name, f"{d.name}/{name}")
            z.write(tmp / d.name / "pack.json", f"{d.name}/pack.json")
        for src, small in jobs:
            z.write(src if full_size else small, f"{src.parent.name}/{src.name}")
    shutil.rmtree(tmp)
    log(f"{len(jobs)} clips -> {out} ({out.stat().st_size / 1e6:.0f} MB)")


def ingest(zip_path):
    count = 0
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            parts = Path(name).parts
            if not name.endswith("_ball.csv") or len(parts) < 2 or not (SEG_DIR / parts[-2]).is_dir():
                continue
            dest = SEG_DIR / parts[-2] / "tracks" / "raw" / parts[-1]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(name))
            count += 1
    log(f"{count} raw tracks ingested")


# === CLEANING ===
def read_raw(path, n_frames):
    """TrackNetV3 CSV -> (detected, xy) per frame; undetected frames are NaN."""
    detected = np.zeros(n_frames, bool)
    xy = np.full((n_frames, 2), np.nan)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            i = int(r["Frame"])
            if 0 <= i < n_frames and int(r["Visibility"]) == 1:
                detected[i] = True
                xy[i] = float(r["X"]), float(r["Y"])
    return detected, xy


def stuck_positions(raw_paths):
    """Exact positions TrackNetV3 reports in a large share of a match's clips. It parks the shuttle at a
    fixed point, (607.5, 177.5) px on the first 5 matches, when it has nothing to track, and slides into
    it through the same few positions; a real shuttle never lands on one exact pixel in rally after rally.
    On the first 5 matches these sat in 37-90% of clips and every other position in 20% or fewer.
    Clips tracked at full size put the same artifact at (609, 178), too few of them to cross the share, so
    process_clip also drops anything within STUCK_RADIUS_PX of a stuck position."""
    if len(raw_paths) < STUCK_MIN_CLIPS:
        return set()
    clips = Counter()
    for path in raw_paths:
        with open(path, newline="") as f:
            clips.update({(float(r["X"]), float(r["Y"])) for r in csv.DictReader(f) if int(r["Visibility"]) == 1})
    return {p for p, n in clips.items() if n >= STUCK_SHARE * len(raw_paths)}


def drop_spikes(detected, xy):
    """Drop a detection that jumps away and straight back: far from both neighbours while they're close
    to each other. A fast shot moves far but its neighbours are far apart too, so it survives."""
    keep = detected.copy()
    idx = np.flatnonzero(detected)
    for k in range(1, len(idx) - 1):
        prev, cur, nxt = xy[idx[k - 1]], xy[idx[k]], xy[idx[k + 1]]
        if (idx[k + 1] - idx[k - 1] <= 2 * (MAX_GAP + 1)
                and min(np.linalg.norm(cur - prev), np.linalg.norm(cur - nxt)) > SPIKE_PX
                and np.linalg.norm(nxt - prev) < SPIKE_PX):
            keep[idx[k]] = False
    return keep


def fill_gaps(detected, xy):
    """Linearly interpolate gaps of up to MAX_GAP frames between two detections."""
    present, out = detected.copy(), xy.copy()
    idx = np.flatnonzero(detected)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= MAX_GAP + 1:
            t = (np.arange(a + 1, b) - a) / (b - a)
            out[a + 1:b] = xy[a] + t[:, None] * (xy[b] - xy[a])
            present[a + 1:b] = True
    return present, out


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def smooth(present, xy):
    """Savitzky-Golay over each continuous run; runs shorter than the window stay as they are."""
    out = xy.copy()
    for s, e in runs(present):
        if e - s >= SG_WINDOW:
            out[s:e] = savgol_filter(xy[s:e], SG_WINDOW, SG_ORDER, axis=0)
    return out


def fmt(v, digits):
    return "" if not np.isfinite(v) else f"{v:.{digits}f}"


def process_clip(raw_path, n_frames, H_image_to_court, out_path, stuck=frozenset()):
    reported, raw_xy = read_raw(raw_path, n_frames)
    is_stuck = np.zeros(n_frames, bool)
    if stuck and reported.any():
        idx = np.flatnonzero(reported)
        dist = np.linalg.norm(raw_xy[idx, None, :] - np.array(sorted(stuck))[None], axis=2).min(axis=1)
        is_stuck[idx] = dist <= STUCK_RADIUS_PX
    detected_raw = reported & ~is_stuck   # TrackNetV3's detections, less its stuck-point artifact
    detected = drop_spikes(detected_raw, raw_xy)
    clean_xy = np.where(detected[:, None], raw_xy, np.nan)
    present, filled_xy = fill_gaps(detected, clean_xy)
    xy = smooth(present, filled_xy)
    floor = np.full_like(xy, np.nan)
    if present.any():
        # Where the line of sight meets the floor: the shuttle's court position only when it's on the floor
        floor[present] = court.apply_h(H_image_to_court, xy[present])
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TRACK_FIELDS)
        for i in range(n_frames):
            w.writerow([i, int(detected[i]), int(present[i] and not detected[i]),
                        fmt(xy[i, 0], 2), fmt(xy[i, 1], 2), fmt(raw_xy[i, 0], 0), fmt(raw_xy[i, 1], 0),
                        fmt(floor[i, 0], 3), fmt(floor[i, 1], 3)])
    # Clips run on past the rally (players walking back, shuttle in hand), so the gate uses the span from
    # the first detection to the last; the whole-clip figure is reported alongside
    seen = np.flatnonzero(detected_raw)
    active_frames = int(seen[-1] - seen[0] + 1) if len(seen) else 0
    return {"frames": n_frames, "detected_raw": float(detected_raw.mean()),
            "visible_active": float(detected_raw[seen[0]:seen[-1] + 1].mean()) if len(seen) else 0.0,
            "active_frames": active_frames, "detected": float(detected.mean()), "present": float(present.mean()),
            "stuck_dropped": int(is_stuck.sum()), "spikes_dropped": int(detected_raw.sum() - detected.sum())}


def process():
    report = []
    for d in match_dirs():
        raw_dir = d / "tracks" / "raw"
        if not raw_dir.exists():
            log(f"- {d.name}: no raw tracks yet")
            continue
        H_inv = np.array(json.loads((d / "court.json").read_text())["H_image_to_court"])
        clips = [(r, raw_dir / r["file"].replace(".mp4", "_ball.csv")) for r in read_rows(d)]
        clips = [(r, raw) for r, raw in clips if raw.exists()]
        stuck = stuck_positions([raw for _, raw in clips])
        stats = []
        for r, raw in clips:
            n = int(r["end_frame"]) - int(r["start_frame"])
            s = process_clip(raw, n, H_inv, d / "tracks" / r["file"].replace(".mp4", ".csv"), stuck)
            stats.append(s)
            report.append({"match_id": d.name, "segment_id": r["segment_id"], "duration_s": r["duration_s"],
                           **{k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()},
                           "passes": s["visible_active"] >= GATE_VISIBILITY})
        if not stats:
            continue
        frames = sum(s["frames"] for s in stats)
        whole = sum(s["detected_raw"] * s["frames"] for s in stats) / frames
        active = sum(s["visible_active"] * s["active_frames"] for s in stats) / max(1, sum(s["active_frames"] for s in stats))
        passing = sum(s["visible_active"] >= GATE_VISIBILITY for s in stats)
        log(f"{d.name:32s} {len(stats):3d} tracks  visibility in play {active:.1%} (whole clips {whole:.1%})  "
            f"segments >= {GATE_VISIBILITY:.0%}: {passing}/{len(stats)}  "
            f"stuck positions {len(stuck)} ({sum(s['stuck_dropped'] for s in stats)} frames dropped)"
            f"{'  <- below 60%, drop this match' if active < DROP_VISIBILITY else ''}")
    if report:
        with open(DATA_DIR / "tracks_report.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(report[0]))
            w.writeheader()
            w.writerows(report)
        log(f"per-segment report -> {DATA_DIR / 'tracks_report.csv'}")


# === EYE CHECK ===
def read_track(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    num = lambda v: float(v) if v else np.nan
    xy = np.array([(num(r["x_px"]), num(r["y_px"])) for r in rows])
    return xy, np.array([r["detected"] == "1" for r in rows]), np.array([r["filled"] == "1" for r in rows])


def overlay(match, segment, trail=12, stills=8):
    d = SEG_DIR / match
    row = next(r for r in read_rows(d) if int(r["segment_id"]) == segment)
    xy, detected, filled = read_track(d / "tracks" / row["file"].replace(".mp4", ".csv"))
    fps = json.loads((d / "match.json").read_text())["fps"]
    contacts_path = d / "contacts" / row["file"].replace(".mp4", ".csv")
    contacts = {}  # frame -> (kind, number, position), from 1D if it has run
    if contacts_path.exists():
        with open(contacts_path, newline="") as f:
            for k, c in enumerate(csv.DictReader(f), 1):
                contacts[int(c["frame"])] = (c["kind"], k, (round(float(c["x_px"])), round(float(c["y_px"]))))
    cap = cv2.VideoCapture(str(d / row["file"]))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = d / "tracks" / f"overlay_{row['file']}"
    writer = subprocess.Popen([
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(out),
    ], stdin=subprocess.PIPE)
    still_at = set(np.linspace(0, len(xy) - 1, stills).astype(int))
    tiles = []
    for i in range(len(xy)):
        ok, frame = cap.read()
        if not ok:
            break
        pts = [(j, xy[j]) for j in range(max(0, i - trail), i + 1) if np.isfinite(xy[j]).all()]
        for (j0, p0), (j1, p1) in zip(pts, pts[1:]):
            if j1 - j0 == 1:
                cv2.line(frame, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), (0, 220, 255), 2, cv2.LINE_AA)
        if np.isfinite(xy[i]).all():
            color = (0, 255, 0) if detected[i] else (255, 160, 0)  # green detected, blue interpolated
            cv2.circle(frame, tuple(np.round(xy[i]).astype(int)), 7, color, 2, cv2.LINE_AA)
        state = "detected" if detected[i] else "interpolated" if filled[i] else "missing"
        if i in contacts:  # a detected contact: marked on its own frame only, so frame-stepping lands on it exactly
            kind, k, pos = contacts[i]
            cv2.circle(frame, pos, 16, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, f"{kind.upper()} {k}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
            state += f"  <- {kind} {k}"
        cv2.putText(frame, f"{match} #{segment}  frame {i}  {state}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        writer.stdin.write(frame.tobytes())
        if i in still_at:
            tiles.append(label(cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA), f"frame {i} {state}"))
    writer.stdin.close()
    writer.wait()
    cap.release()
    sheet(tiles, 4, out.with_suffix(".png"))
    log(f"overlay -> {out}  (stills -> {out.with_suffix('.png').name})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--full-size", action="store_true", help="pack the original clips, not 512x288 copies")
    p.add_argument("--match", action="append", help="only this match (repeatable)")
    sub.add_parser("ingest").add_argument("zip", type=Path)
    sub.add_parser("process")
    o = sub.add_parser("overlay")
    o.add_argument("--match", required=True)
    o.add_argument("--segment", type=int, required=True)
    args = ap.parse_args()
    if args.cmd == "pack":
        pack(args.full_size, args.match)
    elif args.cmd == "ingest":
        ingest(args.zip)
    elif args.cmd == "process":
        process()
    else:
        overlay(args.match, args.segment)
