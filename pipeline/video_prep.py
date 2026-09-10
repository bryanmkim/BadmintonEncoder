"""Phase 1A — video prep.

Downloads each broadcast match in matches.csv at 720p, finds camera cuts, keeps only the main
wide-angle court view as segment clips, then deletes the full download.

  .venv/bin/python video_prep.py                   # every match not processed yet
  .venv/bin/python video_prep.py --match <id>      # one match (--force to redo, --keep-raw to keep the download)
  .venv/bin/python video_prep.py --resegment       # re-apply segment rules from the cached analysis, no download
  .venv/bin/python video_prep.py --spot-check 10   # 1A gate: contact sheet of random kept segments

Output goes to $BADMINTON_DATA_DIR (default ../data):
  segments/<match_id>/seg_0001.mp4 ...  kept clips (H.264 + AAC)
  segments/<match_id>/segments.csv      where every clip sits in the source video
  segments/<match_id>/match.json        source id, format, thresholds, totals
  segments/<match_id>/analysis.npz      per-frame features, used by --resegment
  segments/<match_id>/timeline.png      per-frame diagnostics
  segments/<match_id>/rejected.png      sample of dropped footage, to check nothing live was lost
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("BADMINTON_DATA_DIR", PIPELINE_DIR.parent / "data")).resolve()
RAW_DIR = DATA_DIR / "raw"
SEG_DIR = DATA_DIR / "segments"
MATCHES_CSV = PIPELINE_DIR / "matches.csv"

# Highest-bitrate 720p H.264 plus AAC audio (racket-hit sounds may help contact detection later)
YTDLP_FORMAT = "bv*[height<=720][vcodec^=avc1]+ba[ext=m4a]"

ANALYSIS_SIZE = (160, 90)   # frames are decoded at this size for every per-frame feature
THUMB_SIZE = (32, 18)       # main-view comparison thumbnail
CUT_TV = 0.35               # colour-histogram jump (total variation, 0–1) that counts as a hard cut
SMOOTH_S = 0.5              # majority filter on the per-frame main-view decision
MIN_SEGMENT_S = 4.0         # shorter main-view runs can't hold a rally
EDGE_TRIM_S = 0.2           # frames dropped at each boundary, where transitions bleed in
# A kept segment's median distance to the main view. On the first 5 matches the main view stayed <= 15
# even under score graphics or changed LED boards, while other wide cameras that slipped under the
# per-frame threshold sat at >= 24.
MAX_SEGMENT_VIEW_DIST = 18.0

SEGMENT_FIELDS = ["segment_id", "file", "start_frame", "end_frame", "start_s", "end_s", "duration_s", "view_dist", "motion"]


def log(*args):
    print(*args, flush=True)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_matches():
    with open(MATCHES_CSV, newline="") as f:
        return list(csv.DictReader(f))


# === DOWNLOAD ===
def download(match):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stem = RAW_DIR / match["match_id"]
    subprocess.run([
        sys.executable, "-m", "yt_dlp", "--js-runtimes", "node",
        "-f", YTDLP_FORMAT, "-S", "tbr", "--merge-output-format", "mp4", "-N", "4",
        "--write-info-json", "--no-progress", "-o", f"{stem}.%(ext)s",
        f"https://www.youtube.com/watch?v={match['youtube_id']}",
    ], check=True)
    return stem.with_suffix(".mp4"), json.loads(stem.with_suffix(".info.json").read_text())


def probe(path):
    out = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,width,height:format=duration", "-of", "json", str(path),
    ], capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    num, den = map(int, info["streams"][0]["avg_frame_rate"].split("/"))
    return num / den, float(info["format"]["duration"]), info["streams"][0]["width"], info["streams"][0]["height"]


# === ANALYSIS ===
def decode(path, size):
    """Yield every frame, scaled down by ffmpeg (much faster than full-size decode + resize)."""
    w, h = size
    n = w * h * 3
    proc = subprocess.Popen([
        "ffmpeg", "-v", "error", "-i", str(path), "-an", "-sn",
        "-vf", f"scale={w}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ], stdout=subprocess.PIPE, bufsize=n * 64)
    while len(buf := proc.stdout.read(n)) == n:
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    proc.stdout.close()
    proc.wait()


def analyze(path, fps, duration):
    """Per frame: colour-histogram jump from the previous frame, motion, and a small thumbnail."""
    tv, motion, thumbs = [], [], []
    prev_hist = prev_gray = None
    report_every = int(fps * 600)
    for i, frame in enumerate(decode(path, ANALYSIS_SIZE)):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
        hist /= hist.sum()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
        tv.append(0.0 if prev_hist is None else 0.5 * float(np.abs(hist - prev_hist).sum()))
        motion.append(0.0 if prev_gray is None else float(np.abs(gray - prev_gray).mean()))
        thumbs.append(cv2.resize(frame, THUMB_SIZE, interpolation=cv2.INTER_AREA))
        prev_hist, prev_gray = hist, gray
        if i and i % report_every == 0:
            log(f"    analysed {i / fps / 60:.0f} / {duration / 60:.0f} min")
    return np.array(tv, np.float32), np.array(motion, np.float32), np.stack(thumbs)


def main_view_distance(thumbs, fps):
    """Distance of every frame to the match's main camera view.

    The main wide-angle camera is fixed and on screen more than any other shot, so its frames form
    the densest large cluster: take the 1 fps sample whose k-th nearest neighbour is closest, and
    use the median of those neighbours as the reference view.
    """
    flat = thumbs.reshape(len(thumbs), -1)
    s = flat[:: max(1, round(fps))].astype(np.float32)
    k = max(5, len(s) // 5)  # assumes the main view fills at least a fifth of the broadcast
    sq = (s ** 2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2 * s @ s.T
    center = np.argmin(np.partition(d2, k, axis=1)[:, k])
    reference = np.median(s[np.argpartition(d2[center], k)[:k]], axis=0)
    dist = np.concatenate([
        np.abs(flat[i:i + 20000].astype(np.float32) - reference).mean(1) for i in range(0, len(flat), 20000)
    ])
    return dist, reference.reshape(THUMB_SIZE[1], THUMB_SIZE[0], 3).astype(np.uint8)


def otsu(values, bins=256):
    """Threshold that best splits values into two groups (main view vs everything else)."""
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    s0 = np.cumsum(hist * centers)
    m0 = s0 / np.maximum(w0, 1)
    m1 = (s0[-1] - s0) / np.maximum(w1, 1)
    return float(centers[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def find_segments(tv, dist, threshold, fps):
    """Runs of main-view frames, split at hard cuts, trimmed and filtered. Returns [(start, end)) frames."""
    win = int(SMOOTH_S * fps) | 1
    raw_main = dist < threshold
    is_main = np.lib.stride_tricks.sliding_window_view(np.pad(raw_main, win // 2, mode="edge"), win).mean(1) > 0.5
    cut = tv > CUT_TV
    runs, start = [], None
    for i in range(len(is_main) + 1):
        if start is not None and (i == len(is_main) or not is_main[i] or cut[i]):
            runs.append((start, i))
            start = None
        if i < len(is_main) and is_main[i] and start is None:
            start = i
    trim, min_len = round(EDGE_TRIM_S * fps), round(MIN_SEGMENT_S * fps)
    segments = [(s + trim, e - trim) for s, e in runs if e - s - 2 * trim >= min_len]
    # A whole run from a similar-looking second camera can sit just under the per-frame threshold
    return [(s, e) for s, e in segments if np.median(dist[s:e]) <= MAX_SEGMENT_VIEW_DIST]


def plan_segments(tv, dist, fps):
    threshold = otsu(np.clip(dist, 0, np.percentile(dist, 99)))
    return threshold, find_segments(tv, dist, threshold, fps)


# === OUTPUT ===
def segment_rows(segments, dist, motion, fps):
    return [{
        "segment_id": n, "file": f"seg_{n:04d}.mp4", "start_frame": s, "end_frame": e,
        "start_s": round(s / fps, 3), "end_s": round(e / fps, 3), "duration_s": round((e - s) / fps, 3),
        "view_dist": round(float(dist[s:e].mean()), 2), "motion": round(float(motion[s:e].mean()), 3),
    } for n, (s, e) in enumerate(segments, 1)]


def write_segments_csv(out_dir, rows):
    with open(out_dir / "segments.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SEGMENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def totals(rows, duration, threshold):
    kept = sum(r["duration_s"] for r in rows)
    return {
        "main_view_threshold": threshold, "cut_tv": CUT_TV, "min_segment_s": MIN_SEGMENT_S,
        "max_segment_view_dist": MAX_SEGMENT_VIEW_DIST, "n_segments": len(rows),
        "kept_s": round(kept, 1), "kept_fraction": round(kept / duration, 3),
    }


def cut_clip(raw, out, start_s, dur_s):
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-ss", f"{start_s:.3f}", "-i", str(raw), "-t", f"{dur_s:.3f}",
        "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out),
    ], check=True)


def grab(path, frame_idx, size=(320, 180)):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if ok else np.zeros((size[1], size[0], 3), np.uint8)


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def sheet(tiles, cols, out):
    blank = np.zeros_like(tiles[0])
    tiles = tiles + [blank] * (-len(tiles) % cols)
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out), np.vstack(rows))


def plot_timeline(out, tv, dist, motion, threshold, segments, fps, reference):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(dist)) / fps / 60
    fig, axes = plt.subplots(3, 1, figsize=(18, 9), gridspec_kw={"height_ratios": [2, 1, 1]})
    ax = axes[0]
    ax.plot(t, dist, lw=0.3, color="#475569")
    ax.axhline(threshold, color="#ef4444", lw=1, label=f"main-view threshold {threshold:.1f}")
    ax.axhline(MAX_SEGMENT_VIEW_DIST, color="#f59e0b", lw=1, ls="--", label=f"max segment median {MAX_SEGMENT_VIEW_DIST:.0f}")
    for s, e in segments:
        ax.axvspan(s / fps / 60, e / fps / 60, color="#22c55e", alpha=0.25, lw=0)
    ax.set_ylabel("distance to main view")
    ax.set_ylim(0, np.percentile(dist, 99.5) * 1.1)
    ax.legend(loc="upper right")
    ax.set_title("green = kept segments")
    axes[1].plot(t, tv, lw=0.3, color="#475569")
    axes[1].axhline(CUT_TV, color="#ef4444", lw=1)
    axes[1].set_ylabel("histogram jump")
    axes[2].plot(t, motion, lw=0.3, color="#475569")
    axes[2].set_ylabel("motion")
    axes[2].set_xlabel("minutes")
    inset = axes[0].inset_axes([0.0, 0.62, 0.12, 0.38])
    inset.imshow(cv2.cvtColor(cv2.resize(reference, (160, 90), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB))
    inset.set_axis_off()
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def process(match, force=False, keep_raw=False):
    out_dir = SEG_DIR / match["match_id"]
    if (out_dir / "match.json").exists() and not force:
        log(f"= {match['match_id']}: already processed")
        return
    log(f"> {match['match_id']}: downloading")
    raw, info = download(match)
    fps, duration, width, height = probe(raw)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = out_dir / "analysis.npz"
    if cache.exists() and not force:
        z = np.load(cache)
        tv, motion, dist, reference = z["tv"], z["motion"], z["dist"], z["reference"]
    else:
        log(f"  analysing {duration / 60:.0f} min at {fps:.2f} fps")
        tv, motion, thumbs = analyze(raw, fps, duration)
        dist, reference = main_view_distance(thumbs, fps)
        np.savez_compressed(cache, tv=tv, motion=motion, dist=dist, reference=reference)

    threshold, segments = plan_segments(tv, dist, fps)
    rows = segment_rows(segments, dist, motion, fps)
    log(f"  threshold {threshold:.1f}, {int((tv > CUT_TV).sum())} hard cuts, {len(rows)} segments, "
        f"{sum(r['duration_s'] for r in rows) / 60:.1f} of {duration / 60:.1f} min kept")

    for old in list(out_dir.glob("seg_*.mp4")):
        old.unlink()
    log(f"  cutting {len(rows)} clips")
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda r: cut_clip(raw, out_dir / r["file"], r["start_s"], r["duration_s"]), rows))
    write_segments_csv(out_dir, rows)

    plot_timeline(out_dir / "timeline.png", tv, dist, motion, threshold, segments, fps, reference)
    kept_mask = np.zeros(len(dist), bool)
    for s, e in segments:
        kept_mask[s:e] = True
    dropped = np.flatnonzero(~kept_mask)
    picks = sorted(random.Random(0).sample(list(dropped), min(24, len(dropped))))
    sheet([label(grab(raw, i), f"{i / fps / 60:.1f} min  dist {dist[i]:.1f}") for i in picks], 6, out_dir / "rejected.png")

    (out_dir / "match.json").write_text(json.dumps({
        **match, "title": info.get("title"), "format_id": info.get("format_id"),
        "fps": fps, "width": width, "height": height, "duration_s": round(duration, 2),
        **totals(rows, duration, threshold), "processed_at": now(),
    }, indent=2))
    if not keep_raw:
        raw.unlink()
        raw.with_suffix(".info.json").unlink(missing_ok=True)
    log(f"  done -> {out_dir}")


def resegment(match):
    """Re-apply the segment rules to a processed match from its cached analysis, without the source video.

    Only works when every new segment is an existing clip, i.e. a rule change that drops segments.
    Anything that needs new clips has to be rebuilt with --force.
    """
    out_dir = SEG_DIR / match["match_id"]
    if not (out_dir / "match.json").exists():
        log(f"- {match['match_id']}: not processed yet")
        return
    meta = json.loads((out_dir / "match.json").read_text())
    z = np.load(out_dir / "analysis.npz")
    tv, motion, dist, fps = z["tv"], z["motion"], z["dist"], meta["fps"]
    threshold, segments = plan_segments(tv, dist, fps)
    with open(out_dir / "segments.csv", newline="") as f:
        existing = {(int(r["start_frame"]), int(r["end_frame"])): r["file"] for r in csv.DictReader(f)}
    if missing := [s for s in segments if s not in existing]:
        sys.exit(f"{match['match_id']}: {len(missing)} segments aren't existing clips; rebuild with --force")

    rows = segment_rows(segments, dist, motion, fps)
    renames = {existing[(r["start_frame"], r["end_frame"])]: r["file"] for r in rows}
    for clip in list(out_dir.glob("seg_*.mp4")):
        if clip.name in renames:
            clip.rename(out_dir / f"tmp_{renames[clip.name]}")
        else:
            clip.unlink()
    for clip in list(out_dir.glob("tmp_seg_*.mp4")):
        clip.rename(out_dir / clip.name.removeprefix("tmp_"))
    write_segments_csv(out_dir, rows)
    plot_timeline(out_dir / "timeline.png", tv, dist, motion, threshold, segments, fps, z["reference"])
    meta.update(totals(rows, meta["duration_s"], threshold), resegmented_at=now())
    (out_dir / "match.json").write_text(json.dumps(meta, indent=2))
    log(f"= {match['match_id']}: {len(existing)} -> {len(rows)} segments")


def spot_check(n, seed):
    clips = [(d, row) for d in sorted(SEG_DIR.iterdir()) if (d / "segments.csv").exists()
             for row in csv.DictReader(open(d / "segments.csv"))]
    tiles = []
    for d, row in random.Random(seed).sample(clips, min(n, len(clips))):
        count = int(row["end_frame"]) - int(row["start_frame"])
        for frac in (0.1, 0.5, 0.9):
            tiles.append(label(grab(d / row["file"], int(frac * count)),
                               f"{d.name[:22]} #{row['segment_id']} {row['duration_s']}s @{frac:.0%}"))
    out = DATA_DIR / "spot_check.png"
    sheet(tiles, 3, out)
    log(f"spot check of {len(tiles) // 3} segments -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", help="match_id from matches.csv")
    ap.add_argument("--force", action="store_true", help="re-process even if already done")
    ap.add_argument("--keep-raw", action="store_true", help="keep the full download")
    ap.add_argument("--resegment", action="store_true", help="re-apply segment rules from cached analysis")
    ap.add_argument("--spot-check", type=int, metavar="N", help="contact sheet of N random segments")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.spot_check:
        spot_check(args.spot_check, args.seed)
        sys.exit()
    matches = [m for m in read_matches() if args.match in (None, m["match_id"])]
    if not matches:
        sys.exit(f"no match {args.match!r} in {MATCHES_CSV}")
    for m in matches:
        if args.resegment:
            resegment(m)
        else:
            process(m, force=args.force, keep_raw=args.keep_raw)
