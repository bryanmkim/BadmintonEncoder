"""Phase 1B — court homography.

For each match processed by video_prep.py: builds a clean reference image of the main camera view
(per-pixel median of frames across its clips, which removes the players), finds the painted lines,
fits the homography from court metres to image pixels, and checks it by projecting every line back.

  .venv/bin/python court_calibrate.py                          # every match with clips
  .venv/bin/python court_calibrate.py --match <id>
  .venv/bin/python court_calibrate.py --match <id> --click     # seed from 4 clicked corners instead of the search

Writes segments/<match_id>/court.json (homography, camera, per-line errors), court_reference.jpg,
court_mask.png and court_overlay.jpg, plus court_gate.jpg across all matches.
"""
import argparse
import csv
import json
import sys
from itertools import combinations

import cv2
import numpy as np

import court
from video_prep import DATA_DIR, SEG_DIR, label, log, now

LINE_KERNEL = 17               # top-hat size: keeps bright structures thinner than this (px)
TOPHAT_MIN = 30                # how much brighter than the local background a line pixel must be
MAX_LINE_SAT = 70              # painted lines are white; drops coloured boards and logos
HORIZONTAL_MAX_DEG = 25        # lines this close to horizontal are baseline / service line candidates
SEARCH_LINES = 10              # strongest detected lines of each orientation fed to the search
SEARCH_TAU = 10.0              # px cap on each model point's distance to a painted pixel
MIN_COURT_AREA = 0.08          # a hypothesis's projected court must cover this fraction of the frame
REFINE_BANDS = (8, 6, 5, 4, 4)  # px half-width around each projected line when fitting the painted line
MEASURE_BAND = 12              # px half-width when measuring the final error
MIN_LINE_PIXELS = 40
GATE_PX = 10.0                 # 1B gate: projected lines within this of the painted ones
MIN_VISIBLE_LINES = 10         # of 12; the far service area is seen through the net


def read_rows(match_dir):
    with open(match_dir / "segments.csv", newline="") as f:
        return list(csv.DictReader(f))


def read_frame(path, idx):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def reference_frame(match_dir, rows, n=41):
    """Per-pixel median of n frames spread across the clips: moving players and the shuttle vanish."""
    lengths = [int(r["end_frame"]) - int(r["start_frame"]) for r in rows]
    starts = np.cumsum([0] + lengths[:-1])
    frames = []
    for pos in np.linspace(0, sum(lengths) - 1, n).astype(int):
        k = int(np.searchsorted(starts, pos, side="right") - 1)
        frame = read_frame(match_dir / rows[k]["file"], int(pos - starts[k]))
        if frame is not None:
            frames.append(frame)
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def line_mask(img):
    """Painted lines: thin structures brighter than their surroundings and nearly colourless.

    Uses luminance rather than HSV value, because a red court has almost the value of white paint.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (LINE_KERNEL, LINE_KERNEL))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    return np.where((tophat > TOPHAT_MIN) & (sat < MAX_LINE_SAT), 255, 0).astype(np.uint8)


def mask_pixels(mask):
    ys, xs = np.nonzero(mask)
    return np.c_[xs, ys].astype(float)


# === INITIAL HOMOGRAPHY ===
def detect_lines(mask):
    """Hough segments merged into lines: dicts with unit normal n, offset c (n . p + c = 0) and total length."""
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=60, minLineLength=60, maxLineGap=6)
    segs = [] if segs is None else np.asarray(segs, float).reshape(-1, 4)  # (N, 1, 4) or (N, 4) by OpenCV version
    merged = []
    for x1, y1, x2, y2 in sorted(segs, key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1])):
        d = np.array([x2 - x1, y2 - y1])
        length = np.linalg.norm(d)
        n = np.array([-d[1], d[0]]) / length
        mid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        for m in merged:
            if abs(n @ m["n"]) > np.cos(np.radians(1.5)) and abs(m["n"] @ mid + m["c"]) < 5:
                m["len"] += length
                break
        else:
            merged.append({"n": n, "c": -n @ np.array([x1, y1]), "len": length})
    return merged


def intersect(l1, l2):
    p = np.cross([*l1["n"], l1["c"]], [*l2["n"], l2["c"]])
    return None if abs(p[2]) < 1e-9 else p[:2] / p[2]


def chamfer(H, model, dt):
    """Mean distance (capped) from each projected model point to the nearest painted pixel."""
    h, w = dt.shape
    q = H @ model
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = q[0] / q[2], q[1] / q[2]
    ok = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    cost = np.full(model.shape[1], SEARCH_TAU)
    cost[ok] = np.minimum(dt[v[ok].astype(int), u[ok].astype(int)], SEARCH_TAU)
    return float(cost.mean())


def auto_init(mask):
    """Automatic stand-in for clicking the corners.

    Two detected near-horizontal lines and two steep ones are matched to every pair of model lines of
    the same kind; their four intersections give a homography, and the one whose projected court lands
    best on painted pixels wins.
    """
    h, w = mask.shape
    dt = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    _, pts = court.sample_lines(3)
    model = np.c_[pts, np.ones(len(pts))].T
    lines = detect_lines(mask)
    cos_flat = np.cos(np.radians(HORIZONTAL_MAX_DEG))
    flat = sorted((l for l in lines if abs(l["n"][1]) > cos_flat), key=lambda l: -l["len"])[:SEARCH_LINES]
    steep = sorted((l for l in lines if abs(l["n"][1]) <= cos_flat), key=lambda l: -l["len"])[:SEARCH_LINES]
    flat.sort(key=lambda l: -(l["n"][0] * w / 2 + l["c"]) / l["n"][1])   # top to bottom at the image centre
    steep.sort(key=lambda l: -(l["n"][1] * h / 2 + l["c"]) / l["n"][0])  # left to right at mid-height
    X, Y = court.VERTICAL_X, court.HORIZONTAL_Y

    best_score, best_H, tried = np.inf, None, 0
    for i, j in combinations(range(len(flat)), 2):
        for k, l in combinations(range(len(steep)), 2):
            quad = [intersect(flat[i], steep[k]), intersect(flat[i], steep[l]),
                    intersect(flat[j], steep[l]), intersect(flat[j], steep[k])]
            if any(p is None for p in quad):
                continue
            quad = np.float32(quad)
            # The far edge must look shorter than the near edge, and the quad must be roughly on screen
            if np.linalg.norm(quad[1] - quad[0]) >= np.linalg.norm(quad[2] - quad[3]):
                continue
            if (np.abs(quad[:, 0] - w / 2) > w).any() or (np.abs(quad[:, 1] - h / 2) > h).any():
                continue
            if not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
                continue
            for a, b in combinations(range(len(Y)), 2):
                for c, d in combinations(range(len(X)), 2):
                    src = np.float32([(X[c], Y[a]), (X[d], Y[a]), (X[d], Y[b]), (X[c], Y[b])])
                    H = cv2.getPerspectiveTransform(src, quad)
                    # A near-degenerate quad squashes the whole court onto one painted line, which scores
                    # a perfect chamfer; the full court has to stay a convex, sizeable shape
                    corners = court.apply_h(H, court.CORNERS).astype(np.float32)
                    if not np.isfinite(corners).all() or not cv2.isContourConvex(corners.reshape(-1, 1, 2)):
                        continue
                    if cv2.contourArea(corners) < MIN_COURT_AREA * w * h:
                        continue
                    fl, fr, nr, nl = corners
                    # Seen from behind a baseline: left of right, far above near with real depth on both
                    # sides, and the far edge shorter than the near one
                    if fl[0] >= fr[0] or nl[0] >= nr[0] or min(nl[1] - fl[1], nr[1] - fr[1]) < 0.15 * h:
                        continue
                    if np.linalg.norm(fr - fl) >= np.linalg.norm(nr - nl):
                        continue
                    try:
                        court.camera_from_homography(H, w, h)  # something a real camera could see
                    except ValueError:
                        continue
                    score = chamfer(H, model, dt)
                    tried += 1
                    if score < best_score:
                        best_score, best_H = score, H
    log(f"    search: {len(flat)} flat + {len(steep)} steep lines, {tried} hypotheses, best chamfer {best_score:.2f} px")
    return best_H


def click_init(img):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Click the 4 outer corners: far-left, far-right, near-right, near-left (zoom first with the toolbar)")
    pts = plt.ginput(4, timeout=0)
    plt.close()
    if len(pts) != 4:
        sys.exit("need exactly 4 clicks")
    return cv2.getPerspectiveTransform(np.float32(court.CORNERS), np.float32(pts))


# === REFINEMENT AND ERRORS ===
def parallel_spacing(H, a, b):
    """Smallest image distance (px) from segment a-b to the nearest parallel court line."""
    a, b = np.array(a), np.array(b)
    axis = 1 if a[1] == b[1] else 0  # horizontal lines share Y, vertical ones share X
    pts = a + np.linspace(0.05, 0.95, 10)[:, None] * (b - a)
    proj = court.apply_h(H, pts)
    spacing = np.inf
    for v in (court.HORIZONTAL_Y if axis else court.VERTICAL_X):
        if v != a[axis]:
            other = pts.copy()
            other[:, axis] = v
            spacing = min(spacing, np.linalg.norm(court.apply_h(H, other) - proj, axis=1).min())
    return spacing


def fit_painted_line(H, a, b, pix, band):
    """Fit the painted line near the projection of model segment a-b: (point, unit direction, support, band).

    The band never reaches more than 45% of the way to the next parallel line: at the far end the
    baseline and long service line are only about 10 px apart.
    """
    band = min(band, 0.45 * parallel_spacing(H, a, b))
    pa, pb = court.apply_h(H, [a, b])
    d = pb - pa
    length = np.linalg.norm(d)
    d /= length
    n = np.array([-d[1], d[0]])
    rel = pix - pa
    along, across = rel @ d, rel @ n
    # Skip the ends, where crossing lines would pull the fit
    sel = (np.abs(across) < band) & (along > 0.04 * length) & (along < 0.96 * length)
    if sel.sum() < MIN_LINE_PIXELS:
        return None
    vx, vy, x0, y0 = cv2.fitLine(pix[sel].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0]), np.array([vx, vy]), int(sel.sum()), float(band)


def refine(H, pix):
    """Snap to the painted lines: fit each line's centre near its projection, move the projected points
    onto it, re-solve the homography, and repeat with a narrower band."""
    for band in REFINE_BANDS:
        src, dst = [], []
        for a, b in court.LINES.values():
            fit = fit_painted_line(H, a, b, pix, band)
            if fit is None:
                continue
            p0, d = fit[:2]
            n = np.array([-d[1], d[0]])
            model_pts = np.array(a) + np.linspace(0.05, 0.95, 20)[:, None] * (np.array(b) - np.array(a))
            proj = court.apply_h(H, model_pts)
            src.append(model_pts)
            dst.append(proj - ((proj - p0) @ n)[:, None] * n)
        if len(src) < 4:
            break
        H_new, _ = cv2.findHomography(np.concatenate(src), np.concatenate(dst), 0)
        if H_new is None:
            break
        H = H_new
    return H


def line_errors(H, pix, shape):
    """Per painted line: distance from its projection to the fitted painted line (px), plus chamfer."""
    h, w = shape
    dt = None
    out = {}
    for name, (a, b) in court.LINES.items():
        fit = fit_painted_line(H, a, b, pix, MEASURE_BAND)
        if fit is None:
            out[name] = {"visible": False}
            continue
        p0, d, support, band = fit
        n = np.array([-d[1], d[0]])
        proj = court.apply_h(H, np.array(a) + np.linspace(0.05, 0.95, 30)[:, None] * (np.array(b) - np.array(a)))
        on = (proj[:, 0] >= 0) & (proj[:, 0] < w) & (proj[:, 1] >= 0) & (proj[:, 1] < h)
        dist = np.abs((proj[on] - p0) @ n)
        out[name] = {"visible": True, "mean_px": round(float(dist.mean()), 2), "max_px": round(float(dist.max()), 2),
                     "support_px": support, "band_px": round(band, 1)}
    return out


def worst(errors):
    visible = [e["max_px"] for e in errors.values() if e["visible"]]
    return len(visible), (max(visible) if visible else float("inf"))


def stability(match_dir, rows, H, windows=4):
    """Re-measure the same homography on references built from each quarter of the match."""
    out = []
    for chunk in np.array_split(np.arange(len(rows)), windows):
        sub = [rows[i] for i in chunk]
        ref = reference_frame(match_dir, sub, 15)
        n_visible, max_px = worst(line_errors(H, mask_pixels(line_mask(ref)), ref.shape[:2]))
        out.append({"from_s": float(sub[0]["start_s"]), "to_s": float(sub[-1]["end_s"]),
                    "lines_visible": n_visible, "max_px": max_px})
    return out


# === OUTPUT ===
def draw_overlay(img, H, K, R, t, title):
    out = img.copy()
    for a, b in court.LINES.values():
        pa, pb = np.round(court.apply_h(H, [a, b])).astype(int)
        cv2.line(out, tuple(pa), tuple(pb), (0, 255, 0), 1, cv2.LINE_AA)
    spots = list(court.apply_h(H, court.CORNERS))
    names = ["far-left", "far-right", "near-right", "near-left"]
    if K is not None:
        for poly in court.NET:
            pts = np.round(court.project(K, R, t, poly)).astype(np.int32)
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, (0, 220, 255), 1, cv2.LINE_AA)
        spots.append(court.project(K, R, t, [(0, 0, court.NET_MID_H)])[0])
        names.append("net tape centre")
    # 2x zooms (nearest-neighbour, so single-pixel offsets stay visible) of the corners and net centre
    tiles = []
    for (x, y), name in zip(spots, names):
        x0 = int(np.clip(x - 64, 0, img.shape[1] - 128))
        y0 = int(np.clip(y - 45, 0, img.shape[0] - 90))
        tile = cv2.resize(out[y0:y0 + 90, x0:x0 + 128], (256, 180), interpolation=cv2.INTER_NEAREST)
        tiles.append(label(tile, name))
    strip = np.hstack(tiles)
    strip = cv2.resize(strip, (img.shape[1], int(strip.shape[0] * img.shape[1] / strip.shape[1])))
    return np.vstack([label(out, title), strip])


def calibrate(match_dir, click=False):
    rows = read_rows(match_dir)
    ref = reference_frame(match_dir, rows)
    h, w = ref.shape[:2]
    mask = line_mask(ref)
    cv2.imwrite(str(match_dir / "court_reference.jpg"), ref)
    cv2.imwrite(str(match_dir / "court_mask.png"), mask)
    pix = mask_pixels(mask)

    H0 = click_init(ref) if click else auto_init(mask)
    if H0 is None:
        log("    no court found; try --click")
        return None
    H = refine(H0, pix)
    errors = line_errors(H, pix, (h, w))
    n_visible, max_px = worst(errors)
    try:
        K, R, t = court.camera_from_homography(H, w, h)
        center = court.camera_center(R, t)
    except ValueError:
        K = R = t = center = None
    stab = stability(match_dir, rows, H)
    passed = n_visible >= MIN_VISIBLE_LINES and max_px <= GATE_PX and all(s["max_px"] <= GATE_PX for s in stab)

    title = f"{match_dir.name}: {n_visible}/12 lines, worst {max_px:.1f}px {'PASS' if passed else 'FAIL'}"
    cv2.imwrite(str(match_dir / "court_overlay.jpg"), draw_overlay(ref, H, K, R, t, title))
    (match_dir / "court.json").write_text(json.dumps({
        "image_size": [w, h],
        "court_frame": "metres; X across (+ right), Y along (-6.7 far baseline .. +6.7 near), Z up",
        "H_court_to_image": H.tolist(),
        "H_image_to_court": np.linalg.inv(H).tolist(),
        "camera": None if K is None else {"K": K.tolist(), "R": R.tolist(), "t": t.tolist(),
                                          "focal_px": round(float(K[0, 0]), 1), "center_m": np.round(center, 2).tolist()},
        "line_errors": errors, "lines_visible": n_visible, "max_error_px": max_px,
        "stability": stab, "gate_px": GATE_PX, "passed": passed,
        "method": "click" if click else "auto", "calibrated_at": now(),
    }, indent=2))
    camera = "no camera" if K is None else f"camera at {np.round(center, 1).tolist()} m, f {K[0, 0]:.0f} px"
    log(f"    {n_visible}/12 lines visible, worst {max_px:.2f} px, quarters {[s['max_px'] for s in stab]}, "
        f"{camera} -> {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", help="match_id")
    ap.add_argument("--click", action="store_true", help="seed from 4 clicked corners")
    args = ap.parse_args()
    dirs = [d for d in sorted(SEG_DIR.iterdir()) if (d / "segments.csv").exists() and args.match in (None, d.name)]
    if not dirs:
        sys.exit(f"no processed match {args.match!r} in {SEG_DIR}")
    results = {}
    for d in dirs:
        log(f"> {d.name}")
        results[d.name] = calibrate(d, click=args.click)
    if len(dirs) > 1:
        overlays = [cv2.resize(cv2.imread(str(d / "court_overlay.jpg")), (960, 675)) for d in dirs if (d / "court_overlay.jpg").exists()]
        grid = [np.hstack(overlays[i:i + 2] + [np.zeros_like(overlays[0])] * (2 - len(overlays[i:i + 2]))) for i in range(0, len(overlays), 2)]
        cv2.imwrite(str(DATA_DIR / "court_gate.jpg"), np.vstack(grid))
    log(f"gate: {sum(bool(v) for v in results.values())}/{len(results)} matches pass")
