"""Phase 1D — contact-frame detection.

Between two hits the shuttle follows one smooth curve on screen, and a hit starts a new one. Each clip's
track is split into the flights that explain it best, and a cut where the velocity jumps is a contact.

  .venv/bin/python contact_detect.py labels                           # fetch ShuttleSet's hit frames for matches that have them
  .venv/bin/python contact_detect.py detect                           # every processed track -> segments/<match>/contacts/
  .venv/bin/python contact_detect.py evaluate                         # recall and precision against ShuttleSet's hit frames
  .venv/bin/python contact_detect.py plot --match <id> --segment <n>  # one clip's track, flights, contacts and labels over time

detect and evaluate take --match <id> to limit them to one match.
"""
import argparse
import csv
import json
import subprocess
from collections import defaultdict

import numpy as np

import court
from shuttle_track import match_dirs, read_rows
from video_prep import DATA_DIR, SEG_DIR, log, read_matches

LABEL_DIR = DATA_DIR / "shuttleset"
SHUTTLESET_URL = "https://raw.githubusercontent.com/wywyWang/CoachAI-Projects/main/ShuttleSet/set"

# Frame counts assume 30 fps and pixel speeds 1280x720, like every match so far
DECAY = (2, 4, 8, 16, 32)  # frames: how fast drag slows the shuttle after a hit; every flight tries each
MIN_POINTS = 6       # detections a flight needs; a flight has 4 terms per axis, so fewer fit anything
MIN_FLIGHT = 11      # frames between contacts (0.37 s): ShuttleSet's 1st-percentile gap between hits. At 8,
                     # a flight the model couldn't fit was cut into 8-frame pieces (887 hand-checked hits)
MAX_FLIGHT = 150     # frames (5 s) one flight may span, gaps included
MAX_BRIDGE = 20      # frames without a detection that end a piece of track: the shuttle left the frame or was lost
PENALTY = 1000.0     # px²: the squared residual a new flight has to remove to be worth adding. False cuts
                     # removed a median 1,900 px², real ones 13,700; at 400 most false cuts got through
MIN_KINK = 3.0       # px/frame: a smaller velocity jump at a cut is the cubic running out of shape, not a hit
STILL = 2.0          # px/frame: a slower flight is the shuttle held, carried or lying on the floor
# A contact on the floor ends the rally. The camera looks down from behind the near baseline, so anything in
# the air maps onto the floor further from it than it is: a racket contact maps toward the net or beyond,
# and only a shuttle on the floor maps to where it really is. On 280 hand-checked hits, 95% of real hits
# mapped to Y <= 0.6 m.
FLOOR_Y = 4.0        # m: a contact mapping this deep into the near half is on the floor. At 3.0, near-player
                     # shots hit low around mid-court (mapping to 3.2-3.8 m) ended rallies early; 4.0 was best
                     # on the ShuttleSet dev match (recall 88.3 -> 90.1%) and left the hand review unchanged
FLOOR_SETTLE = 0.5   # m: ...as is one whose next detections stay this close to it on the floor, inside the court
SETTLE_POINTS = 7    # detections checked for that
COURT_BOX = (4.0, 8.5)  # m: |X|, |Y| a floor contact can be at: the court (3.05, 6.7) plus a margin for shots out

# Evaluation
LEAD, TAIL = 15, 45        # frames before a rally's first labelled hit and after its last that count as the rally
LABEL_LAG = 1              # frames ShuttleSet's hit frames run behind 1D's. Detected minus labelled peaked at -1
                           # and -2 on all three matches; +1 was best on the dev match. On a filmstrip of 6 hits the
                           # racket met the shuttle at or just after 1D's frame, and the labelled one was the
                           # follow-through, so the labels are moved rather than the detections
EDGE = 5                   # frames: a labelled hit this close to a clip's start or end can't show as a turn
TOLERANCES = (1, 2, 3, 5)  # frames; ±2 is the CoachAI challenge's rule

EVENT_FIELDS = ["frame", "kind", "x_px", "y_px", "floor_x_m", "floor_y_m", "speed_in", "speed_out", "kink"]
TYPES = {  # ShuttleSet's stroke types
    "發短球": "short service", "發長球": "long service", "放小球": "net shot", "擋小球": "return net",
    "殺球": "smash", "點扣": "wrist smash", "挑球": "lob", "防守回挑": "defensive lob", "長球": "clear",
    "平球": "drive", "小平球": "driven flight", "後場抽平球": "back-court drive", "切球": "drop",
    "過度切球": "passive drop", "推球": "push", "撲球": "rush", "防守回抽": "defensive drive",
    "勾球": "cross-court net shot", "未知球種": "unknown",
}


# === FLIGHTS ===
def read_detections(path):
    """A processed 1C track -> (clip length in frames, frames with a detection, their raw positions).
    Raw rather than smoothed: smoothing rounds off exactly the turn a contact makes."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    seen = [r for r in rows if r["detected"] == "1"]
    return (len(rows), np.array([int(r["frame"]) for r in seen], int),
            np.array([(float(r["raw_x_px"]), float(r["raw_y_px"])) for r in seen]).reshape(-1, 2))


def pieces(t):
    """Detection index ranges [a, b), split wherever the shuttle is lost for more than MAX_BRIDGE frames."""
    edges = np.r_[0, np.flatnonzero(np.diff(t) > MAX_BRIDGE + 1) + 1, len(t)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b - a >= MIN_POINTS]


def basis(frames, t0, decay):
    """A flight's terms on each image axis: where it starts, the slowdown after the hit, then drift and gravity.
    Under quadratic drag a shuttle travels ln(1 + t / decay), decay being shorter the harder it was hit; a
    cubic can't bend that sharply, and cut such flights in two. The linear and squared terms take up gravity,
    the slow end of the flight and perspective."""
    dt = np.asarray(frames, float) - t0
    s = dt / 30  # seconds, which keeps the normal equations well conditioned
    return np.stack([np.ones_like(dt), np.log1p(dt / decay), s, s * s], axis=-1)


def flight_costs(t, xy):
    """costs[i][m]: squared residual (px²) of the best flight through detections i..i+m, over every DECAY,
    for each flight that starts at i and spans at most MAX_FLIGHT frames; inf where there are too few
    detections. The normal equations are accumulated along each start, so all of its ends are solved in
    one batch."""
    costs = []
    for i in range(len(t)):
        end = np.searchsorted(t, t[i] + MAX_FLIGHT, side="right")
        d = xy[i:end] - xy[i]
        P = np.stack([basis(t[i:end], t[i], k) for k in DECAY])  # (decays, points, terms)
        G = np.cumsum(P[..., :, None] * P[..., None, :], axis=1)
        H = np.cumsum(P[..., :, None] * d[None, :, None, :], axis=1)
        S = np.cumsum((d ** 2).sum(1))
        c = np.full(end - i, np.inf)
        ok = np.arange(1, end - i + 1) >= MIN_POINTS
        if ok.any():
            beta = np.linalg.solve(G[:, ok], H[:, ok])
            # residual = S - H.beta, so the best decay is the one with the largest H.beta
            c[ok] = np.maximum(S[ok] - (H[:, ok] * beta).sum((2, 3)).max(0), 0)
        costs.append(c)
    return costs


def split_flights(t, xy):
    """Cut one piece of track into flights, minimising the total residual plus PENALTY per flight.
    Neighbouring flights share the detection at their cut, which keeps the path continuous through a hit.
    Returns the detection indices of the cuts, first and last included."""
    n = len(t)
    costs = flight_costs(t, xy)
    best, prev = np.full(n, np.inf), np.zeros(n, int)
    best[0] = 0.0
    for i in range(n - 1):
        if not np.isfinite(best[i]):
            continue
        j = i + np.arange(len(costs[i]))
        # Only the first and last flight may be short: the clip or the track cuts them off
        ok = (t[j] - t[i] >= MIN_FLIGHT) | (i == 0) | (j == n - 1)
        total = np.where(ok, best[i] + costs[i] + PENALTY, np.inf)
        better = total < best[j]
        best[j[better]], prev[j[better]] = total[better], i
    knots = [n - 1]
    while knots[-1] > 0:
        knots.append(prev[knots[-1]])
    return knots[::-1]


def fit(t, xy, i, j):
    """The best flight through detections i..j: (start frame, decay, coefficients of shape (4, 2))."""
    best = None
    for k in DECAY:
        A = basis(t[i:j + 1], t[i], k)
        coef, *_ = np.linalg.lstsq(A, xy[i:j + 1], rcond=None)
        residual = float(((A @ coef - xy[i:j + 1]) ** 2).sum())
        if best is None or residual < best[0]:
            best = residual, (t[i], k, coef)
    return best[1]


def position(flight, frames):
    t0, decay, coef = flight
    return basis(frames, t0, decay) @ coef


def velocity(flight, frames):
    """px/frame."""
    t0, decay, coef = flight
    dt = np.asarray(frames, float) - t0
    return np.stack([np.zeros_like(dt), 1 / (decay + dt), np.full_like(dt, 1 / 30), 2 * dt / 900], axis=-1) @ coef


def on_floor(H, pos, after):
    """Whether a contact at image point `pos`, followed by detections `after`, is the shuttle meeting the floor."""
    X, Y = court.apply_h(H, [pos])[0]
    fp = court.apply_h(H, after[:SETTLE_POINTS])
    settled = len(fp) >= 5 and np.linalg.norm(fp - fp[0], axis=1).max() <= FLOOR_SETTLE
    return Y > FLOOR_Y or (abs(X) <= COURT_BOX[0] and abs(Y) <= COURT_BOX[1] and settled)


def end_rally(events):
    """After the first hit, the first floor contact or landing ends the rally: a floor contact becomes the
    landing, and every hit after it (bounces, pick-ups, the shuttle knocked back to the server) is marked
    after_rally."""
    started = ended = False
    for e in events:
        floor = e.pop("floor")
        if ended:
            if e["kind"] == "hit":
                e["kind"] = "after_rally"
        elif started and (e["kind"] == "landing" or floor):
            e["kind"], ended = "landing", True
        elif e["kind"] == "hit":
            started = True
    return events


def detect_clip(t, xy, H):
    """Contacts ("hit"), landings and after_rally hits in one clip, plus every fitted flight as
    (flight, first frame, last frame). H maps image to court."""
    events, flights = [], []
    for a, b in pieces(t):
        pt, pxy = t[a:b], xy[a:b]
        knots = split_flights(pt, pxy)
        spans = list(zip(knots[:-1], knots[1:]))
        fl = [fit(pt, pxy, i, j) for i, j in spans]
        # Top speed, not typical speed: drag slows a hit shuttle to a crawl well before the next hit
        top = [float(np.linalg.norm(velocity(f, pt[i:j + 1]), axis=1).max()) for f, (i, j) in zip(fl, spans)]
        flights += [(f, pt[i], pt[j]) for f, (i, j) in zip(fl, spans)]
        for k in range(1, len(knots) - 1):
            frame = pt[knots[k]]
            v_in, v_out = velocity(fl[k - 1], [frame])[0], velocity(fl[k], [frame])[0]
            kink = float(np.linalg.norm(v_out - v_in))
            if kink < MIN_KINK:
                continue
            # A still shuttle that starts moving was hit (a serve, or a pick-up after the rally);
            # a moving one that stops has landed
            kind = "hit" if top[k] > STILL else "landing" if top[k - 1] > STILL else None
            if kind:
                x, y = (position(fl[k - 1], [frame])[0] + position(fl[k], [frame])[0]) / 2
                X, Y = court.apply_h(H, [(x, y)])[0]  # the true court position only if it's on the floor
                events.append({"frame": int(frame), "kind": kind, "x_px": x, "y_px": y,
                               "floor_x_m": float(X), "floor_y_m": float(Y),
                               "speed_in": float(np.linalg.norm(v_in)), "speed_out": float(np.linalg.norm(v_out)),
                               "kink": kink, "floor": on_floor(H, (x, y), pxy[knots[k]:])})
    return end_rally(events), flights


def image_to_court(d):
    return np.array(json.loads((d / "court.json").read_text())["H_image_to_court"])


def tracked_clips(d):
    """(segments.csv row, processed track path) for every clip of a match that has a track."""
    return [(r, d / "tracks" / r["file"].replace(".mp4", ".csv")) for r in read_rows(d)
            if (d / "tracks" / r["file"].replace(".mp4", ".csv")).exists()]


def detect(match=None):
    for d in match_dirs(match):
        clips = tracked_clips(d)
        if not clips:
            log(f"- {d.name}: no processed tracks yet")
            continue
        (d / "contacts").mkdir(exist_ok=True)
        H = image_to_court(d)
        counts = defaultdict(int)
        for r, path in clips:
            _, t, xy = read_detections(path)
            events, _ = detect_clip(t, xy, H)
            for e in events:
                counts[e["kind"]] += 1
            with open(d / "contacts" / r["file"].replace(".mp4", ".csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
                w.writeheader()
                w.writerows({k: round(v, 2) if isinstance(v, float) else v for k, v in e.items()} for e in events)
        log(f"{d.name:32s} {len(clips):3d} clips  {counts['hit']} hits, {counts['landing']} landings, "
            f"{counts['after_rally']} after the rally ended")


# === SHUTTLESET LABELS ===
def fetch_labels():
    """ShuttleSet's per-set stroke CSVs for every match in matches.csv with a shuttleset_id.
    curl rather than urllib: the python.org build has no CA certificates until its installer script is run."""
    for m in read_matches():
        if not m.get("shuttleset_id"):
            continue
        out = LABEL_DIR / m["match_id"]
        out.mkdir(parents=True, exist_ok=True)
        for n in range(1, 4):
            dest = out / f"set{n}.csv"
            url = f"{SHUTTLESET_URL}/{m['shuttleset_id']}/set{n}.csv"
            if subprocess.run(["curl", "-sfL", url, "-o", str(dest)]).returncode:
                dest.unlink(missing_ok=True)  # a two-set match has no set3
        log(f"{m['match_id']}: {len(list(out.glob('set*.csv')))} sets")


def read_labels(d):
    """ShuttleSet's hits for one match, each placed in the clip it falls in (file and frame None if none)."""
    rows = read_rows(d)
    H = np.array(json.loads((d / "court.json").read_text())["H_image_to_court"])
    hits = []
    for path in sorted((LABEL_DIR / d.name).glob("set*.csv")):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                frame = int(float(r["frame_num"]))  # frames of the full broadcast, as in segments.csv
                clip = next((c for c in rows if int(c["start_frame"]) <= frame < int(c["end_frame"])), None)
                px, py = r["player_location_x"], r["player_location_y"]  # the hitter's feet, in pixels
                ox, oy = r["opponent_location_x"], r["opponent_location_y"]
                side = "" if not (px and py) else "near" if court.apply_h(H, [(float(px), float(py))])[0, 1] > 0 else "far"
                hits.append({"set": path.stem, "rally": int(r["rally"]), "round": int(float(r["ball_round"])),
                             "type": TYPES.get(r["type"], r["type"]), "side": side,
                             "hitter_px": (float(px), float(py)) if px and py else None,
                             "opponent_px": (float(ox), float(oy)) if ox and oy else None,
                             "file": clip["file"] if clip else None,
                             "frame": frame - int(clip["start_frame"]) - LABEL_LAG if clip else None})
    return hits


# === EVALUATION ===
def pair_up(labels, detections, tol):
    """One-to-one (label index, detection index) pairs at most tol frames apart, closest first."""
    pairs = sorted((abs(dt - lt), i, j) for i, lt in enumerate(labels) for j, dt in enumerate(detections)
                   if abs(dt - lt) <= tol)
    used_l, used_d, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_l and j not in used_d:
            used_l.add(i)
            used_d.add(j)
            out.append((i, j))
    return out


def evaluate(match=None):
    """Recall and precision of detected hits against ShuttleSet's, inside each labelled rally.

    A rally runs from LEAD frames before its first labelled hit to TAIL after its last, so hits made while
    picking the shuttle up between rallies don't count against precision. Labelled hits within EDGE frames
    of a clip's ends are left out: there's no track on one side of them to turn."""
    table, rows_out = [], []
    for d in match_dirs(match):
        if not (LABEL_DIR / d.name).exists() or not (d / "contacts").exists():
            continue
        by_clip = defaultdict(list)
        hits = read_labels(d)
        for h in hits:
            if h["file"]:
                by_clip[h["file"]].append(h)
        counts = {tol: [0, 0, 0] for tol in TOLERANCES}  # matched, labels, detections
        at_edge = 0
        for r, path in tracked_clips(d):
            labels = by_clip.get(r["file"], [])
            contacts = d / "contacts" / r["file"].replace(".mp4", ".csv")
            if not labels or not contacts.exists():
                continue
            n, t, _ = read_detections(path)
            rallies = defaultdict(list)
            for h in labels:
                rallies[(h["set"], h["rally"])].append(h["frame"])
            windows = [(max(0, min(f) - LEAD), min(n, max(f) + TAIL)) for f in rallies.values()]
            with open(contacts, newline="") as f:
                found = [int(e["frame"]) for e in csv.DictReader(f) if e["kind"] == "hit"]
            found = [x for x in found if any(a <= x < b for a, b in windows)]
            usable = [h for h in labels if EDGE <= h["frame"] < n - EDGE]
            at_edge += len(labels) - len(usable)
            matched = {tol: dict(pair_up([h["frame"] for h in usable], found, tol)) for tol in TOLERANCES}
            for tol in TOLERANCES:
                counts[tol][0] += len(matched[tol])
                counts[tol][1] += len(usable)
                counts[tol][2] += len(found)
            loose = max(TOLERANCES)
            for i, h in enumerate(usable):
                j = matched[loose].get(i)
                rows_out.append({"match_id": d.name, "file": r["file"], **{k: h[k] for k in ("set", "rally", "round", "type", "side", "frame")},
                                 "seen": bool(np.any(np.abs(t - h["frame"]) <= 3)),
                                 "error": "" if j is None else found[j] - h["frame"],
                                 **{f"within_{tol}": i in matched[tol] for tol in TOLERANCES}})
        if not counts[TOLERANCES[0]][1]:
            continue
        log(f"{d.name}: {counts[TOLERANCES[0]][1]} labelled hits in tracked clips ({at_edge} at a clip edge left out)")
        for tol in TOLERANCES:
            m, nl, nd = counts[tol]
            log(f"  ±{tol} frames: recall {m / nl:.1%}  precision {m / max(nd, 1):.1%}")
        table.append(d.name)
    if not rows_out:
        log("nothing to evaluate: needs ShuttleSet labels (labels), tracks (1C) and contacts (detect)")
        return
    out = DATA_DIR / "contacts_eval.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    within = lambda rs: f"{np.mean([r['within_2'] for r in rs]):.1%} of {len(rs)}" if rs else "-"
    log("recall at ±2 frames, all matches:")
    for side in ("near", "far"):
        log(f"  {side} player: {within([r for r in rows_out if r['side'] == side])}")
    log(f"  shuttle seen by TrackNet within 3 frames: {within([r for r in rows_out if r['seen']])}, "
        f"not seen: {within([r for r in rows_out if not r['seen']])}")
    by_type = defaultdict(list)
    for r in rows_out:
        by_type[r["type"]].append(r)
    for name, rs in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        if len(rs) >= 20:
            log(f"  {name}: {within(rs)}")
    errors = [r["error"] for r in rows_out if r["error"] != ""]
    log(f"timing (detected - labelled, matches within ±{max(TOLERANCES)}): median {np.median(errors):+.1f} frames, "
        f"mean {np.mean(errors):+.2f}")
    log(f"per-hit results -> {out}")


# === EYE CHECK ===
def plot(match, segment):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = SEG_DIR / match
    row = next(r for r in read_rows(d) if int(r["segment_id"]) == segment)
    n, t, xy = read_detections(d / "tracks" / row["file"].replace(".mp4", ".csv"))
    events, flights = detect_clip(t, xy, image_to_court(d))
    labels =[h for h in read_labels(d) if h["file"] == row["file"]] if (LABEL_DIR / match).exists() else []
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    for ax, k, name in zip(axes, (0, 1), ("x px", "y px (down)")):
        for h in labels:
            ax.axvline(h["frame"], color="#16a34a", lw=4, alpha=0.3)
        for e in events:
            ax.axvline(e["frame"], color={"hit": "#dc2626", "landing": "#0f172a"}.get(e["kind"], "#94a3b8"), lw=1,
                       ls="-" if e["kind"] == "hit" else "--")
        ax.plot(t, xy[:, k], ".", ms=3, color="#475569")
        for m, (f, a, b) in enumerate(flights):
            frames = np.arange(a, b + 1)
            ax.plot(frames, position(f, frames)[:, k], lw=1.5, color=("#2563eb", "#f97316")[m % 2])
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("frame")
    axes[0].set_xlim(0, n)
    axes[0].set_title(f"{match} #{segment}: detections (grey), flights (blue/orange), hits (red), "
                      f"landings (black dashed), after the rally (grey dashed), ShuttleSet hits (green)")
    out = d / "contacts" / f"plot_{row['file'].replace('.mp4', '.png')}"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)
    log(f"plot -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("labels")
    for name in ("detect", "evaluate"):
        sub.add_parser(name).add_argument("--match")
    p = sub.add_parser("plot")
    p.add_argument("--match", required=True)
    p.add_argument("--segment", type=int, required=True)
    args = ap.parse_args()
    if args.cmd == "labels":
        fetch_labels()
    elif args.cmd == "detect":
        detect(args.match)
    elif args.cmd == "evaluate":
        evaluate(args.match)
    else:
        plot(args.match, args.segment)
