"""Phase 1E — the players at each contact.

Pose runs only on the frames around each contact 1D found, not the whole match. People whose feet map onto
the court are the players (the umpire, line judges and coaches stand off it); the one on the near half is
the near player. The hitter is the player whose wrist is closest to the shuttle at the contact.

  .venv/bin/python player_detect.py detect [--match <id>]              # -> segments/<match>/players/seg_0001.csv
  .venv/bin/python player_detect.py evaluate                            # hitter and feet against ShuttleSet's labels
  .venv/bin/python player_detect.py sheet --match <id> --segment <n>    # stills of each contact, players marked
"""
import argparse
import csv
from collections import Counter

import cv2
import numpy as np

import court
from contact_detect import LABEL_DIR, image_to_court, pair_up, read_labels
from shuttle_track import match_dirs, read_rows
from video_prep import DATA_DIR, SEG_DIR, label, log, sheet

MODEL = "yolo11n-pose.pt"   # 100 ms/frame on an M1's GPU at IMGSZ, both players found on 16 of 16 test
                            # contacts; yolo11s-pose took 890 ms. The larger ones are for a Colab GPU
IMGSZ = 1280          # px; the far player is only ~60 px tall in a 720p broadcast
CONF = 0.25           # person detections below this are ignored
KP_CONF = 0.3         # keypoints below this aren't used
WINDOW = 1            # frames either side of a contact also posed; a player's position is the median over them
BATCH = 8
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP, L_ANKLE, R_ANKLE = 5, 6, 9, 10, 11, 12, 15, 16  # COCO keypoints
ON_COURT = (court.HALF_W + 1.2, court.HALF_L + 1.5)  # m, |X| and |Y|: a player chasing a wide or deep shot
                                                     # steps off the court; the officials sit further out
KINDS = ("hit", "landing")
TOL = 2               # frames, when matching contacts to ShuttleSet's hits
PLAYER_FIELDS = ["frame", "kind", "hitter", "near_reach", "far_reach",
                 "near_x_m", "near_y_m", "far_x_m", "far_y_m",
                 "near_foot_x_px", "near_foot_y_px", "far_foot_x_px", "far_foot_y_px"]


def load_model():
    import torch
    from ultralytics import YOLO
    from ultralytics.utils import nms
    # NMS gives up after 2 s + 0.05 s per image and returns nobody for the rest of the batch. On an M1's GPU
    # its clock also runs while it waits for the model to finish, so it gave up ~50 times over the 2025
    # matches; 3% of reviewed hits were missing a player. The limit is only a guard against runaway boxes.
    if not getattr(nms.non_max_suppression, "patient", False):
        original = nms.non_max_suppression

        def patient(*args, **kwargs):
            return original(*args, **{**kwargs, "max_time_img": 10.0})
        patient.patient = True
        nms.non_max_suppression = patient
    return YOLO(MODEL), "mps" if torch.backends.mps.is_available() else "cpu"


def people(result, H):
    """Each person detected in one frame: feet (px), their court position, wrists (px) and box height."""
    out = []
    if result.keypoints is None or result.boxes is None:
        return out
    kps = result.keypoints.data.cpu().numpy()          # (people, 17, 3): x, y, confidence
    boxes = result.boxes.xyxy.cpu().numpy()
    for kp, (x0, y0, x1, y1) in zip(kps, boxes):
        ankles = kp[[L_ANKLE, R_ANKLE]]
        ankles = ankles[ankles[:, 2] >= KP_CONF]
        feet = ankles[:, :2].mean(0) if len(ankles) else np.array([(x0 + x1) / 2, y1])  # else the box's bottom
        wrists = kp[[L_WRIST, R_WRIST]]
        wrists = wrists[wrists[:, 2] >= KP_CONF][:, :2]
        X, Y = court.apply_h(H, [feet])[0]
        out.append({"feet": feet, "X": X, "Y": Y, "wrists": wrists, "height": max(y1 - y0, 1.0),
                    "top": np.array([(x0 + x1) / 2, y0]), "box": np.array([x0, y0, x1, y1]),
                    "shirt": shirt_colour(result.orig_img, kp)})
    return out


def shirt_colour(img, kp):
    """Median colour (Lab) inside the torso, the quadrilateral of shoulders and hips; None if any of the four
    isn't confidently placed. 1F tells the two players apart by it."""
    corners = kp[[L_SHOULDER, R_SHOULDER, R_HIP, L_HIP]]
    if (corners[:, 2] < KP_CONF).any():
        return None
    poly = corners[:, :2].astype(np.int32)
    x0, y0 = np.maximum(poly.min(0), 0)
    x1, y1 = poly.max(0) + 1
    patch = img[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    mask = np.zeros(patch.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, poly - [x0, y0], 1)
    if mask.sum() < 30:
        return None
    return np.median(cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)[mask.astype(bool)], axis=0)


def players(result, H):
    """The near and far player in one frame (either may be None): the person on each half, feet on the court,
    with the largest box, since a player is nearer the camera than anyone behind them."""
    on = [p for p in people(result, H) if abs(p["X"]) <= ON_COURT[0] and abs(p["Y"]) <= ON_COURT[1]]
    pick = lambda side: max((p for p in on if (p["Y"] > 0) == (side == "near")), key=lambda p: p["height"], default=None)
    return {"near": pick("near"), "far": pick("far")}


def reach(player, shuttle):
    """How far the shuttle is from the player's nearer wrist (their head if no wrist shows), in the player's
    own heights, so it means the same for the small far player as the large near one."""
    if player is None:
        return np.inf
    points = player["wrists"] if len(player["wrists"]) else player["top"][None]
    return float(np.linalg.norm(points - np.asarray(shuttle), axis=1).min() / player["height"])


def summarize(event, frames):
    """One contact's row from the players in each posed frame around it (the contact frame first)."""
    row = {"frame": event["frame"], "kind": event["kind"]}
    shuttle = (float(event["x_px"]), float(event["y_px"]))
    for side in ("near", "far"):
        seen = [f[side] for f in frames if f[side] is not None]
        row[f"{side}_reach"] = min((reach(p, shuttle) for p in seen), default=np.inf)
        if seen:
            X, Y = np.median([(p["X"], p["Y"]) for p in seen], axis=0)
            fx, fy = np.median([p["feet"] for p in seen], axis=0)
            row.update({f"{side}_x_m": X, f"{side}_y_m": Y, f"{side}_foot_x_px": fx, f"{side}_foot_y_px": fy})
    if np.isfinite(min(row["near_reach"], row["far_reach"])):
        row["hitter"] = "near" if row["near_reach"] < row["far_reach"] else "far"
    return row


def fmt(v):
    if isinstance(v, float):
        return "" if not np.isfinite(v) else round(v, 3)
    return v


def clip_frames(path, wanted):
    """The frames in `wanted` from one clip, read in order (seeking an H.264 clip is slower than decoding)."""
    cap, out, i, last = cv2.VideoCapture(str(path)), {}, 0, max(wanted)
    while i <= last:
        if i in wanted:
            ok, frame = cap.read()
            if ok:
                out[i] = frame
        else:
            ok = cap.grab()
        if not ok:
            break
        i += 1
    cap.release()
    return out


def pose_frames(model, device, frames):
    """frame index -> YOLO result, run in batches."""
    keys, out = sorted(frames), {}
    for i in range(0, len(keys), BATCH):
        chunk = keys[i:i + BATCH]
        results = model([frames[k] for k in chunk], imgsz=IMGSZ, conf=CONF, device=device, verbose=False)
        out.update(zip(chunk, results))
    return out


def read_players(path):
    """frame -> row of an existing players CSV; empty if there isn't one."""
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {int(r["frame"]): r for r in csv.DictReader(f)}


def read_contacts(d, row):
    path = d / "contacts" / row["file"].replace(".mp4", ".csv")
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [e for e in csv.DictReader(f) if e["kind"] in KINDS]


def detect(match=None):
    model, device = load_model()
    log(f"{MODEL} on {device}")
    for d in match_dirs(match):
        if not (d / "contacts").exists():
            continue
        H = image_to_court(d)
        (d / "players").mkdir(exist_ok=True)
        stats = Counter()
        for row in read_rows(d):
            events = read_contacts(d, row)
            if not events:
                continue
            out = d / "players" / row["file"].replace(".mp4", ".csv")
            old = read_players(out)
            if set(old) == {int(e["frame"]) for e in events}:
                # Posed already: 1D reran without moving a contact, though it may have relabelled one
                rows = [dict(old[int(e["frame"])], kind=e["kind"]) for e in events]
                stats["reused"] += 1
            else:
                n = int(row["end_frame"]) - int(row["start_frame"])
                wanted = {f for e in events for f in range(int(e["frame"]) - WINDOW, int(e["frame"]) + WINDOW + 1) if 0 <= f < n}
                results = pose_frames(model, device, clip_frames(d / row["file"], wanted))
                posed = {f: players(r, H) for f, r in results.items()}
                rows = []
                for e in events:
                    f0 = int(e["frame"])
                    around = [posed[f] for f in sorted(range(f0 - WINDOW, f0 + WINDOW + 1), key=lambda f: abs(f - f0)) if f in posed]
                    rows.append(summarize(e, around))
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=PLAYER_FIELDS)
                w.writeheader()
                w.writerows({k: fmt(r.get(k, "")) for k in PLAYER_FIELDS} for r in rows)
            has = lambda r, k: r.get(k) not in (None, "")
            hits = [r for r in rows if r["kind"] == "hit"]
            stats["hits"] += len(hits)
            stats["both"] += sum(has(r, "near_x_m") and has(r, "far_x_m") for r in hits)
            stats["named"] += sum(has(r, "hitter") for r in hits)
            named = [r["hitter"] for r in hits if has(r, "hitter")]
            stats["pairs"] += max(0, len(named) - 1)
            stats["alternate"] += sum(a != b for a, b in zip(named, named[1:]))
        if stats["hits"]:
            log(f"{d.name:32s} {stats['hits']:5d} hits  both players found {stats['both'] / stats['hits']:.1%}  "
                f"hitter named {stats['named'] / stats['hits']:.1%}  "
                f"consecutive hitters alternate {stats['alternate'] / max(1, stats['pairs']):.1%}")


def evaluate(match=None):
    """Against ShuttleSet: the hitter (near or far) at every detected hit within TOL frames of a labelled one,
    and the distance from our feet to the labelled feet of the hitter and the opponent."""
    side_ok, side_n, named, matched = 0, 0, 0, 0
    foot_err = {"hitter": [], "opponent": []}
    for d in match_dirs(match):
        if not (LABEL_DIR / d.name).exists() or not (d / "players").exists():
            continue
        labels = [h for h in read_labels(d) if h["file"]]
        m_ok = m_n = 0
        for row in read_rows(d):
            path = d / "players" / row["file"].replace(".mp4", ".csv")
            clip_labels = [h for h in labels if h["file"] == row["file"]]
            if not path.exists() or not clip_labels:
                continue
            with open(path, newline="") as f:
                ours = [r for r in csv.DictReader(f) if r["kind"] == "hit"]
            for i, j in pair_up([h["frame"] for h in clip_labels], [int(r["frame"]) for r in ours], TOL):
                h, r = clip_labels[i], ours[j]
                matched += 1
                named += bool(r["hitter"])
                if h["side"] and r["hitter"]:
                    side_n += 1
                    m_n += 1
                    ok = r["hitter"] == h["side"]
                    side_ok += ok
                    m_ok += ok
                for who, px in (("hitter", h["hitter_px"]), ("opponent", h["opponent_px"])):
                    side = h["side"] if who == "hitter" else {"near": "far", "far": "near"}.get(h["side"])
                    if px and side and r[f"{side}_foot_x_px"]:
                        foot_err[who].append(np.hypot(float(r[f"{side}_foot_x_px"]) - px[0], float(r[f"{side}_foot_y_px"]) - px[1]))
        if m_n:
            log(f"{d.name}: hitter right at {m_ok}/{m_n} matched hits ({m_ok / m_n:.1%})")
    if not matched:
        log("nothing to evaluate: needs ShuttleSet labels, contacts (1D) and players (detect)")
        return
    log(f"all: hitter right {side_ok}/{side_n} ({side_ok / max(1, side_n):.1%}); named at {named}/{matched} matched hits")
    for who, e in foot_err.items():
        if e:
            log(f"  {who} feet vs ShuttleSet: median {np.median(e):.0f} px, within 30 px {np.mean(np.array(e) <= 30):.0%} (n={len(e)})")


def sheet_cmd(match, segment, cols=3):
    """Every contact in one clip: the pose model's skeletons, our near (green) and far (orange) feet, the
    shuttle (red) and the hitter."""
    model, device = load_model()
    d = SEG_DIR / match
    row = next(r for r in read_rows(d) if int(r["segment_id"]) == segment)
    with open(d / "players" / row["file"].replace(".mp4", ".csv"), newline="") as f:
        ours = {int(r["frame"]): r for r in csv.DictReader(f)}
    events = read_contacts(d, row)
    results = pose_frames(model, device, clip_frames(d / row["file"], {int(e["frame"]) for e in events}))
    tiles = []
    for e in events:
        f0, r = int(e["frame"]), ours.get(int(e["frame"]), {})
        if f0 not in results:
            continue
        img = results[f0].plot(boxes=False, labels=False)
        for side, color in (("near", (80, 220, 80)), ("far", (0, 160, 255))):
            if r.get(f"{side}_foot_x_px"):
                cv2.circle(img, (int(float(r[f"{side}_foot_x_px"])), int(float(r[f"{side}_foot_y_px"]))), 9, color, 3)
        cv2.circle(img, (int(float(e["x_px"])), int(float(e["y_px"]))), 7, (0, 0, 255), -1)
        tiles.append(label(cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA),
                           f"frame {f0} {e['kind']}  hitter: {r.get('hitter') or '?'}"))
    out = d / "players" / f"sheet_{row['file'].replace('.mp4', '.jpg')}"
    sheet(tiles, cols, out)
    log(f"sheet -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("detect", "evaluate"):
        sub.add_parser(name).add_argument("--match")
    s = sub.add_parser("sheet")
    s.add_argument("--match", required=True)
    s.add_argument("--segment", type=int, required=True)
    args = ap.parse_args()
    if args.cmd == "detect":
        detect(args.match)
    elif args.cmd == "evaluate":
        evaluate(args.match)
    else:
        sheet_cmd(args.match, args.segment)
