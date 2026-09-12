"""Phase 1F — events for the annotator.

One event per hit 1D found, in the shape badminton-annotator.jsx reads: the match, its two players, rally and
shot number, time in the broadcast, and `cv` measurements from 1D and 1E (both players' court positions,
where the shot went, its average speed and launch angle), plus a 5-frame strip around the contact, cropped
on the hitter and the shuttle.

Which named player hit each shot: 1E knows near or far, but players change ends between games. Their shirts
don't change, so the two are told apart by shirt colour, clustered per match. The names go to the clusters
in the match's identity.json; identity_check.jpg shows each cluster, and `swap` flips the names if they're
the wrong way round.

  .venv/bin/python feature_assemble.py assemble [--match <id>]   # -> data/events.json, strips, identity sheets
  .venv/bin/python feature_assemble.py validate                  # events.json against the annotator's event shape
  .venv/bin/python feature_assemble.py evaluate                  # identity, landing and shot numbers vs ShuttleSet
  .venv/bin/python feature_assemble.py swap --match <id>         # the match's two names are the wrong way round

The ShuttleSet matches stay out of events.json: ShuttleSet is the predictor's own training data and holds its
1I test set, so labelling them again would leak into that test. evaluate builds their events in memory.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict

import cv2
import numpy as np

import court
from contact_detect import COURT_BOX, LABEL_DIR, image_to_court, pair_up, read_labels
from player_detect import clip_frames, load_model, players, pose_frames
from shuttle_track import match_dirs, read_rows
from video_prep import DATA_DIR, label, log, read_matches, sheet

EVENTS_JSON = DATA_DIR / "events.json"
MIN_RALLY_HITS = 2         # a clip with fewer hits is a pick-up or a lone serve, not a rally
STRIP = (-2, -1, 0, 1, 2)  # frames around the contact, as the annotator's FrameStrip lays them out
TILE = (320, 180)          # px per frame in a strip
PLAYER_H = 1.8             # m: room left above the hitter's feet in the crop
CROP_PAD = 1.35            # crop size over the box holding the hitter and the shuttle
MIN_CROP_W = 240           # px at 720p: the far player is ~60 px tall and a tighter crop is mostly blur
SIDE_SAMPLES = 4           # hits posed per clip to tell which player is on which half
MIN_FLIGHT_S = 0.2         # s: a landing sooner after the hit than this is 1D pairing a false hit with a landing;
                           # no speed then (ShuttleSet's 1st-percentile gap between hits is 0.37 s)
MAX_SPEED_KMH = 300        # an average over the whole flight above this isn't a real shot; no speed then
KMEANS = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
OTHER = {"near": "far", "far": "near"}
TOL = 2                    # frames, when matching events to ShuttleSet's hits


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def match_clips(d):
    """(segments.csv row, hits, the landing or None, players by frame) for every clip holding a rally."""
    out = []
    for row in read_rows(d):
        name = row["file"].replace(".mp4", ".csv")
        events = read_csv(d / "contacts" / name)
        hits = [e for e in events if e["kind"] == "hit"]
        if len(hits) >= MIN_RALLY_HITS:
            landing = next((e for e in events if e["kind"] == "landing"), None)
            out.append((row, hits, landing, {int(p["frame"]): p for p in read_csv(d / "players" / name)}))
    return out


# === WHO IS WHO ===
def shirts(model, device, d, H, clips):
    """Per clip, the near and far player's shirt colour (Lab), each a median over up to SIDE_SAMPLES hits,
    and the first crop of each for the check sheet."""
    colours, crops = {}, {}
    for row, hits, _, _ in clips:
        frames = [int(e["frame"]) for e in hits]
        sample = frames[::max(1, len(frames) // SIDE_SAMPLES)][:SIDE_SAMPLES]
        found = {"near": [], "far": []}
        for r in pose_frames(model, device, clip_frames(d / row["file"], set(sample))).values():
            for side, p in players(r, H).items():
                if p is not None and p["shirt"] is not None:
                    found[side].append(p["shirt"])
                    x0, y0, x1, y1 = np.maximum(p["box"], 0).astype(int)
                    crops.setdefault((row["file"], side), r.orig_img[y0:y1, x0:x1].copy())
        if found["near"] and found["far"]:
            colours[row["file"]] = (np.median(found["near"], 0), np.median(found["far"], 0))
    return colours, crops


def identify(colours, known=None):
    """Two clusters of shirt colour, one per player, and for each clip the cluster on the near half.
    `known` centres from an earlier run keep cluster 0 the same player from run to run."""
    X = np.float32([c for pair in colours.values() for c in pair])
    _, _, centres = cv2.kmeans(X, 2, None, KMEANS, 5, cv2.KMEANS_PP_CENTERS)
    if known is not None:
        same = np.linalg.norm(centres - known, axis=1).sum()
        if np.linalg.norm(centres[::-1] - known, axis=1).sum() < same:
            centres = centres[::-1]
    near = {}
    for f, (cn, cf) in colours.items():
        stay = np.linalg.norm(cn - centres[0]) + np.linalg.norm(cf - centres[1])
        swap = np.linalg.norm(cn - centres[1]) + np.linalg.norm(cf - centres[0])
        near[f] = 0 if stay <= swap else 1
    return centres, near


def identity(d, clips, get_model, meta, reidentify=False):
    """The match's identity.json, made (posing a few frames per clip) if it's missing or out of date."""
    path = d / "identity.json"
    old = json.loads(path.read_text()) if path.exists() else None
    files = [row["file"] for row, *_ in clips]
    if old and not reidentify and all(f in old["near_cluster"] for f in files):
        return old
    model, device = get_model()
    colours, crops = shirts(model, device, d, image_to_court(d), clips)
    centres, near = identify(colours, np.array(old["centres"]) if old else None)
    # A clip where a shirt wasn't seen keeps the previous clip's ends: players change ends only between games
    filled, last = {}, None
    for f in files:
        last = near.get(f, last)
        filled[f] = last
    first = next((v for v in filled.values() if v is not None), 0)
    ident = {"names": old["names"] if old else [meta["player_a"], meta["player_b"]],
             "centres": centres.tolist(),
             "near_cluster": {f: first if v is None else v for f, v in filled.items()},
             "clips_with_both_shirts": len(colours), "clips": len(files)}
    path.write_text(json.dumps(ident, indent=1))
    check_sheet(d, ident, crops)
    return ident


def check_sheet(d, ident, crops, per=8):
    """identity_check.jpg: a row of each cluster's player, labelled with the name it gets."""
    rows = {0: [], 1: []}
    for (f, side), img in crops.items():
        if img.size and f in ident["near_cluster"]:
            c = ident["near_cluster"][f] if side == "near" else 1 - ident["near_cluster"][f]
            if len(rows[c]) < per:
                rows[c].append(img)
    tiles = []
    for c in (0, 1):
        for img in rows[c] + [None] * (per - len(rows[c])):
            tile = np.zeros((220, 150, 3), np.uint8)
            if img is not None:
                s = min(150 / img.shape[1], 200 / img.shape[0])
                small = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
                x = (150 - small.shape[1]) // 2
                tile[20:20 + small.shape[0], x:x + small.shape[1]] = small
            tiles.append(label(tile, ident["names"][c][:24]))
    sheet(tiles, per, d / "identity_check.jpg")


# === EVENTS ===
def pos(pl, frame, side, frames):
    """A player's court position (m) and feet (px) at a contact, else at the nearest contact in the clip
    where they were found."""
    for f in sorted(frames, key=lambda f: abs(f - frame)):
        p = pl.get(f)
        if p and p.get(f"{side}_x_m"):
            return ((float(p[f"{side}_x_m"]), float(p[f"{side}_y_m"])),
                    (float(p[f"{side}_foot_x_px"]), float(p[f"{side}_foot_y_px"])))
    return None, None


def under_shuttle(H, shuttle_px, feet_px):
    """The floor point below the shuttle at a contact: straight down the screen from the shuttle to the
    hitter's feet level (verticals stay close to vertical in these broadcasts), then onto the court."""
    return tuple(court.apply_h(H, [(shuttle_px[0], feet_px[1])])[0])


def crop_box(cam, feet_m, feet_px, shuttle_px, size=(1280, 720)):
    """A 16:9 box holding the hitter (feet to PLAYER_H above them) and the shuttle."""
    head = court.project(cam["K"], cam["R"], cam["t"], [[*feet_m, PLAYER_H]])[0]
    pts = np.array([feet_px, head, shuttle_px])
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    w = min(max((x1 - x0) * CROP_PAD, (y1 - y0) * CROP_PAD * 16 / 9, MIN_CROP_W), size[0], size[1] * 16 / 9)
    h = w * 9 / 16
    x = np.clip((x0 + x1) / 2 - w / 2, 0, size[0] - w)
    y = np.clip((y0 + y1) / 2 - h / 2, 0, size[1] - h)
    return int(x), int(y), int(w), int(h)


def clock(seconds):
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def match_events(d, meta, ident, H, cam, fps, clips):
    """The annotator's events for one match. Hits without a hitter position are left out; shot_num still
    counts them, so the numbering matches the order of contacts in the clip."""
    names, events = ident["names"], []
    for rally, (row, hits, landing, pl) in enumerate(clips, 1):
        near_c = ident["near_cluster"].get(row["file"], 0)
        frames = [int(e["frame"]) for e in hits]
        track = None
        for i, e in enumerate(hits):
            f, p = frames[i], pl.get(frames[i]) or {}
            side = p.get("hitter")
            if side not in OTHER:
                continue
            me, me_px = pos(pl, f, side, frames)
            opp, _ = pos(pl, f, OTHER[side], frames)
            if me is None or opp is None:
                continue
            shuttle = (float(e["x_px"]), float(e["y_px"]))
            # ShuttleSet's landing for a returned shot is where the next player hits it; the last shot's is
            # where it meets the floor, else where the track ends (in the air, so only roughly right)
            if i + 1 < len(hits):
                nxt, nf = hits[i + 1], frames[i + 1]
                _, n_px = pos(pl, nf, (pl.get(nf) or {}).get("hitter") or OTHER[side], frames)
                land = under_shuttle(H, (float(nxt["x_px"]), float(nxt["y_px"])), n_px) if n_px else None
                t_land, source = nf, "next_hit"
            elif landing:
                land, t_land, source = (float(landing["floor_x_m"]), float(landing["floor_y_m"])), int(landing["frame"]), "floor"
            else:
                # The last detection that maps onto the court: a shuttle leaving the top of the frame maps
                # far past the far baseline (up to 100 m on the 2025 matches), so walk back from the track's end
                track = track or read_csv(d / "tracks" / row["file"].replace(".mp4", ".csv"))
                seen = [r for r in track if r["detected"] == "1" and int(r["frame"]) > f and r["floor_x_m"]
                        and abs(float(r["floor_x_m"])) <= COURT_BOX[0] and abs(float(r["floor_y_m"])) <= COURT_BOX[1]]
                land = (float(seen[-1]["floor_x_m"]), float(seen[-1]["floor_y_m"])) if seen else None
                t_land, source = (int(seen[-1]["frame"]) if seen else f + 1), "track_end"
            if land is None:
                continue
            cluster = near_c if side == "near" else 1 - near_c
            # Average speed over the floor from hitter to landing: the initial speed of a shot moving
            # toward or away from the camera can't be read off one view
            flight = (t_land - f) / fps
            speed = np.hypot(land[0] - me[0], land[1] - me[1]) / flight * 3.6 if flight >= MIN_FLIGHT_S else None
            if speed is not None and speed > MAX_SPEED_KMH:
                speed = None
            angle = (round(float(np.degrees(np.arctan2(-float(e["vy_out"]), abs(float(e["vx_out"]))))))
                     if e.get("vx_out") else None)
            to01 = lambda xy: [round(float(v), 4) for v in court.to_normalized([xy])[0]]
            events.append({
                "id": f"{d.name}:{int(row['segment_id']):04d}:{f}",
                "match": f"{meta['player_a']} vs {meta['player_b']} - {meta['event']} {meta['round']}",
                "players": [meta["player_a"], meta["player_b"]],
                "rally": rally,
                "shot_num": i + 1,
                "frame_time": clock((int(row["start_frame"]) + f) / fps),
                "cv": {"player_xy": to01(me), "opponent_xy": to01(opp), "landing_xy": to01(land),
                       "speed": None if speed is None else int(round(speed)), "trajectory_angle": angle,
                       "hitting_player": [meta["player_a"], meta["player_b"]].index(names[cluster]) + 1,
                       "hitter_side": side, "landing_source": source},
                "frames": None,
                "source": {"match_id": d.name, "segment": int(row["segment_id"]), "clip": row["file"], "frame": f},
                "claude_label": None,
                "annotation": None,
                "_crop": (crop_box(cam, me, me_px, shuttle), shuttle),
            })
    return events


def write_strips(d, events):
    """One JPEG per event: the STRIP frames side by side, cropped on the hitter and the shuttle, the shuttle
    ringed on the contact frame. Existing strips are kept."""
    (d / "strips").mkdir(exist_ok=True)
    by_clip = defaultdict(list)
    for ev in events:
        by_clip[ev["source"]["clip"]].append(ev)
    for clip, evs in by_clip.items():
        path = lambda ev: d / "strips" / f"{clip[:-4]}_f{ev['source']['frame']:05d}.jpg"
        todo = [ev for ev in evs if not path(ev).exists()]
        if todo:
            frames = clip_frames(d / clip, {ev["source"]["frame"] + o for ev in todo for o in STRIP if ev["source"]["frame"] + o >= 0})
            for ev in todo:
                (x, y, w, h), (sx, sy) = ev["_crop"]
                tiles = []
                for o in STRIP:
                    img = frames.get(ev["source"]["frame"] + o)
                    tile = np.zeros((TILE[1], TILE[0], 3), np.uint8) if img is None else \
                        cv2.resize(img[y:y + h, x:x + w], TILE, interpolation=cv2.INTER_AREA)
                    if o == 0:
                        s = TILE[0] / w
                        cv2.circle(tile, (int((sx - x) * s), int((sy - y) * s)), 11, (0, 0, 255), 1, cv2.LINE_AA)
                    tiles.append(tile)
                cv2.imwrite(str(path(ev)), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
        for ev in evs:
            # The annotator's dev server serves the repo root, where data/ lives by default
            ev["frames"] = "/" + str(path(ev).relative_to(DATA_DIR.parent))


def build(d, meta, get_model, reidentify=False, strips=True):
    clips = match_clips(d)
    ident = identity(d, clips, get_model, meta, reidentify)
    cam = {k: np.array(v, float) for k, v in json.loads((d / "court.json").read_text())["camera"].items() if k in ("K", "R", "t")}
    fps = json.loads((d / "match.json").read_text())["fps"]
    events = match_events(d, meta, ident, image_to_court(d), cam, fps, clips)
    if strips:
        write_strips(d, events)
    for ev in events:
        ev.pop("_crop")
    return events, ident


def lazy_model():
    cache = {}
    return lambda: cache.setdefault("model", load_model())


def assemble(match=None, reidentify=False):
    metas = {m["match_id"]: m for m in read_matches()}
    get_model = lazy_model()
    out = [e for e in (json.loads(EVENTS_JSON.read_text()) if EVENTS_JSON.exists() and match else [])
           if e["source"]["match_id"] != match]
    for d in match_dirs(match):
        if (LABEL_DIR / d.name).exists() or not (d / "players").exists():
            continue
        events, ident = build(d, metas[d.name], get_model, reidentify)
        out += events
        sources = Counter(e["cv"]["landing_source"] for e in events)
        log(f"{d.name:32s} {len(events):5d} events in {len({e['rally'] for e in events})} rallies  "
            f"landing from {dict(sources)}  both shirts seen in {ident['clips_with_both_shirts']}/{ident['clips']} clips  "
            f"-> check {d.name}/identity_check.jpg")
    EVENTS_JSON.write_text(json.dumps(out))
    log(f"{len(out)} events -> {EVENTS_JSON}")


def swap(match):
    d = DATA_DIR / "segments" / match
    ident = json.loads((d / "identity.json").read_text())
    ident["names"] = ident["names"][::-1]
    (d / "identity.json").write_text(json.dumps(ident, indent=1))
    log(f"{match}: cluster 0 is now {ident['names'][0]}, cluster 1 {ident['names'][1]}")
    assemble(match, reidentify=True)  # re-poses the shirts so identity_check.jpg is redrawn with the new names


# === CHECKS ===
def validate():
    """Every event in events.json against the shape badminton-annotator.jsx reads, plus sanity checks on the
    measurements. Returns True if nothing is wrong."""
    events = json.loads(EVENTS_JSON.read_text())
    wrong, ids = Counter(), set()
    shape = {"id": str, "match": str, "players": list, "rally": int, "shot_num": int, "frame_time": str, "cv": dict}
    for e in events:
        for k, t in shape.items():
            if not isinstance(e.get(k), t):
                wrong[f"{k} missing or not {t.__name__}"] += 1
        if e.get("id") in ids:
            wrong["duplicate id"] += 1
        ids.add(e.get("id"))
        if "claude_label" not in e or e.get("annotation") is not None:
            wrong["claude_label missing or annotation not null"] += 1
        if len(e.get("players", [])) != 2:
            wrong["players is not two names"] += 1
        cv_ = e.get("cv", {})
        for k, lo, hi in (("player_xy", -0.3, 1.3), ("opponent_xy", -0.3, 1.3), ("landing_xy", -0.6, 1.6)):
            xy = cv_.get(k)
            if not (isinstance(xy, list) and len(xy) == 2 and all(isinstance(v, (int, float)) and lo <= v <= hi for v in xy)):
                wrong[f"cv.{k} not two numbers in [{lo}, {hi}]"] += 1
        if cv_.get("speed") is not None and (not isinstance(cv_["speed"], int) or not 0 <= cv_["speed"] <= MAX_SPEED_KMH):
            wrong[f"cv.speed not null or an int in 0-{MAX_SPEED_KMH} km/h"] += 1
        if cv_.get("trajectory_angle") is not None and not -90 <= cv_["trajectory_angle"] <= 90:
            wrong["cv.trajectory_angle outside -90..90"] += 1
        if cv_.get("hitting_player") not in (1, 2):
            wrong["cv.hitting_player not 1 or 2"] += 1
        if e.get("frames") and not (DATA_DIR.parent / e["frames"].lstrip("/")).exists():
            wrong["frames file missing"] += 1
    shots = defaultdict(list)
    for e in events:
        shots[(e["match"], e["rally"])].append(e["shot_num"])
    wrong["repeated shot_num in a rally"] += sum(len(v) != len(set(v)) for v in shots.values())
    wrong = {k: v for k, v in wrong.items() if v}
    # Sanity: the hitter stands on their own half, and a returned shot lands on the other one
    own = np.mean([(e["cv"]["player_xy"][1] > 0.5) == (e["cv"]["hitter_side"] == "near") for e in events])
    returned = [e for e in events if e["cv"]["landing_source"] == "next_hit"]
    across = np.mean([(e["cv"]["landing_xy"][1] > 0.5) != (e["cv"]["player_xy"][1] > 0.5) for e in returned])
    log(f"{len(events)} events, {len(shots)} rallies, {len({e['match'] for e in events})} matches; "
        f"{sum(bool(e['frames']) for e in events)} with a frame strip")
    log(f"  hitter on their own half: {own:.1%}; returned shots landing across the net: {across:.1%}")
    speeds = [e["cv"]["speed"] for e in events if e["cv"]["speed"] is not None]
    log(f"  speed km/h median {np.median(speeds):.0f} (10-90%: {np.percentile(speeds, 10):.0f}-{np.percentile(speeds, 90):.0f}); "
        f"{len(events) - len(speeds)} events without one")
    for k, v in wrong.items():
        log(f"  WRONG: {k}: {v}")
    log("shape OK" if not wrong else "shape has problems")
    return not wrong


def evaluate():
    """Against ShuttleSet, on its matches' events built in memory: the hitter's name (A/B up to which is
    which), the landing point, and shot numbers."""
    metas = {m["match_id"]: m for m in read_matches()}
    get_model = lazy_model()
    for d in match_dirs():
        if not (LABEL_DIR / d.name).exists() or not (d / "players").exists():
            continue
        events, _ = build(d, metas[d.name], get_model, strips=False)
        H = image_to_court(d)
        labels = [h for h in read_labels(d) if h["file"]]
        same, n, land_err, offsets = 0, 0, defaultdict(list), Counter()
        for clip in {e["source"]["clip"] for e in events}:
            ours = sorted((e for e in events if e["source"]["clip"] == clip), key=lambda e: e["source"]["frame"])
            lab = [h for h in labels if h["file"] == clip]
            for i, j in pair_up([h["frame"] for h in lab], [e["source"]["frame"] for e in ours], TOL):
                h, e = lab[i], ours[j]
                n += 1
                same += (h["player"] == "A") == (e["cv"]["hitting_player"] == 1)
                offsets[e["shot_num"] - h["round"]] += 1
                if h["landing_px"]:
                    truth = court.apply_h(H, [h["landing_px"]])[0]
                    ours_m = (np.array(e["cv"]["landing_xy"]) - 0.5) * [2 * court.HALF_W, 2 * court.HALF_L]
                    land_err[e["cv"]["landing_source"]].append(float(np.linalg.norm(ours_m - truth)))
        if not n:
            continue
        agree = max(same, n - same) / n
        log(f"{d.name}: {n} matched hits")
        log(f"  hitter's name right {agree:.1%} ({'A is player_a' if same >= n - same else 'A is player_b'})")
        for src, errs in land_err.items():
            log(f"  landing ({src}): median {np.median(errs):.2f} m from ShuttleSet's, within 1 m {np.mean(np.array(errs) <= 1):.0%} (n={len(errs)})")
        total = sum(offsets.values())
        log("  shot_num minus ShuttleSet's: " + ", ".join(f"{k:+d}: {v / total:.0%}" for k, v in sorted(offsets.items()) if v / total >= 0.02))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("assemble")
    a.add_argument("--match")
    a.add_argument("--reidentify", action="store_true", help="cluster the shirts again even if identity.json is current")
    sub.add_parser("validate")
    sub.add_parser("evaluate")
    sub.add_parser("swap").add_argument("--match", required=True)
    args = ap.parse_args()
    if args.cmd == "assemble":
        assemble(args.match, args.reidentify)
    elif args.cmd == "validate":
        validate()
    elif args.cmd == "evaluate":
        evaluate()
    else:
        swap(args.match)
