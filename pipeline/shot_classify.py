"""Shot-type suggestions for the annotator, from a classifier trained on ShuttleSet.

Every exported shot is reviewed by hand, so a suggestion only has to make the review faster: Enter confirms
it, 1-0 corrects it. A shot's type follows mostly from where it was hit from, where it went and how fast,
and ShuttleSet labels those for 30,000 shots in exactly the predictor's 10 types. Trained on the
predictor's own og_train.csv, the model learns ShuttleSet's conventions (lob or clear, push/rush or drive)
directly, where a prompt would have to describe them. It stands in for 1G's Claude labelling.

  .venv/bin/python shot_classify.py train      # og_train.csv -> data/shot_classifier.pkl, with cross-validated accuracy
  .venv/bin/python shot_classify.py evaluate   # accuracy on the ShuttleSet matches: from their labels and from 1F's events
  .venv/bin/python shot_classify.py label      # adds model_label to every event in data/events.json (rerun after assemble)

Both sources give the same features: the hitter's, opponent's and landing position in metres, turned so the
hitter is on the near half; where the previous hitter stood; the time since the previous hit and to the next
one, and the average speeds of the incoming and outgoing flights. ShuttleSet has no shuttle height or launch
angle, so 1D's angle isn't used. Each shot is also trained with a copy carrying 1D-1F's measured errors:
on the dev match's events that took top-1 from 61.7% to 68.9%.
"""
import argparse
import csv
import json
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from contact_detect import LABEL_DIR, pair_up, read_labels
from court import HALF_L, HALF_W
from shuttle_track import match_dirs
from video_prep import DATA_DIR, log, read_matches

PREDICTOR_DIR = Path(os.environ.get("PREDICTOR_DIR", Path.home() / "Desktop" / "Personal Projects" / "BadmintonShotPredictor"))
TRAIN_CSV = PREDICTOR_DIR / "og_train.csv"  # ShuttleSet22's training split: 44 matches, 30,172 shots, the predictor's 10 types
MODEL_PATH = DATA_DIR / "shot_classifier.pkl"
LABEL_FPS = 30        # ShuttleSet's frame_num is time x 30 in every match
LEAK_SHARE = 0.5      # an og_train match sharing this share of its hit frames with a ShuttleSet match is that match
                      # (wtf2020 is og_train's match 7: 676 of 676; the next best overlap is 14 of 684)
CV_FOLDS = 5          # og_train's matches are split this many ways for the cross-validated accuracy
TOL = 2               # frames, matching 1F's events to ShuttleSet's hits (as in 1D and 1F's evaluations)
TOP = 3               # alternatives kept per event
CONFIDENT = 0.7       # probability the evaluation reports accuracy above, as a guide to trusting Enter
# Training adds copies of each shot with the pipeline's errors, so the model doesn't lean on precision it won't get
ROUGH_COPIES = 1      # rough copies per shot
NOISE_POSITION = 0.4  # m, sd per axis on the players' feet (1E's are a median 22-35 px from ShuttleSet's)
NOISE_LANDING = 0.85  # m, sd per axis: 1F's landings are a median 1.0-1.1 m from ShuttleSet's (1.18 sd for a 2-D normal)
NOISE_TIME = 0.05     # s: 1D's hits are 1-2 frames from ShuttleSet's
HIDE_PREV = 0.3       # share of rough copies without a previous hit: in 1F a rally's first event is often not the serve
                      # (35 of 69 yto2021 clips start after it), and 1D misses 16-21% of hits
HIDE_LANDING = 0.18   # share of rough copies of a returned shot without its landing: 18% of 1F's returned shots land
                      # on the hitter's own half, which features() throws out
# ShuttleSet's court template (see CLAUDE.md): the doubles court spans x 27.5-327.5 px and y 150-810 px, net at 480.
# On og_train's wtf2020 rows these land a median 0.07 m from the raw pixels mapped through our own court.json.
TEMPLATE_CX, TEMPLATE_NET, TEMPLATE_W, TEMPLATE_L = 177.5, 480.0, 300.0, 660.0
TO_PREDICTOR = {  # ShuttleSet's 18 types (contact_detect.TYPES) -> the predictor's 10, merged as ShuttleSet22's preprocess_data.py does
    "short service": "short service", "long service": "long service", "clear": "clear", "lob": "lob",
    "net shot": "net shot", "cross-court net shot": "net shot", "drop": "drop", "passive drop": "drop",
    "smash": "smash", "wrist smash": "smash", "drive": "drive", "driven flight": "drive", "back-court drive": "drive",
    "push": "push/rush", "rush": "push/rush",
    "return net": "defensive shot", "defensive lob": "defensive shot", "defensive drive": "defensive shot",
}
FEATURES = ["hitter_x", "hitter_y", "opponent_x", "opponent_y", "landing_x", "landing_y", "prev_hitter_x",
            "prev_hitter_y", "since_prev_s", "flight_s", "speed_ms", "distance_m", "across_m", "incoming_ms"]


# === FEATURES ===
def features(hitter, opponent, landing, prev_hitter, since_prev, flight):
    """One shot's features. Positions are court metres (X across, Y along, net at 0), turned 180 degrees when
    the hitter is on the far half: a shot from there is the same shot seen from the other end. None where
    unknown, which the model takes as missing."""
    s = 1.0 if hitter[1] >= 0 else -1.0
    turn = lambda p: (np.nan, np.nan) if p is None else (s * p[0], s * p[1])
    (hx, hy), (ox, oy), (lx, ly), (px, py) = turn(hitter), turn(opponent), turn(landing), turn(prev_hitter)
    since_prev = np.nan if since_prev is None else since_prev
    flight = np.nan if flight is None else flight
    if ly > 0:
        # A landing on the hitter's own half is never used (the user's rule): the landing is wrong or a hit is
        # false, so neither it nor the time to that hit counts. Since 2026-09-15 that includes a rally's last
        # shot, though 687 of ShuttleSet's 708 such last shots are net errors
        lx = ly = flight = np.nan
    distance = np.hypot(lx - hx, ly - hy)
    speed = distance / flight if flight > 0 else np.nan  # a few og_train rallies repeat a hit's frame
    # How fast the incoming shot came, previous hitter to this one: a net shot and a block of a smash
    # ("return net", a defensive shot) start from the same place
    incoming = np.hypot(hx - px, hy - py) / since_prev if since_prev > 0 else np.nan
    return [hx, hy, ox, oy, lx, ly, px, py, since_prev, flight, speed, distance, lx - hx, incoming]


def template_m(px, py):
    return ((px - TEMPLATE_CX) / TEMPLATE_W * 2 * HALF_W, (py - TEMPLATE_NET) / TEMPLATE_L * 2 * HALF_L)


def normalized_m(xy):
    """The annotator's 0-1 court fractions -> court metres (court.to_normalized backwards)."""
    return None if xy is None else ((xy[0] - 0.5) * 2 * HALF_W, (xy[1] - 0.5) * 2 * HALF_L)


def num(v):
    return float(v) if v not in ("", None) else None


# === TRAINING DATA ===
def leaked_matches(rows):
    """og_train match ids that are one of the ShuttleSet matches we evaluate on, found by shared hit frames."""
    frames = defaultdict(set)
    for r in rows:
        frames[r["match_id"]].add(int(float(r["frame_num"])))
    ours = {}
    for d in sorted(LABEL_DIR.iterdir()):
        ours[d.name] = {int(float(r["frame_num"])) for p in d.glob("set*.csv") for r in csv.DictReader(open(p, newline=""))}
    return {m: name for m, f in frames.items() for name, o in ours.items() if len(f & o) >= LEAK_SHARE * len(f)}


def training_set():
    """og_train's shots as (the inputs to features(), type, match id), leaving out the ShuttleSet matches we
    evaluate on."""
    with open(TRAIN_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    leaks = leaked_matches(rows)
    for m, name in leaks.items():
        log(f"og_train match {m} is {name}: left out of training")
    rallies = defaultdict(list)
    for r in rows:
        if r["match_id"] not in leaks:
            rallies[r["rally_id"]].append(r)
    out = []
    for shots in rallies.values():
        shots.sort(key=lambda r: int(r["ball_round"]))
        where = lambda r: (template_m(num(r["player_location_x"]), num(r["player_location_y"]))
                           if r["player_location_x"] else None)
        for i, r in enumerate(shots):
            me = where(r)
            if me is None:
                continue
            opp = template_m(num(r["opponent_location_x"]), num(r["opponent_location_y"])) if r["opponent_location_x"] else None
            # og_train's landing is ((px - 175)/82, (py - 467)/192) in the template
            land = template_m(82 * num(r["landing_x"]) + 175, 192 * num(r["landing_y"]) + 467) if r["landing_x"] else None
            frame = int(float(r["frame_num"]))
            since = (frame - int(float(shots[i - 1]["frame_num"]))) / LABEL_FPS if i else None
            flight = (int(float(shots[i + 1]["frame_num"])) - frame) / LABEL_FPS if i + 1 < len(shots) else None
            out.append(((me, opp, land, where(shots[i - 1]) if i else None, since, flight), r["type"], r["match_id"]))
    return out


def roughen(inputs, rng):
    """One shot's inputs with 1D-1F's measurement errors added, so the model doesn't lean on precision the
    pipeline doesn't have."""
    me, opp, land, prev, since, flight = inputs
    jitter = lambda p, s: None if p is None else (p[0] + rng.normal(0, s), p[1] + rng.normal(0, s))
    t = lambda v: None if v is None else max(v + rng.normal(0, NOISE_TIME), 1 / LABEL_FPS)
    if rng.random() < HIDE_PREV:
        prev, since = None, None
    if flight is not None and rng.random() < HIDE_LANDING:
        land, flight = None, None
    x, y = jitter(me, NOISE_POSITION)
    return (x, np.copysign(abs(y), me[1])), jitter(opp, NOISE_POSITION), jitter(land, NOISE_LANDING), \
        jitter(prev, NOISE_POSITION), t(since), t(flight)


def design(shots, rng=None):
    return np.array([features(*(roughen(s, rng) if rng is not None else s)) for s, _, _ in shots], float)


def fit(shots):
    """The model on the clean shots plus ROUGH_COPIES copies with the pipeline's errors added."""
    rng = np.random.default_rng(0)
    X = np.vstack([design(shots)] + [design(shots, rng) for _ in range(ROUGH_COPIES)])
    return new_model().fit(X, [t for _, t, _ in shots] * (ROUGH_COPIES + 1))


def new_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    # Gradient-boosted trees take the missing values (no previous hit, the rally's last shot) as they come
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, max_leaf_nodes=31, l2_regularization=1.0,
                                          early_stopping=True, random_state=0)


def train(cv=True):
    from sklearn.model_selection import GroupKFold
    shots = training_set()
    y, groups = np.array([t for _, t, _ in shots]), np.array([m for _, _, m in shots])
    log(f"{len(y)} shots from {len(set(groups))} og_train matches")
    if cv:
        X, classes = design(shots), np.unique(y)
        proba = np.zeros((len(y), len(classes)))
        for tr, te in GroupKFold(CV_FOLDS).split(X, y, groups):
            proba[te] = fit([shots[k] for k in tr]).predict_proba(X[te])
        report(f"og_train, {CV_FOLDS}-fold by match (ShuttleSet's own positions)", y, proba, classes)
    model = fit(shots)
    MODEL_PATH.write_bytes(pickle.dumps({"model": model, "features": FEATURES}))
    log(f"-> {MODEL_PATH}")


def load():
    if not MODEL_PATH.exists():
        raise SystemExit(f"no {MODEL_PATH}: run `shot_classify.py train` first")
    saved = pickle.loads(MODEL_PATH.read_bytes())
    if saved["features"] != FEATURES:
        raise SystemExit("the saved model has different features: run `shot_classify.py train` again")
    return saved["model"]


# === EVENTS ===
def event_rows(events):
    """Features for 1F's events. The previous hitter is the event one shot_num earlier in the same clip."""
    by_clip = {(e["source"]["match_id"], e["source"]["clip"], e["shot_num"]): e for e in events}
    X = []
    for e in events:
        cv = e["cv"]
        prev = by_clip.get((e["source"]["match_id"], e["source"]["clip"], e["shot_num"] - 1))
        # A landing where the track ends is still in the air and maps well past where it came down
        land = normalized_m(cv["landing_xy"]) if cv["landing_source"] != "track_end" else None
        # ShuttleSet times a flight to the next hit only; a floor landing has no time there
        flight = cv.get("flight_s") if cv["landing_source"] == "next_hit" else None
        X.append(features(normalized_m(cv["player_xy"]), normalized_m(cv["opponent_xy"]), land,
                          normalized_m(prev["cv"]["player_xy"]) if prev else None, cv.get("since_prev_s"), flight))
    return np.array(X, float)


def suggestion(p, classes):
    order = np.argsort(-p)[:TOP]
    return {"shot_type": str(classes[order[0]]), "p": round(float(p[order[0]]), 2),
            "top": [[str(classes[k]), round(float(p[k]), 2)] for k in order], "source": "shot_classify"}


def label():
    from feature_assemble import EVENTS_JSON
    model = load()
    events = json.loads(EVENTS_JSON.read_text())
    proba = model.predict_proba(event_rows(events))
    for e, p in zip(events, proba):
        e["model_label"] = suggestion(p, model.classes_)
    EVENTS_JSON.write_text(json.dumps(events))
    log(f"{len(events)} events labelled -> {EVENTS_JSON}")
    log("  suggested: " + ", ".join(f"{t} {n}" for t, n in Counter(e["model_label"]["shot_type"] for e in events).most_common()))
    log(f"  p >= {CONFIDENT}: {np.mean(proba.max(1) >= CONFIDENT):.0%} of events")


# === EVALUATION ===
def report(name, y, proba, classes):
    """Top-1 and top-3 accuracy, accuracy on the confident suggestions, per-type recall and precision, and the
    commonest confusions."""
    y = np.asarray(y)
    pred = classes[proba.argmax(1)]
    top = classes[np.argsort(-proba, 1)[:, :TOP]]
    sure = proba.max(1) >= CONFIDENT
    log(f"{name}: {len(y)} shots, top-1 {np.mean(pred == y):.1%}, top-{TOP} {np.mean([t in row for t, row in zip(y, top)]):.1%}; "
        f"p >= {CONFIDENT} on {sure.mean():.0%} of shots, right at {np.mean(pred[sure] == y[sure]) if sure.any() else 0:.1%} of those")
    for c in classes:
        n, m = (y == c).sum(), (pred == c).sum()
        log(f"  {c:15s} n={n:5d}  recall {np.mean(pred[y == c] == c) if n else 0:6.1%}  precision {np.mean(y[pred == c] == c) if m else 0:6.1%}")
    wrong = Counter((t, p) for t, p in zip(y, pred) if t != p).most_common(6)
    log("  most confused (true -> suggested): " + ", ".join(f"{t} -> {p} {k}" for (t, p), k in wrong))


def evaluate():
    """On each ShuttleSet match: the model given ShuttleSet's own positions and timings (how good it can be),
    then given 1F's events matched to ShuttleSet's hits (how good it is on the pipeline's measurements)."""
    from feature_assemble import build, lazy_model
    from contact_detect import image_to_court
    import court
    model = load()
    classes = model.classes_
    metas = {m["match_id"]: m for m in read_matches()}
    get_model = lazy_model()
    for d in match_dirs():
        if not (LABEL_DIR / d.name).exists() or not (d / "players").exists():
            continue
        role = "dev" if d.name == "yto2021-ms-f-axelsen-vs-ng" else "test"
        H = image_to_court(d)
        m = lambda px: None if px is None else tuple(court.apply_h(H, [px])[0])
        labels = read_labels(d)
        # From the labels
        rallies = defaultdict(list)
        for h in labels:
            rallies[(h["set"], h["rally"])].append(h)
        X, y = [], []
        for shots in rallies.values():
            shots.sort(key=lambda h: h["round"])
            for i, h in enumerate(shots):
                t = TO_PREDICTOR.get(h["type"])
                if t is None or h["hitter_px"] is None:
                    continue
                since = (h["broadcast_frame"] - shots[i - 1]["broadcast_frame"]) / LABEL_FPS if i else None
                flight = (shots[i + 1]["broadcast_frame"] - h["broadcast_frame"]) / LABEL_FPS if i + 1 < len(shots) else None
                X.append(features(m(h["hitter_px"]), m(h["opponent_px"]), m(h["landing_px"]),
                                  m(shots[i - 1]["hitter_px"]) if i else None, since, flight))
                y.append(t)
        report(f"{d.name} ({role}), ShuttleSet's positions", y, model.predict_proba(np.array(X, float)), classes)
        # From 1F's events
        events, _ = build(d, metas[d.name], get_model, strips=False)
        ours, truth = [], []
        for clip in {e["source"]["clip"] for e in events}:
            evs = sorted((e for e in events if e["source"]["clip"] == clip), key=lambda e: e["source"]["frame"])
            lab = [h for h in labels if h["file"] == clip]
            for i, j in pair_up([h["frame"] for h in lab], [e["source"]["frame"] for e in evs], TOL):
                t = TO_PREDICTOR.get(lab[i]["type"])
                if t is not None:
                    ours.append(evs[j])
                    truth.append(t)
        proba = model.predict_proba(event_rows(events))
        at = {e["id"]: k for k, e in enumerate(events)}
        report(f"{d.name} ({role}), 1F's events", truth, proba[[at[e["id"]] for e in ours]], classes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "evaluate", "label"])
    ap.add_argument("--no-cv", action="store_true", help="train: skip the cross-validation, just fit and save")
    args = ap.parse_args()
    if args.cmd == "train":
        train(cv=not args.no_cv)
    else:
        {"evaluate": evaluate, "label": label}[args.cmd]()
