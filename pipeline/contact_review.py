"""Phase 1D — check detected hits by hand, on the matches ShuttleSet doesn't label.

  .venv/bin/python contact_review.py queue      # the rallies most likely to be wrong, plus 1 random per match
  .venv/bin/python contact_review.py review     # step through their hits; every key is saved, so it resumes
  .venv/bin/python contact_review.py summary    # precision, recall, timing, and which flags predict mistakes

Review keys. The view starts on the detected hit's frame; "current" is whichever frame you've stepped to.
  y / Enter   a real hit at the current frame (step to the right frame first if it's off; the offset is saved)
  n           no hit here                u   unsure
  <- -> / a d step 1 frame               up down / w s   step 5 frames      r   back to the detected frame
  p           play from the previous hit to the next at half speed (any key stops it there)
  m           a real hit the detector missed, at the current frame
  b           undo the last verdict and go back to it
  ]           skip the rest of this rally      q / Esc   quit (everything is already saved)
"""
import argparse
import csv
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from contact_detect import pair_up
from shuttle_track import match_dirs, read_rows
from video_prep import DATA_DIR, SEG_DIR, log

QUEUE = DATA_DIR / "contact_review_queue.csv"
VERDICTS = DATA_DIR / "contact_review.csv"
LABEL_DIR = DATA_DIR / "shuttleset"   # matches with ShuttleSet labels are scored by contact_detect evaluate instead
TOL = 2                               # frames; the CoachAI rule, as in contact_detect evaluate
MIN_HITS = 4                          # shorter clips are mostly serves and pickups, not rallies
MAX_HITS = 30                         # longer clips take too long to review; a typical rally is 10-25 hits

# Why a detected hit is suspect
WEAK_KINK = 8     # px/frame velocity jump; a cut needs 3
CLOSE_GAP = 12    # frames to another contact; ShuttleSet's 1st-percentile gap between hits is 11
SLOW_OUT = 4      # px/frame; the shuttle barely moves after it
GAP_MISSING = 3   # of the 7 frames around it with no detection
EDGE = 5          # frames from the clip's start or end
FLAGS = ["weak", "close", "slow", "gap", "edge"]

VERDICT_FIELDS = ["match_id", "segment_id", "reason", "detected_frame", "verdict", "true_frame", "offset",
                  "flags", "kink", "speed_out", "reviewed_at"]
WIN = "contact review"
PLAY_FPS = 15
LEFT, RIGHT = {63234, 2424832, 65361, ord("a")}, {63235, 2555904, 65363, ord("d")}
UP, DOWN = {63232, 2490368, 65362, ord("w")}, {63233, 2621440, 65364, ord("s")}


# === QUEUE ===
def read_segment(d, row):
    name = row["file"].replace(".mp4", ".csv")
    with open(d / "contacts" / name, newline="") as f:
        contacts = list(csv.DictReader(f))
    with open(d / "tracks" / name, newline="") as f:
        track = list(csv.DictReader(f))
    return contacts, track


def flag(contacts, track):
    """Per contact, the reasons it may be wrong."""
    missing = np.array([r["detected"] != "1" for r in track])
    frames = [int(c["frame"]) for c in contacts]
    out = []
    for k, (c, f) in enumerate(zip(contacts, frames)):
        fl = []
        if float(c["kink"]) < WEAK_KINK:
            fl.append("weak")
        if any(abs(f - g) < CLOSE_GAP for j, g in enumerate(frames) if j != k):
            fl.append("close")
        if float(c["speed_out"]) < SLOW_OUT:
            fl.append("slow")
        if missing[max(0, f - 3):f + 4].sum() >= GAP_MISSING:
            fl.append("gap")
        if f < EDGE or f >= len(track) - EDGE:
            fl.append("edge")
        out.append(fl)
    return out


def build_queue(per_match, controls, add=False):
    """Per unlabelled match, the rallies with the largest share of flagged hits, plus `controls` picked at
    random from the rest. The suspects find the failure modes; the random ones estimate precision without that bias.
    With add, the new rallies go after the existing queue and none already in it is picked again."""
    existing = read_csv(QUEUE) if add else []
    taken = {(q["match_id"], int(q["segment_id"])) for q in existing}
    rng = random.Random(len(existing))
    ranked = []
    for d in match_dirs():
        if (LABEL_DIR / d.name).exists() or not (d / "contacts").exists():
            continue
        cands = []
        for r in read_rows(d):
            if not (d / "contacts" / r["file"].replace(".mp4", ".csv")).exists():
                continue
            contacts, track = read_segment(d, r)
            flags = [fl for c, fl in zip(contacts, flag(contacts, track)) if c["kind"] == "hit"]
            if MIN_HITS <= len(flags) <= MAX_HITS and (d.name, int(r["segment_id"])) not in taken:
                cands.append({"match_id": d.name, "segment_id": int(r["segment_id"]), "hits": len(flags),
                              "flagged": sum(bool(fl) for fl in flags)})
        cands.sort(key=lambda c: (-c["flagged"] / c["hits"], -c["flagged"]))
        suspects = [dict(c, reason="suspect") for c in cands[:per_match]]
        randoms = [dict(c, reason="random") for c in rng.sample(cands[per_match:], min(controls, len(cands[per_match:])))]
        ranked.append((suspects, randoms))
    # Round-robin across matches: the random ones first, as they give the gate's number if you stop early,
    # then the suspects, most flagged first
    new = ([rs[i] for i in range(controls) for _, rs in ranked if i < len(rs)]
           + [s[i] for i in range(per_match) for s, _ in ranked if i < len(s)])
    queue = existing + new
    with open(QUEUE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["match_id", "segment_id", "reason", "hits", "flagged"])
        w.writeheader()
        w.writerows(queue)
    n_s = sum(q["reason"] == "suspect" for q in new)
    log(f"{'added ' if add else ''}{len(new)} rallies ({len(new) - n_s} random, {n_s} suspect), "
        f"{sum(int(q['hits']) for q in new)} hits -> {QUEUE} ({len(queue)} rallies in all)")
    return queue


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def covered(rows, frame):
    """Whether one of a rally's verdicts already settles its detected hit at `frame`: a hit judged within TOL
    frames of it, or a real hit (confirmed or missed) put there. A change to the detector can move a cut a frame
    or two, and that shouldn't mean judging it again."""
    for r in rows:
        judged = r["verdict"] in ("hit", "false", "unsure") and abs(int(r["detected_frame"]) - frame) <= TOL
        real = r["verdict"] in ("hit", "missed") and abs(int(r["true_frame"]) - frame) <= TOL
        if judged or real:
            return True
    return False


# === REVIEW ===
def positions(track):
    """Shuttle position per frame: raw where TrackNet saw it, else the gap-filled track, else NaN."""
    xy = np.full((len(track), 2), np.nan)
    for i, r in enumerate(track):
        if r["detected"] == "1":
            xy[i] = float(r["raw_x_px"]), float(r["raw_y_px"])
        elif r["x_px"]:
            xy[i] = float(r["x_px"]), float(r["y_px"])
    return xy


def ipt(p):
    return int(round(p[0])), int(round(p[1]))


def crop(img, center, size):
    h, w = img.shape[:2]
    x = int(np.clip(center[0] - size // 2, 0, w - size))
    y = int(np.clip(center[1] - size // 2, 0, h - size))
    return img[y:y + size, x:x + size]


def put(img, text, org, scale=0.55, color=(235, 235, 235), thick=1):
    # Outline by offset copies: OpenCV 5's font gets wider with thickness, so a thicker black copy sticks out
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        cv2.putText(img, text, (org[0] + dx, org[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class Rally:
    """One queued clip: its hits and flags, the track, and every frame held as JPEG so stepping is instant."""

    def __init__(self, item):
        self.match, self.seg, self.reason = item["match_id"], int(item["segment_id"]), item["reason"]
        d = SEG_DIR / self.match
        row = next(r for r in read_rows(d) if int(r["segment_id"]) == self.seg)
        contacts, track = read_segment(d, row)
        self.hits = [dict(c, flags=fl) for c, fl in zip(contacts, flag(contacts, track)) if c["kind"] == "hit"]
        self.contact_at = {int(c["frame"]): (c["kind"], k) for k, c in enumerate(contacts, 1)}
        self.xy = positions(track)
        cap = cv2.VideoCapture(str(d / row["file"]))
        self.jpgs = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            self.jpgs.append(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])[1])
        cap.release()
        self.n = len(self.jpgs)
        self.cache = {}

    def frame(self, i):
        i = int(np.clip(i, 0, self.n - 1))
        if i not in self.cache:
            if len(self.cache) > 80:
                self.cache.clear()
            self.cache[i] = cv2.imdecode(self.jpgs[i], cv2.IMREAD_COLOR)
        return self.cache[i]


class Review:
    def __init__(self, queue, scale):
        self.queue, self.scale = queue, scale
        self.rows = read_csv(VERDICTS)
        self.history = []           # rows added this session, newest last, for undo
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = {}           # queue index -> future Rally
        self.qi, self.rally, self.k, self.cur = -1, None, 0, 0

    # --- bookkeeping ---
    def key_of(self, match, seg):
        return [r for r in self.rows if r["match_id"] == match and int(r["segment_id"]) == seg]

    def todo(self, match, seg, frames):
        """The detected hits among `frames` no verdict settles yet; none once the rally is skipped."""
        rows = self.key_of(match, seg)
        if any(r["verdict"] == "skip-rally" for r in rows):
            return []
        return [f for f in frames if not covered(rows, f)]

    def todo_hits(self):
        """Indices of the current rally's hits still to review."""
        frames = [int(h["frame"]) for h in self.rally.hits]
        left = set(self.todo(self.rally.match, self.rally.seg, frames))
        return [j for j, f in enumerate(frames) if f in left]

    def complete(self, qi):
        item = self.queue[qi]
        with open(SEG_DIR / item["match_id"] / "contacts" / f"seg_{int(item['segment_id']):04d}.csv", newline="") as f:
            hits = [int(c["frame"]) for c in csv.DictReader(f) if c["kind"] == "hit"]
        return not self.todo(item["match_id"], int(item["segment_id"]), hits)

    def save(self):
        with open(VERDICTS, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=VERDICT_FIELDS)
            w.writeheader()
            w.writerows(self.rows)

    def add(self, verdict, hit=None, frame=None):
        r = self.rally
        row = {"match_id": r.match, "segment_id": r.seg, "reason": r.reason, "verdict": verdict,
               "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "detected_frame": "", "true_frame": "", "offset": "", "flags": "", "kink": "", "speed_out": ""}
        if hit is not None:
            det = int(hit["frame"])
            row.update(detected_frame=det, flags=" ".join(hit["flags"]), kink=hit["kink"], speed_out=hit["speed_out"])
            if verdict == "hit":
                row.update(true_frame=self.cur, offset=self.cur - det)
        elif frame is not None:
            row.update(true_frame=frame)
        self.rows.append(row)
        self.history.append(row)
        self.save()

    # --- navigation ---
    def load(self, qi):
        fut = self.pending.pop(qi, None)
        if fut is None:
            log(f"loading {self.queue[qi]['match_id']} #{self.queue[qi]['segment_id']} ...")
            return Rally(self.queue[qi])
        return fut.result()

    def next_incomplete(self, qi):
        qi += 1
        while qi < len(self.queue) and self.complete(qi):
            qi += 1
        return qi

    def open(self, qi, k=None, cur=None):
        self.qi, self.rally = qi, self.load(qi)
        todo = self.todo_hits()
        self.k = k if k is not None else (todo[0] if todo else 0)
        self.cur = cur if cur is not None else int(self.rally.hits[self.k]["frame"])
        nxt = self.next_incomplete(qi)
        if nxt < len(self.queue) and nxt not in self.pending:
            self.pending[nxt] = self.pool.submit(Rally, self.queue[nxt])   # ready by the time you get there
        log(f"[{qi + 1}/{len(self.queue)}] {self.rally.match} #{self.rally.seg} ({self.rally.reason}): "
            f"{len(todo)} of {len(self.rally.hits)} hits to review")

    def advance(self):
        """Next unreviewed hit in this rally, else the next incomplete rally. False when the queue is done."""
        todo = self.todo_hits()
        if todo:
            self.k = next((j for j in todo if j > self.k), todo[0])
            self.cur = int(self.rally.hits[self.k]["frame"])
            return True
        qi = self.next_incomplete(self.qi)
        if qi >= len(self.queue):
            return False
        self.open(qi)
        return True

    def undo(self):
        if not self.history:
            return
        row = self.history.pop()
        self.rows.remove(row)
        self.save()
        seg = int(row["segment_id"])
        if (row["match_id"], seg) != (self.rally.match, self.rally.seg):
            qi = next(i for i, q in enumerate(self.queue) if q["match_id"] == row["match_id"] and int(q["segment_id"]) == seg)
            self.open(qi)
        if row["verdict"] in ("hit", "false", "unsure"):
            det = int(row["detected_frame"])
            self.k = next(j for j, h in enumerate(self.rally.hits) if int(h["frame"]) == det)
            self.cur = det
        elif row["true_frame"]:
            self.cur = int(row["true_frame"])

    # --- drawing ---
    def compose(self):
        r, h = self.rally, self.rally.hits[self.k]
        det, cur = int(h["frame"]), self.cur
        hp = (float(h["x_px"]), float(h["y_px"]))
        clean = r.frame(cur)
        img = clean.copy()
        pts = [(f, r.xy[f]) for f in range(max(0, cur - 12), min(cur + 1, len(r.xy))) if np.isfinite(r.xy[f]).all()]
        for (f0, p0), (f1, p1) in zip(pts, pts[1:]):
            if f1 - f0 == 1:
                cv2.line(img, ipt(p0), ipt(p1), (0, 220, 255), 2, cv2.LINE_AA)
        cv2.circle(img, ipt(hp), 18, (0, 0, 255), 3 if cur == det else 1, cv2.LINE_AA)
        other = r.contact_at.get(cur)
        missed = {int(x["true_frame"]) for x in self.key_of(r.match, r.seg) if x["verdict"] == "missed"}

        canvas = np.full((790, 1280, 3), 28, np.uint8)
        canvas[:540, :960] = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
        off = cur - det
        put(canvas, "DETECTED HIT FRAME" if off == 0 else f"{off:+d} frames from the detected hit",
            (14, 34), 0.9, (60, 60, 255) if off == 0 else (235, 235, 235), 2)
        if other and cur != det:
            put(canvas, f"(another detected {other[0]} here)", (14, 66), 0.6, (0, 165, 255), 2)
        if cur in missed:
            put(canvas, "(you marked a missed hit here)", (14, 94), 0.6, (0, 255, 120), 2)
        put(canvas, f"frame {cur} / {r.n - 1}", (14, 528), 0.6)

        # Zoom on the detected hit's position, current frame, without drawings
        zoom = cv2.resize(crop(clean, hp, 160), (320, 320), interpolation=cv2.INTER_CUBIC)
        for a, b in [((160, 140), (160, 150)), ((160, 170), (160, 180)), ((140, 160), (150, 160)), ((170, 160), (180, 160))]:
            cv2.line(zoom, a, b, (0, 0, 255), 1)
        canvas[:320, 960:] = zoom

        done = len(r.hits) - len(self.todo_hits())
        counts = {v: sum(x["verdict"] == v for x in self.rows) for v in ("hit", "false", "unsure", "missed")}
        info = [
            (f"rally {self.qi + 1}/{len(self.queue)}  ({r.reason})", (235, 235, 235)),
            (f"{r.match}", (190, 190, 190)),
            (f"segment {r.seg}   hit {self.k + 1}/{len(r.hits)}   done {done}", (235, 235, 235)),
            (f"detected frame {det}   now {cur} ({off:+d})", (235, 235, 235)),
            (f"flags: {', '.join(h['flags']) or 'none'}", (0, 165, 255) if h["flags"] else (150, 220, 150)),
            (f"kink {float(h['kink']):.1f}  in {float(h['speed_in']):.1f}  out {float(h['speed_out']):.1f} px/f", (190, 190, 190)),
            (f"all: {counts['hit']} hit  {counts['false']} false  {counts['unsure']} unsure  {counts['missed']} missed", (190, 190, 190)),
        ]
        for i, (text, color) in enumerate(info):
            put(canvas, text, (972, 348 + 26 * i), 0.5, color)

        # Filmstrip: 3 frames either side of the current one, all cropped on the detected hit's position
        for j, f in enumerate(range(cur - 3, cur + 4)):
            x0, y0 = 4 + j * 182, 552
            tile = (cv2.resize(crop(r.frame(f), hp, 200), (178, 178), interpolation=cv2.INTER_AREA)
                    if 0 <= f < r.n else np.zeros((178, 178, 3), np.uint8))
            canvas[y0:y0 + 178, x0:x0 + 178] = tile
            color = (60, 60, 255) if f == det else (255, 255, 255) if f == cur else (90, 90, 90)
            cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + 178, y0 + 178), color, 3 if f in (det, cur) else 1)
            put(canvas, f"{f - det:+d}", (x0 + 6, y0 + 170), 0.55, color if f in (det, cur) else (235, 235, 235))
        put(canvas, "y/Enter hit at this frame   n no hit   u unsure   m missed hit here   b undo   ] skip rally   q quit",
            (14, 758), 0.55)
        put(canvas, "<- -> step 1   up/down step 5   r back to detected   p play prev->next hit (any key stops)",
            (14, 782), 0.55, (190, 190, 190))
        return canvas

    def show(self):
        canvas = self.compose()
        if self.scale != 1:
            canvas = cv2.resize(canvas, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        cv2.imshow(WIN, canvas)

    def wait_key(self, ms=None):
        """Next key, or Esc if the window was closed."""
        end = None if ms is None else time.time() + ms / 1000
        while True:
            k = cv2.waitKeyEx(20 if ms is None else max(1, min(20, int((end - time.time()) * 1000))))
            if k != -1:
                return k
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                return 27
            if end is not None and time.time() >= end:
                return -1

    def play(self):
        hits = sorted(int(h["frame"]) for h in self.rally.hits)
        start = max([f for f in hits if f < self.cur], default=max(0, self.cur - 30))
        end = min([f for f in hits if f > self.cur], default=min(self.rally.n - 1, self.cur + 30))
        back = self.cur
        for f in range(start, end + 1):
            self.cur = f
            self.show()
            if self.wait_key(1000 // PLAY_FPS) != -1:
                return                      # stop on the frame you pressed at
        self.cur = back

    # --- main loop ---
    def run(self):
        first = self.next_incomplete(-1)
        if first >= len(self.queue):
            log("every queued rally is reviewed; run summary")
            return
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        self.open(first)
        while True:
            self.show()
            k = self.wait_key()
            ch = chr(k & 0xFF).lower() if k < 0x10000 and k != -1 else ""
            hit = self.rally.hits[self.k]
            if k == 27 or ch == "q":
                break
            elif k in LEFT:
                self.cur = max(0, self.cur - 1)
            elif k in RIGHT:
                self.cur = min(self.rally.n - 1, self.cur + 1)
            elif k in UP:
                self.cur = min(self.rally.n - 1, self.cur + 5)
            elif k in DOWN:
                self.cur = max(0, self.cur - 5)
            elif ch == "r":
                self.cur = int(hit["frame"])
            elif ch == "p":
                self.play()
            elif ch == "b":
                self.undo()
            elif ch == "m":
                self.add("missed", frame=self.cur)
            elif ch in ("y", "\r", "\n", "n", "u"):
                self.add({"n": "false", "u": "unsure"}.get(ch, "hit"), hit=hit)
                if not self.advance():
                    break
            elif ch == "]":
                self.add("skip-rally")
                if not self.advance():
                    break
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        self.pool.shutdown(wait=False, cancel_futures=True)
        log(f"saved {len(self.history)} verdicts this session -> {VERDICTS}")
        summary()


# === SUMMARY ===
def summary():
    """Scores contact_detect's current hits against your verdicts, matching within ±TOL frames: a hit at one
    you confirmed (or marked missed) is right, one at a hit you rejected is false, and anything else isn't
    judged yet. A change to the detector can move or add cuts; review then asks about just those."""
    rows = read_csv(VERDICTS)
    if not rows:
        log("no verdicts yet")
        return
    by = defaultdict(list)
    for r in rows:
        by[(r["match_id"], int(r["segment_id"]))].append(r)
    stats, errors, by_flag = defaultdict(Counter), defaultdict(list), defaultdict(Counter)
    for (m, s), rs in by.items():
        d = SEG_DIR / m
        contacts, track = read_segment(d, next(r for r in read_rows(d) if int(r["segment_id"]) == s))
        hits = [(int(c["frame"]), fl) for c, fl in zip(contacts, flag(contacts, track)) if c["kind"] == "hit"]
        real = [int(r["true_frame"]) for r in rs if r["verdict"] in ("hit", "missed")]
        # A hit confirmed more than TOL frames from where it was detected was a false alarm at that frame
        false = [int(r["detected_frame"]) for r in rs
                 if r["verdict"] == "false" or (r["verdict"] == "hit" and abs(int(r["offset"])) > TOL)]
        unsure = [int(r["detected_frame"]) for r in rs if r["verdict"] == "unsure"]
        pairs = dict(pair_up(real, [f for f, _ in hits], TOL))
        right = set(pairs.values())
        c = stats[rs[0]["reason"]]
        c["rallies"] += 1
        c["real"] += len(real)
        errors[rs[0]["reason"]] += [hits[j][0] - real[i] for i, j in pairs.items()]
        for j, (f, fl) in enumerate(hits):
            near = lambda frames: any(abs(f - x) <= TOL for x in frames)
            v = "right" if j in right else "false" if near(false) else "unsure" if near(unsure) else "unjudged"
            c[v] += 1
            for name in fl or ["none"]:
                by_flag[name][v] += 1
    for reason in ("random", "suspect"):
        c = stats.get(reason)
        if not c:
            continue
        log(f"{reason} rallies ({c['rallies']}): {c['right']} right, {c['false']} false, "
            f"{c['unjudged']} not judged yet, {c['unsure']} unsure")
        log(f"  precision (±{TOL}) {c['right'] / max(1, c['right'] + c['false']):.1%}   "
            f"recall {c['right'] / max(1, c['real']):.1%} of the {c['real']} real hits you confirmed or marked missed")
        e = np.abs(errors[reason])
        if len(e):
            log(f"  timing: {np.mean(e == 0):.0%} exact, {np.mean(e <= 1):.0%} within 1 frame")
    log("false rate by flag (all rallies):")
    for name in FLAGS + ["none"]:
        c = by_flag[name]
        if c["right"] + c["false"]:
            log(f"  {name:6s} {c['false']:4d} / {c['right'] + c['false']:4d} false ({c['false'] / (c['right'] + c['false']):.0%})")
    if sum(c["unjudged"] for c in stats.values()):
        log("run review to judge the hits not judged yet")
    log("the random rallies are the unbiased estimate for the gate; the suspects show where it fails")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("queue")
    q.add_argument("--per-match", type=int, default=4, help="suspect rallies per match")
    q.add_argument("--random", type=int, default=1, help="random control rallies per match")
    q.add_argument("--add", action="store_true", help="append new rallies to the queue instead of replacing it")
    sub.add_parser("review").add_argument("--scale", type=float, default=1.0, help="window size, e.g. 0.8 on a small screen")
    sub.add_parser("summary")
    args = ap.parse_args()
    if args.cmd == "queue":
        build_queue(args.per_match, args.random, args.add)
    elif args.cmd == "review":
        Review(read_csv(QUEUE) or build_queue(4, 1), args.scale).run()
    else:
        summary()
