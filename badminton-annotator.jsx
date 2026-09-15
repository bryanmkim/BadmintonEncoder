import { useState, useEffect, useCallback, useRef, useMemo } from "react";

// === CONSTANTS ===
// Shot classes exactly as they appear in BadmintonShotPredictor's `type` column
const SHOT_TYPES = [
  "short service", "long service", "net shot", "lob", "clear",
  "drop", "smash", "drive", "push/rush", "defensive shot"
];
const SHOT_KEYS = Object.fromEntries(SHOT_TYPES.map((st, i) => [(i + 1) % 10, st])); // keys 1–9, then 0
const SHOT_COLORS = {
  "short service":"#60a5fa","long service":"#a78bfa","net shot":"#34d399","lob":"#fbbf24",
  "clear":"#f87171","drop":"#fb923c","smash":"#ef4444","drive":"#38bdf8","push/rush":"#4ade80","defensive shot":"#e879f9"
};
// Only the review is stored, by event id: { annotation, gap_before }. The events themselves come from
// EVENTS_URL; thousands of them with their measurements would overflow localStorage.
// v4 added gap_before; v3 stored bare annotations and is read once as a fallback.
const STORAGE_KEY = "annotator:review:v4";
const LEGACY_STORAGE_KEY = "annotator:annotations:v3";
const POSITION_KEY = "annotator:position:v1"; // { id, filter, match } of the event on screen, so a reload continues there
const EVENTS_URL = "/data/events.json"; // written by pipeline/feature_assemble.py (Phase 1F)

// === MOCK DATA (used when EVENTS_URL isn't there) ===
const generateMockEvents = () => {
  const matches = [
    { name: "Axelsen vs Shi Yuqi - All England 2026", players: ["Viktor Axelsen", "Shi Yuqi"] },
    { name: "An Se Young vs Yamaguchi - BWF Finals 2025", players: ["An Se Young", "Akane Yamaguchi"] },
  ];
  const events = [];
  for (let i = 0; i < 40; i++) {
    const matchIdx = i < 20 ? 0 : 1;
    const rallyId = Math.floor((i % 20) / 4) + 1;
    const shotNum = (i % 4) + 1;
    const hitter = shotNum % 2 === 1 ? 1 : 2;
    const shotType = SHOT_TYPES[Math.floor(Math.random() * SHOT_TYPES.length)];
    const playerX = 0.3 + Math.random() * 0.4;
    const playerY = hitter === 1 ? 0.6 + Math.random() * 0.3 : 0.1 + Math.random() * 0.3;
    const landX = 0.15 + Math.random() * 0.7;
    const landY = hitter === 1 ? 0.05 + Math.random() * 0.4 : 0.55 + Math.random() * 0.4;
    events.push({
      id: `evt_${i}`,
      match: matches[matchIdx].name,
      players: matches[matchIdx].players, // [P1, P2]; hitting_player indexes into this
      rally: rallyId,
      shot_num: shotNum,
      frame_time: `${Math.floor(i * 3.2)}:${String(Math.floor(Math.random()*60)).padStart(2,'0')}`,
      cv: { player_xy: [playerX, playerY], opponent_xy: [1-playerX, 1-playerY], landing_xy: [landX, landY], speed: 50 + Math.floor(Math.random() * 300), trajectory_angle: Math.floor(Math.random() * 180) },
      claude_label: { shot_type: shotType, hitting_player: hitter, confidence: Math.random() > 0.3 ? "high" : "medium", reasoning: `Trajectory angle ${Math.floor(Math.random()*50+10)}° with ${hitter === 1 ? "downward" : "upward"} motion. Player stance and shuttle speed (${50+Math.floor(Math.random()*300)} km/h) consistent with ${shotType.toLowerCase()}.` },
      annotation: null
    });
  }
  return events;
};

// === 3D COURT ===
// World units are metres: X across the court, Y along it (net at Y=0, near baseline at +Y), Z up.
const COURT = { halfW: 3.05, halfL: 6.7, singles: 2.59, shortSvc: 1.98, longSvcDbl: 5.94, netPost: 1.55, netMid: 1.524, netBottom: 0.76 };
const toWorld = ([x, y], z = 0) => [(x - 0.5) * COURT.halfW * 2, (y - 0.5) * COURT.halfL * 2, z];
const netTopAt = X => COURT.netMid + (COURT.netPost - COURT.netMid) * Math.min(1, Math.abs(X) / COURT.halfW);

const COURT_LINES = [
  ...[-COURT.halfW, -COURT.singles, COURT.singles, COURT.halfW].map(x => [[x, -COURT.halfL, 0], [x, COURT.halfL, 0]]),
  ...[-COURT.halfL, -COURT.longSvcDbl, -COURT.shortSvc, COURT.shortSvc, COURT.longSvcDbl, COURT.halfL].map(y => [[-COURT.halfW, y, 0], [COURT.halfW, y, 0]]),
  [[0, -COURT.halfL, 0], [0, -COURT.shortSvc, 0]],
  [[0, COURT.shortSvc, 0], [0, COURT.halfL, 0]],
];

// Heuristic flight profile per shot: contact height (m), target apex (m, 0 = straight down), air-drag easing.
const SHOT_PROFILES = {
  "short service":  { contact: 1.0, apex: 1.9, drag: 1.4 },
  "long service":   { contact: 1.0, apex: 6.0, drag: 2.4 },
  "net shot":       { contact: 1.1, apex: 1.8, drag: 1.2 },
  "lob":            { contact: 0.6, apex: 5.5, drag: 2.4 },
  "clear":          { contact: 2.6, apex: 6.5, drag: 2.6 },
  "drop":           { contact: 2.6, apex: 2.7, drag: 1.8 },
  "smash":          { contact: 2.9, apex: 0,   drag: 1.0 },
  "drive":          { contact: 1.5, apex: 1.9, drag: 1.2 },
  "push/rush":      { contact: 1.6, apex: 0,   drag: 1.2 },
  "defensive shot": { contact: 0.7, apex: 2.4, drag: 1.6 },
};
const NET_CLEARANCE = 1.65;

// Returns the shuttle path as a function of normalised flight time t ∈ [0,1].
const buildTrajectory = (cv, shotType, steps = 48) => {
  const prof = SHOT_PROFILES[shotType] || SHOT_PROFILES.drive;
  const [sx, sy] = toWorld(cv.player_xy), [ex, ey] = toWorld(cv.landing_xy);
  const k = prof.drag, norm = 1 - Math.exp(-k);
  const ground = t => (1 - Math.exp(-k * t)) / norm; // horizontal progress decelerates under drag
  // Height is a quadratic Bezier from contact height to the floor: z(t) = h0(1-t)² + 2c·t(1-t)
  const h0 = prof.contact;
  let c = prof.apex > h0 ? prof.apex + Math.sqrt(prof.apex * prof.apex - prof.apex * h0) : h0 * 0.5;
  let tNet = null;
  if (Math.sign(sy) !== Math.sign(ey)) {
    tNet = -Math.log(1 - (sy / (sy - ey)) * norm) / k;
    c = Math.max(c, (NET_CLEARANCE - h0 * (1 - tNet) ** 2) / (2 * tNet * (1 - tNet)));
  }
  const at = t => {
    const g = ground(t);
    return [sx + (ex - sx) * g, sy + (ey - sy) * g, h0 * (1 - t) ** 2 + 2 * c * t * (1 - t)];
  };
  const ts = Array.from({ length: steps + 1 }, (_, i) => i / steps);
  const points = ts.map(at);
  const segments = tNet == null ? [points] : [[...ts.filter(t => t < tNet), tNet].map(at), [tNet, ...ts.filter(t => t > tNet)].map(at)];
  const apex = points.reduce((a, p) => (p[2] > a[2] ? p : a));
  const netPt = tNet == null ? null : at(tNet);
  return { at, points, segments, apex, contact: h0, netClearance: netPt && netPt[2] - netTopAt(netPt[0]) };
};

const VIEWS = { Broadcast: { yaw: 0, pitch: 0.5 }, Side: { yaw: Math.PI / 2, pitch: 0.22 }, Top: { yaw: 0, pitch: 1.54 } };
const CAM_DIST = 30, CAM_TARGET_Z = 1.2;

// Perspective camera orbiting the court centre, auto-fitted to the SVG box.
const makeProjector = ({ yaw, pitch }, W, H) => {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const raw = ([X, Y, Z]) => {
    const x = X * cy - Y * sy;
    const y = X * sy + Y * cy - CAM_DIST * cp;
    const z = Z - CAM_TARGET_Z - CAM_DIST * sp;
    const depth = -cp * y - sp * z;
    return [x / depth, (sp * y - cp * z) / depth, depth];
  };
  // Fit the floor plus the envelope where high arcs (clears, lifts) peak, so scale stays stable between shots
  const fit = [];
  for (const X of [-COURT.halfW, COURT.halfW]) {
    for (const Y of [-COURT.halfL, COURT.halfL]) fit.push(raw([X, Y, 0]));
    for (const Y of [-4, 4]) fit.push(raw([X, Y, 6.5]));
  }
  const us = fit.map(p => p[0]), vs = fit.map(p => p[1]);
  const u0 = Math.min(...us), u1 = Math.max(...us), v0 = Math.min(...vs), v1 = Math.max(...vs);
  const box = { x: 12, y: 18, w: W - 24, h: H - 44 };
  const scale = Math.min(box.w / (u1 - u0), box.h / (v1 - v0));
  const ox = box.x + (box.w - (u1 - u0) * scale) / 2 - u0 * scale;
  const oy = box.y + box.h - v1 * scale; // bottom-aligned: spare room goes above, where the arcs are
  return {
    project: p => { const [u, v, depth] = raw(p); return [ox + u * scale, oy + v * scale, depth]; },
    size: (metres, depth) => (metres * scale) / depth,
  };
};

const FLIGHT_MS = 1400, HOLD_MS = 700;

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// Gaussian elimination with partial pivoting, for the small systems below
const solve = (A, b) => {
  const n = b.length, M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
};

// The floor's image under a perspective camera is a homography: fit it on the four court corners and invert it,
// so a point on the screen gives the court position (metres) under it
const screenToFloor = P => {
  const A = [], b = [];
  [[-COURT.halfW, -COURT.halfL], [COURT.halfW, -COURT.halfL], [COURT.halfW, COURT.halfL], [-COURT.halfW, COURT.halfL]].forEach(([X, Y]) => {
    const [u, v] = P([X, Y, 0]);
    A.push([u, v, 1, 0, 0, 0, -u * X, -v * X]); b.push(X);
    A.push([0, 0, 0, u, v, 1, -u * Y, -v * Y]); b.push(Y);
  });
  const h = solve(A, b);
  return (u, v) => {
    const w = h[6] * u + h[7] * v + 1;
    return [(h[0] * u + h[1] * v + h[2]) / w, (h[3] * u + h[4] * v + h[5]) / w];
  };
};

// With editLanding, a click or drag on the court places the landing; it's handed to onLanding on release
const Court3D = ({ cv, shotType, hitter, players, shotColor, editLanding = false, onLanding }) => {
  const W = 420, H = 440;
  const [view, setView] = useState(VIEWS.Broadcast);
  const [t, setT] = useState(0);
  const [draft, setDraft] = useState(null); // landing being dragged, normalised
  const drag = useRef(null);
  const shown = useMemo(() => (draft ? { ...cv, landing_xy: draft } : cv), [cv, draft]);
  const traj = useMemo(() => buildTrajectory(shown, shotType), [shown, shotType]);
  const { project: P, size } = useMemo(() => makeProjector(view, W, H), [view]);
  const toFloor = useMemo(() => screenToFloor(P), [P]);

  useEffect(() => setDraft(null), [cv]);                         // a new shot, or the landing was committed
  useEffect(() => { if (editLanding) setView(VIEWS.Top); }, [editLanding]); // placing is easiest from above

  // Loop the shuttle along its arc, restarting whenever the shot changes
  useEffect(() => {
    let raf;
    const start = performance.now();
    const tick = now => {
      setT(Math.min(1, ((now - start) % (FLIGHT_MS + HOLD_MS)) / FLIGHT_MS));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [traj]);

  const xy = p => P(p).slice(0, 2).map(n => n.toFixed(1));
  const path = pts => pts.map((p, i) => `${i ? "L" : "M"}${xy(p).join(",")}`).join("");
  const poly = pts => pts.map(p => xy(p).join(",")).join(" ");
  const ring = ([x, y], r, n = 28) => Array.from({ length: n }, (_, i) => [x + r * Math.cos((2 * Math.PI * i) / n), y + r * Math.sin((2 * Math.PI * i) / n), 0]);
  const floorOf = pts => pts.map(([x, y]) => [x, y, 0]);

  // Painter's order: floor, then objects on the far side of the net, the net, then the near side
  const camSide = Math.cos(view.yaw) >= 0 ? 1 : -1;
  const layers = { far: [], near: [] };
  const add = (Y, depth, el) => layers[Y * camSide > 0 ? "near" : "far"].push({ depth, el });
  const floor = [];

  const player = (xy2d, color, label, key) => {
    const [X, Y] = toWorld(xy2d);
    const [bx, by, depth] = P([X, Y, 0]);
    const [hx, hy] = P([X, Y, 1.65]);
    const [lx, ly] = P([X, Y, 2.15]);
    floor.push(<polygon key={`${key}-base`} points={poly(ring([X, Y], 0.35))} fill="#000" opacity={0.3}/>);
    add(Y, depth, (
      <g key={key}>
        <line x1={bx} y1={by} x2={hx} y2={hy} stroke={color} strokeWidth={Math.max(2, size(0.22, depth))} strokeLinecap="round"/>
        <circle cx={hx} cy={hy} r={Math.max(3, size(0.17, depth))} fill={color} stroke="#fff" strokeWidth={1.2}/>
        <text x={lx} y={ly} textAnchor="middle" fill="#fff" fontSize={10} fontWeight="bold"
          stroke="#0f172a" strokeWidth={3} paintOrder="stroke">{label}</text>
      </g>
    ));
  };
  // Hitter in blue, opponent in red, each labelled with their name (P1/P2 = players[0]/[1] when there are none)
  const name = n => players?.[n - 1] ?? `P${n}`;
  player(cv.player_xy, "#3b82f6", name(hitter), "hitter");
  player(cv.opponent_xy, "#ef4444", name(hitter === 1 ? 2 : 1), "opponent");

  // Trajectory: shadow + landing on the floor, arc split at the net for correct layering
  const [lX, lY] = toWorld(shown.landing_xy);
  floor.push(
    <path key="shadow" d={path(floorOf(traj.points))} stroke="#000" strokeWidth={2} strokeDasharray="4,3" fill="none" opacity={0.35}/>,
    <polygon key="land-outer" points={poly(ring([lX, lY], 0.35))} fill={shotColor} opacity={0.35}/>,
    <polygon key="land-inner" points={poly(ring([lX, lY], 0.12))} fill={shotColor}/>,
  );
  traj.segments.forEach((seg, i) => {
    const mid = seg[Math.floor(seg.length / 2)];
    add(mid[1], P(mid)[2], (
      <g key={`seg${i}`} fill="none" strokeLinecap="round">
        <path d={path(seg)} stroke={shotColor} strokeWidth={6} opacity={0.15}/>
        <path d={path(seg)} stroke={shotColor} strokeWidth={2.2}/>
      </g>
    ));
  });
  if (traj.apex[2] > traj.contact + 0.15) {
    const [ax, ay, depth] = P(traj.apex);
    const [fx, fy] = P([traj.apex[0], traj.apex[1], 0]);
    add(traj.apex[1], depth + 0.01, (
      <g key="apex">
        <line x1={ax} y1={ay} x2={fx} y2={fy} stroke={shotColor} strokeWidth={1} strokeDasharray="2,3" opacity={0.6}/>
        <text x={ax + 6} y={ay - 4} fill={shotColor} fontSize={10} fontWeight={600}>{traj.apex[2].toFixed(1)} m</text>
      </g>
    ));
  }

  // Animated shuttle with a drop line to its floor shadow
  const s = traj.at(t);
  const [shx, shy, shDepth] = P(s);
  const [sfx, sfy] = P([s[0], s[1], 0]);
  floor.push(<polygon key="shuttle-shadow" points={poly(ring([s[0], s[1]], 0.09, 12))} fill="#000" opacity={0.5}/>);
  add(s[1], shDepth - 0.01, (
    <g key="shuttle">
      <line x1={shx} y1={shy} x2={sfx} y2={sfy} stroke="#fff" strokeWidth={0.8} opacity={0.25}/>
      <circle cx={shx} cy={shy} r={Math.max(3, size(0.08, shDepth))} fill="#fff" stroke={shotColor} strokeWidth={2}/>
    </g>
  ));

  const strands = Array.from({ length: 13 }, (_, i) => -3 + i * 0.5);
  const net = (
    <g key="net">
      <polygon points={poly([[-COURT.halfW, 0, COURT.netBottom], [COURT.halfW, 0, COURT.netBottom], [COURT.halfW, 0, COURT.netPost], [0, 0, COURT.netMid], [-COURT.halfW, 0, COURT.netPost]])} fill="#e2e8f0" fillOpacity={0.12}/>
      {strands.map(X => <path key={X} d={path([[X, 0, COURT.netBottom], [X, 0, netTopAt(X)]])} stroke="#e2e8f0" strokeWidth={0.5} opacity={0.2}/>)}
      <path d={path([[-COURT.halfW, 0, COURT.netPost], [0, 0, COURT.netMid], [COURT.halfW, 0, COURT.netPost]])} stroke="#fff" strokeWidth={2} fill="none"/>
      {[-COURT.halfW, COURT.halfW].map(X => <path key={X} d={path([[X, 0, 0], [X, 0, COURT.netPost]])} stroke="#cbd5e1" strokeWidth={2.5} strokeLinecap="round"/>)}
    </g>
  );

  const byDepth = list => list.sort((a, b) => b.depth - a.depth).map(d => d.el);
  const apron = [[-COURT.halfW - 0.8, -COURT.halfL - 0.9, 0], [COURT.halfW + 0.8, -COURT.halfL - 0.9, 0], [COURT.halfW + 0.8, COURT.halfL + 0.9, 0], [-COURT.halfW - 0.8, COURT.halfL + 0.9, 0]];
  const surface = [[-COURT.halfW, -COURT.halfL, 0], [COURT.halfW, -COURT.halfL, 0], [COURT.halfW, COURT.halfL, 0], [-COURT.halfW, COURT.halfL, 0]];

  // Client px -> normalised court position under the pointer (a little way off court is allowed, for shots out)
  const floorAt = e => {
    const r = e.currentTarget.getBoundingClientRect();
    const [X, Y] = toFloor(((e.clientX - r.left) * W) / r.width, ((e.clientY - r.top) * H) / r.height);
    return [clamp(X / (2 * COURT.halfW) + 0.5, -0.3, 1.3), clamp(Y / (2 * COURT.halfL) + 0.5, -0.3, 1.3)];
  };
  const onPointerDown = e => {
    e.currentTarget.setPointerCapture(e.pointerId);
    if (editLanding) {
      drag.current = { landing: floorAt(e) };
      setDraft(drag.current.landing);
      return;
    }
    drag.current = { x: e.clientX, y: e.clientY };
  };
  const onPointerMove = e => {
    if (!drag.current) return;
    if (drag.current.landing) {
      drag.current.landing = floorAt(e);
      setDraft(drag.current.landing);
      return;
    }
    const dx = e.clientX - drag.current.x, dy = e.clientY - drag.current.y;
    drag.current = { x: e.clientX, y: e.clientY };
    setView(v => ({ yaw: v.yaw - dx * 0.01, pitch: Math.min(1.54, Math.max(0.08, v.pitch + dy * 0.008)) }));
  };
  const endDrag = () => {
    if (drag.current?.landing && onLanding) onLanding(drag.current.landing); // committed once, on release
    drag.current = null;
  };

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={endDrag} onPointerCancel={endDrag}
        onDoubleClick={() => !editLanding && setView(VIEWS.Broadcast)}
        style={{ width: "100%", background: "#1a2332", borderRadius: 8, cursor: editLanding ? "crosshair" : drag.current ? "grabbing" : "grab",
          touchAction: "none", userSelect: "none", display: "block", outline: editLanding ? "2px solid #f59e0b" : "none" }}>
        <polygon points={poly(apron)} fill="#22472a"/>
        <polygon points={poly(surface)} fill="#2d5a27"/>
        {COURT_LINES.map(([a, b], i) => <path key={i} d={path([a, b])} stroke="#fff" strokeWidth={1.2} opacity={0.8}/>)}
        {floor}
        {byDepth(layers.far)}
        {net}
        {byDepth(layers.near)}
        <text x={10} y={14} fill={editLanding ? "#f59e0b" : "#475569"} fontSize={editLanding ? 11 : 9} fontWeight={editLanding ? 600 : 400}>
          {editLanding ? "click or drag on the court to place the landing · L when done" : "drag to orbit · double-click to reset"}
        </text>
        <text x={W / 2} y={H - 8} textAnchor="middle" fill="#94a3b8" fontSize={10}>
          {cv.speed ?? "—"} km/h · {cv.trajectory_angle ?? "—"}° · contact {traj.contact.toFixed(1)} m · apex {traj.apex[2].toFixed(1)} m
          {traj.netClearance != null && ` · net +${traj.netClearance.toFixed(2)} m`}
        </text>
      </svg>
      <div style={{ display: "flex", gap: 4, marginTop: 6 }}>
        {Object.entries(VIEWS).map(([name, v]) => (
          <button key={name} onClick={() => setView(v)}
            style={{ ...btnStyle("#1e293b"), padding: "3px 10px", fontSize: 11,
              color: view.yaw === v.yaw && view.pitch === v.pitch ? "#f1f5f9" : "#64748b" }}>
            {name}
          </button>
        ))}
      </div>
    </div>
  );
};

// === BADMINTONSHOTPREDICTOR CSV ===
// Same 7 columns as BadmintonShotPredictor/train.csv, which its main.py trains on directly.
const PREDICTOR_COLUMNS = ["rally_id", "ball_round", "player", "type", "landing_x", "landing_y", "rally_length"];

// Predictor player ids (0–34). Its CSVs are anonymised, but ShuttleSet22's preprocess_data.py numbers players
// by first appearance in its set/match.csv (pd.unique over winner, loser, row by row), and in every set CSV
// player A is the match winner. Rebuilt that way, these ids matched the winner/loser ids of all 58 train, val
// and test matches and 30,162 of 30,172 og_train.csv shots. Names are matched ignoring case, spaces and
// punctuation (playerKey), so "Shi Yu Qi" finds "SHI Yuqi". A rally with anyone else is left out of the export:
// the predictor's player_embedding has rows for these 35 only (36 with padding), and a new id would crash it.
const PLAYER_IDS = {
  "NG Ka Long Angus": 0, "LEE Cheuk Yiu": 1, "An Se Young": 2, "Carolina MARIN": 3, "Anthony Sinisuka GINTING": 4,
  "LEE Zii Jia": 5, "KIDAMBI Srikanth": 6, "PUSARLA V. Sindhu": 7, "Pornpawee CHOCHUWONG": 8, "Anders ANTONSEN": 9,
  "Viktor AXELSEN": 10, "Lakshya SEN": 11, "LOH Kean Yew": 12, "Kunlavut VITIDSARN": 13, "Akane YAMAGUCHI": 14,
  "HE Bingjiao": 15, "CHEN Yufei": 16, "Kento MOMOTA": 17, "WANG Zhi Yi": 18, "Jonatan CHRISTIE": 19,
  "Ratchanok INTANON": 20, "Chico Aura DWI WARDOYO": 21, "LU Guang Zu": 22, "Gregoria Mariska TUNJUNG": 23,
  "ZHAO Jun Peng": 24, "PRANNOY H. S.": 25, "Kenta NISHIMOTO": 26, "Busanan ONGBAMRUNGPHAN": 27,
  "Supanida KATETHONG": 28, "Aakarshi KASHYAP": 29, "LI Shi Feng": 30, "Kodai NARAOKA": 31, "SHI Yuqi": 32,
  "LIEW Daren": 33, "Rasmus GEMKE": 34,
};
const playerKey = name => name.toLowerCase().replace(/[^a-z]/g, "");

// Keeps annotator rally ids clear of the predictor's (0–4938) so the CSVs can be concatenated
const RALLY_ID_OFFSET = 10000;

// Predictor coordinates live on ShuttleSet's court template: landing = ((px − 175) / 82, (py − 467) / 192).
// Measured from the predictor's in/out landing_area labels, the doubles court spans template
// x 27.5–327.5 and y 150–810 (net at 480, far side at the top), i.e. 49.25 template px per metre.
const TEMPLATE = { centerX: 177.5, netY: 480, pxPerMetre: 49.25 };
const toPredictorXY = xy => {
  const [X, Y] = toWorld(xy);
  return [
    (TEMPLATE.centerX + X * TEMPLATE.pxPerMetre - 175) / 82,
    (TEMPLATE.netY + Y * TEMPLATE.pxPerMetre - 467) / 192,
  ];
};

// The predictor's RallyDataset reads each rally's rows in order and learns shot -> next shot; it ignores
// ball_round and rally_length, and its loss skips the first 3 positions. So a rally may start mid-way,
// but a piece needs 5 shots to give it a single target.
const MIN_PIECE_SHOTS = 5;

// Splits one fully reviewed rally (shots sorted by shot_num) into the pieces the predictor can learn from.
// A false hit ("not a shot") is dropped, and the shot before it takes its landing: a returned shot's
// landing is where the next shot was played, which the dropped event's landing already is. A missing shot
// (gap_before) splits the rally, and the shot before the gap is dropped too: its landing is where the shot
// after the missing one was played, and the transition across the gap never happened.
// A landing on the hitter's own half is never accepted: the landing is wrong or a hit was false. That includes
// a rally's last shot (the user's call, 2026-09-15), though in ShuttleSet most such last shots are net errors
const onOwnHalf = (hitterXY, landingXY) => (hitterXY[1] - 0.5) * (landingXY[1] - 0.5) > 0;

const rallyPieces = shots => {
  const parts = [[]];
  const landing = new Map(); // event id -> landing to export, where it differs from the event's own
  let notShots = 0;
  shots.forEach(s => {
    if (s.gap_before && parts[parts.length - 1].length) {
      parts[parts.length - 1].pop();
      parts.push([]);
    }
    const part = parts[parts.length - 1];
    if (s.annotation.not_shot) {
      notShots++;
      if (part.length) landing.set(part[part.length - 1].id, s.landing_fix ?? s.cv.landing_xy);
      return;
    }
    part.push(s);
  });
  // A shot landing on its own half is left out, splitting the rally there, until the review fixes it
  // (L moves the landing, X on the false hit passes the landing on)
  const landingOf = s => s.landing_fix ?? landing.get(s.id) ?? s.cv.landing_xy;
  const out = [];
  let ownHalf = 0;
  parts.forEach(part => {
    let cur = [];
    part.forEach(s => {
      if (onOwnHalf(s.cv.player_xy, landingOf(s))) { ownHalf++; out.push(cur); cur = []; return; }
      cur.push(s);
    });
    out.push(cur);
  });
  return { parts: out, landing, notShots, ownHalf };
};

const toPredictorCsv = events => {
  const playerIds = Object.fromEntries(Object.entries(PLAYER_IDS).map(([name, id]) => [playerKey(name), id]));
  const known = name => playerKey(name) in playerIds;
  const unknownPlayers = [...new Set(events.flatMap(e => e.players).filter(name => !known(name)))];

  const rallies = new Map();
  events.forEach(e => {
    const key = `${e.match}|${e.rally}`;
    if (!rallies.has(key)) rallies.set(key, []);
    rallies.get(key).push(e);
  });

  const rows = [];
  let complete = 0, pieces = 0, short = 0, notShots = 0, outsiders = 0, ownHalf = 0;
  [...rallies.values()].forEach((shots, i) => {
    // Fully reviewed rallies only: an unreviewed shot could be a false hit or hide a missing one
    if (!shots.every(s => s.annotation)) return;
    if (!shots[0].players.every(known)) { outsiders++; return; }
    complete++;
    const split = rallyPieces([...shots].sort((a, b) => a.shot_num - b.shot_num));
    notShots += split.notShots;
    ownHalf += split.ownHalf;
    split.parts.forEach((part, p) => {
      if (part.length < MIN_PIECE_SHOTS) { if (part.length) short++; return; }
      pieces++;
      part.forEach((s, k) => {
        // A landing you moved wins over the pipeline's, and over one inherited from a dropped false hit
        const [x, y] = toPredictorXY(s.landing_fix ?? split.landing.get(s.id) ?? s.cv.landing_xy);
        const hitter = s.players[s.annotation.hitting_player - 1];
        // rally * 100 + piece keeps every id stable as other rallies are finished
        rows.push([RALLY_ID_OFFSET + i * 100 + p, k + 1, playerIds[playerKey(hitter)], s.annotation.shot_type, x, y, part.length]);
      });
    });
  });

  return {
    csv: [PREDICTOR_COLUMNS, ...rows].map(r => r.join(",")).join("\n") + "\n",
    rallies: pieces,
    reviewed: complete,
    short,
    ownHalf,
    notShots,
    shots: rows.length,
    unknownPlayers,
    outsiders,
  };
};

// === FRAME STRIP ===
// Phase 1F writes one image per event: frames f-2..f+2 side by side, cropped on the hitter and the shuttle.
// Only the middle three are shown, at 5/3 width shifted left by one frame, so each is large enough to read.
const STRIP_OFFSETS = [-1, 0, 1];
const FrameStrip = ({ event }) => event.frames ? (
  <div style={{ marginBottom:8 }}>
    <div style={{ position:"relative", overflow:"hidden", borderRadius:6 }}>
      <img src={event.frames} alt="frames around the contact" style={{ width:"166.667%", marginLeft:"-33.333%", display:"block" }}/>
      <div style={{ position:"absolute", top:0, bottom:0, left:"33.333%", width:"33.333%", border:"2px solid #3b82f6", borderRadius:4, pointerEvents:"none" }}/>
    </div>
    <div style={{ display:"flex", marginTop:2 }}>
      {STRIP_OFFSETS.map(offset => (
        <span key={offset} style={{ flex:1, textAlign:"center", fontSize:10, color: offset===0 ? "#3b82f6" : "#64748b", fontWeight: offset===0 ? 600 : 400 }}>
          {offset===0 ? "CONTACT" : `f${offset>0?"+":""}${offset}`}
        </span>
      ))}
    </div>
  </div>
) : (
  <div style={{ display:"flex", gap:4, marginBottom:8 }}>
    {STRIP_OFFSETS.map(offset => (
      <div key={offset} style={{
        flex:1, aspectRatio:"16/9", background: offset===0 ? "#1e3a5f" : "#1a1a2e",
        border: offset===0 ? "2px solid #3b82f6" : "1px solid #334155",
        borderRadius:6, display:"flex", alignItems:"center", justifyContent:"center", flexDirection:"column", position:"relative"
      }}>
        <span style={{color:"#64748b",fontSize:10}}>f{offset>=0?"+":""}{offset}</span>
        {offset===0 && <span style={{position:"absolute",bottom:2,fontSize:8,color:"#3b82f6",fontWeight:600}}>CONTACT</span>}
      </div>
    ))}
  </div>
);

const StripPanel = ({ event, title, color }) => (
  <div>
    <div style={{ fontSize:12, fontWeight:600, color, marginBottom:3, whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>{title}</div>
    <FrameStrip event={event} />
  </div>
);

const inFilter = (e, filter) => filter === "all" ? true : filter === "pending" ? !e.annotation : !!e.annotation;
const inMatch = (e, match) => match === "all" || e.match === match;

// Who hit it: Claude's call once 1G has labelled the event, else the pipeline's (1E side + 1F shirt identity)
const hitterOf = e => e.claude_label?.hitting_player ?? e.cv.hitting_player ?? 1;
// The shot type to confirm: Claude's label (1G) when there is one, else the ShuttleSet classifier's (pipeline/shot_classify.py)
const suggestionOf = e => e.claude_label ?? e.model_label ?? null;
const CONFIDENT_P = 0.7; // a classifier suggestion this likely shows green (shot_classify.CONFIDENT)

const loadEvents = async () => {
  try {
    const res = await fetch(EVENTS_URL);
    if (res.ok) return await res.json(); // without the file, the dev server answers with index.html, which fails here
  } catch {}
  return generateMockEvents();
};

// Saved review by event id: { annotation, gap_before }, falling back to v3's bare annotations
const loadSaved = async () => {
  try {
    const s = await window.storage.get(STORAGE_KEY);
    if (s) return JSON.parse(s.value);
    const old = await window.storage.get(LEGACY_STORAGE_KEY);
    if (old) return Object.fromEntries(Object.entries(JSON.parse(old.value)).map(([id, annotation]) => [id, { annotation }]));
  } catch {}
  // A browser without the review (storage cleared, another port) picks it up from review.json on disk
  try {
    const res = await fetch("/review.json");
    if (res.ok) return await res.json();
  } catch {}
  return {};
};

// === MAIN APP ===
export default function BadmintonAnnotator() {
  const [events, setEvents] = useState([]);
  const [idx, setIdx] = useState(0);
  const [filter, setFilter] = useState("all"); // all | pending | done
  const [match, setMatch] = useState("all");   // "all" or one match's name: review one match at a time
  const [showExport, setShowExport] = useState(false);
  const [editLanding, setEditLanding] = useState(false); // L: click the court to move the landing
  const containerRef = useRef(null);
  const [csvSaved, setCsvSaved] = useState(null); // result of the last write to disk
  const saveQueue = useRef(Promise.resolve());
  // Another tab saved the review: this one's copy is out of date, so it stops saving (to the browser and to disk)
  // rather than overwrite that work. A tab left open from an earlier session once rewrote annotations.csv this way
  const [stale, setStale] = useState(false);
  const staleRef = useRef(false);
  useEffect(() => {
    const onStorage = e => { if (e.key?.includes(STORAGE_KEY)) { staleRef.current = true; setStale(true); } };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  // Events from the pipeline (mock ones without it), with the saved review applied by id
  useEffect(() => {
    (async () => {
      const base = await loadEvents();
      const saved = await loadSaved();
      const loaded = base.map(e => ({ ...e, annotation: saved[e.id]?.annotation ?? null, gap_before: !!saved[e.id]?.gap_before,
        landing_fix: saved[e.id]?.landing_fix ?? null }));
      // Continue where you left off: the event last on screen, else the first one not reviewed yet
      let pos = null;
      try {
        const p = await window.storage.get(POSITION_KEY);
        if (p) pos = JSON.parse(p.value);
      } catch {}
      const f = pos?.filter ?? "all";
      const m = pos?.match && loaded.some(e => e.match === pos.match) ? pos.match : "all";
      const list = loaded.filter(e => inFilter(e, f) && inMatch(e, m));
      let i = pos ? list.findIndex(e => e.id === pos.id) : -1;
      if (i < 0) i = Math.max(0, list.findIndex(e => !e.annotation));
      setFilter(f);
      setMatch(m);
      setIdx(i);
      setEvents(loaded);
    })();
  }, []);

  // Save the review on change
  useEffect(() => {
    if (events.length === 0 || staleRef.current) return;
    const review = Object.fromEntries(events.filter(e => e.annotation || e.gap_before || e.landing_fix)
      .map(e => [e.id, { annotation: e.annotation, gap_before: e.gap_before || undefined, landing_fix: e.landing_fix || undefined }]));
    const json = JSON.stringify(review);
    (async () => { try { await window.storage.set(STORAGE_KEY, json); } catch {} })();
    // ...and to review.json on disk (dev server only), queued with the CSV writes. Never an empty review over it
    if (Object.keys(review).length) {
      saveQueue.current = saveQueue.current.then(() =>
        fetch("/api/review.json", { method: "PUT", headers: { "Content-Type": "application/json" }, body: json }).catch(() => {}));
    }
  }, [events]);

  const filtered = events.filter(e => inFilter(e, filter) && inMatch(e, match));
  const matchNames = useMemo(() => [...new Set(events.map(e => e.match))], [events.length]);
  // Picking a match goes to its first shot not reviewed yet
  const chooseMatch = m => {
    setMatch(m);
    const list = events.filter(e => inFilter(e, filter) && inMatch(e, m));
    setIdx(Math.max(0, list.findIndex(e => !e.annotation)));
  };
  const current = filtered[idx];
  // The court shows your landing when you've moved it (same object while the event is unchanged, so a drag isn't reset)
  const shownCv = useMemo(() => current && (current.landing_fix ? { ...current.cv, landing_xy: current.landing_fix } : current.cv), [current]);
  // The next shot in the same rally: its contact is where this shot went (ShuttleSet's landing for a return)
  const nextEvent = useMemo(() => current && events
    .filter(e => e.match === current.match && e.rally === current.rally && e.shot_num > current.shot_num)
    .sort((a, b) => a.shot_num - b.shot_num)[0], [current, events]);
  // A landing on the hitter's own half, never accepted ("returned" when a shot follows, "last" on the rally's
  // last shot). The landing checked is yours, else the one passed on from a next shot marked not a shot, else
  // the pipeline's
  const landingProblem = useMemo(() => {
    if (!current?.cv?.player_xy || current.annotation?.not_shot) return null;
    const passedOn = nextEvent?.annotation?.not_shot ? (nextEvent.landing_fix ?? nextEvent.cv.landing_xy) : null;
    if (!onOwnHalf(current.cv.player_xy, current.landing_fix ?? passedOn ?? current.cv.landing_xy)) return null;
    return nextEvent ? "returned" : "last";
  }, [current, nextEvent]);

  // Remember the event on screen and the filter, for the next load
  useEffect(() => {
    if (!current) return;
    (async () => { try { await window.storage.set(POSITION_KEY, JSON.stringify({ id: current.id, filter, match })); } catch {} })();
  }, [current?.id, filter, match]);
  // The 3D court previews the human label once set, otherwise the suggestion
  const shownShot = current && (current.annotation?.shot_type || suggestionOf(current)?.shot_type);

  const annotate = useCallback((shotType, confirmed = false) => {
    if (!current) return;
    setEvents(prev => prev.map(e => e.id === current.id ? {
      ...e, annotation: { shot_type: shotType, hitting_player: hitterOf(e), confirmed, corrected: !confirmed, timestamp: Date.now() }
    } : e));
    if (idx < filtered.length - 1) setIdx(i => i + 1);
  }, [current, idx, filtered.length]);

  // A false hit from the pipeline: counts as reviewed, and the export drops it
  const markNotShot = useCallback(() => {
    if (!current) return;
    setEvents(prev => prev.map(e => e.id === current.id ? { ...e, annotation: { not_shot: true, timestamp: Date.now() } } : e));
    if (idx < filtered.length - 1) setIdx(i => i + 1);
  }, [current, idx, filtered.length]);

  // The pipeline missed a shot between the previous event and this one: the export splits the rally here
  const toggleGap = useCallback(() => {
    if (!current) return;
    setEvents(prev => prev.map(e => e.id === current.id ? { ...e, gap_before: !e.gap_before } : e));
  }, [current]);

  // A landing you placed on the court (normalised), or null to go back to the pipeline's
  const setLanding = useCallback(xy => {
    if (!current) return;
    setEvents(prev => prev.map(e => e.id === current.id ? { ...e, landing_fix: xy } : e));
  }, [current]);

  const handleKey = useCallback((e) => {
    if (showExport) return;
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); const s = current && suggestionOf(current); if (s) annotate(s.shot_type, true); }
    else if (e.key === "Backspace") { e.preventDefault(); if (idx > 0) setIdx(i => i - 1); }
    else if (e.key === "ArrowRight") { e.preventDefault(); if (idx < filtered.length-1) setIdx(i => i+1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); if (idx > 0) setIdx(i => i-1); }
    else if (SHOT_KEYS[e.key] && current) annotate(SHOT_KEYS[e.key], false);
    else if (e.key === "u" && current) {
      setEvents(prev => prev.map(ev => ev.id === current.id ? {...ev, annotation: null} : ev));
    }
    else if (e.key === "x" && current) markNotShot();
    else if (e.key === "g" && current) toggleGap();
    else if (e.key === "l" && current) setEditLanding(v => !v);
  }, [current, idx, filtered.length, annotate, markNotShot, toggleGap, showExport]);

  useEffect(() => {
    const el = containerRef.current;
    if (el) { el.focus(); }
  }, [idx, filter]);

  // Agreement is your label against the suggestion the event has now, whichever key entered it: counting only
  // Enter-confirmed labels scored every label made before the suggestions existed as a correction
  const withSuggestion = events.filter(e => e.annotation && !e.annotation.not_shot && suggestionOf(e));
  const agreed = withSuggestion.filter(e => e.annotation.shot_type === suggestionOf(e).shot_type).length;
  const stats = {
    total: events.length,
    done: events.filter(e => e.annotation).length,
    confirmed: agreed,
    corrected: withSuggestion.length - agreed,
    notShots: events.filter(e => e.annotation?.not_shot).length,
  };
  const accuracy = withSuggestion.length > 0 ? ((agreed / withSuggestion.length) * 100).toFixed(1) : "—";

  const predictorCsv = useMemo(() => toPredictorCsv(events), [events]);

  // Write the CSV to disk through the dev server (see vite.config.js); chained so writes land in order
  useEffect(() => {
    if (events.length === 0 || staleRef.current) return;
    const { csv } = predictorCsv;
    saveQueue.current = saveQueue.current.then(async () => {
      try {
        const res = await fetch("/api/annotations.csv", { method: "PUT", headers: { "Content-Type": "text/csv" }, body: csv });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        setCsvSaved({ ok: true, path: (await res.json()).path });
      } catch {
        setCsvSaved({ ok: false });
      }
    });
  }, [predictorCsv, events.length]);

  const downloadCsv = () => {
    const url = URL.createObjectURL(new Blob([predictorCsv.csv], { type: "text/csv" }));
    Object.assign(document.createElement("a"), { href: url, download: "annotations.csv" }).click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  // Clears the whole review, in the browser and in review.json on disk (the dev server copies the old file to
  // review.<time>.json first); annotations.csv follows from the empty review. The one save of an empty review
  const resetAll = async () => {
    if (staleRef.current) return;
    if (!window.confirm("Clear every annotation? review.json is copied to review.<time>.json first.")) return;
    const base = await loadEvents();
    setEvents(base.map(e => ({ ...e, annotation: null, gap_before: false, landing_fix: null })));
    setFilter("all");
    setIdx(0);
    try { await window.storage.set(STORAGE_KEY, "{}"); } catch {}
    saveQueue.current = saveQueue.current.then(() =>
      fetch("/api/review.json", { method: "PUT", headers: { "Content-Type": "application/json" }, body: "{}" }).catch(() => {}));
  };

  if (events.length === 0) return <div style={{color:"#94a3b8",padding:40,textAlign:"center",fontFamily:"system-ui"}}>Loading...</div>;

  return (
    <div ref={containerRef} tabIndex={0} onKeyDown={handleKey}
      style={{ fontFamily:"'Inter',system-ui,sans-serif", background:"#0f1729", color:"#e2e8f0", minHeight:"100vh", outline:"none", padding:"16px 20px", maxWidth:1400, margin:"0 auto" }}>

      {/* Header */}
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:16, flexWrap:"wrap", gap:8 }}>
        <div>
          <h1 style={{ fontSize:20, fontWeight:700, margin:0, color:"#f1f5f9", letterSpacing:"-0.02em" }}>Shuttle Annotator</h1>
          <span style={{ fontSize:12, color:"#64748b" }}>BWF Match Shot Labeling Tool</span>
        </div>
        <div style={{ display:"flex", gap:8, alignItems:"center" }}>
          {csvSaved && (
            <span title={csvSaved.path} style={{ fontSize:11, color: csvSaved.ok ? "#34d399" : "#fbbf24" }}>
              {csvSaved.ok ? `✓ ${csvSaved.path.split("/").pop()} · ${predictorCsv.rallies} rallies` : "Not saved to disk"}
            </span>
          )}
          <button onClick={() => setShowExport(!showExport)} style={btnStyle("#1e293b")}>Export CSV</button>
          <button onClick={resetAll} style={btnStyle("#1e293b")}>Reset</button>
        </div>
      </div>

      {/* Progress bar */}
      <div style={{ display:"flex", gap:16, alignItems:"center", marginBottom:16, flexWrap:"wrap" }}>
        <div style={{ flex:1, minWidth:200, background:"#1e293b", borderRadius:6, height:8, overflow:"hidden" }}>
          <div style={{ width:`${(stats.done/stats.total)*100}%`, height:"100%", background:"linear-gradient(90deg,#3b82f6,#8b5cf6)", transition:"width 0.3s" }}/>
        </div>
        <div style={{ display:"flex", gap:16, fontSize:12, color:"#94a3b8", flexShrink:0 }}>
          <span><b style={{color:"#f1f5f9"}}>{stats.done}</b>/{stats.total}</span>
          <span style={{color:"#34d399"}} title="your label matches the suggestion">✓ {stats.confirmed}</span>
          <span style={{color:"#fb923c"}} title="your label differs from the suggestion">✎ {stats.corrected}</span>
          <span style={{color:"#f87171"}} title="not a shot">✗ {stats.notShots}</span>
          <span title="share of your labels that match the suggestion shown, however they were entered">Suggestion agreement: <b style={{color: Number(accuracy) > 80 ? "#34d399" : "#fbbf24"}}>{accuracy}%</b></span>
        </div>
      </div>

      {/* Filter tabs */}
      <div style={{ display:"flex", gap:4, marginBottom:16 }}>
        {[["all","All"],["pending","Pending"],["done","Done"]].map(([key,label]) => (
          <button key={key} onClick={() => { setFilter(key); setIdx(0); }}
            style={{ padding:"6px 14px", fontSize:12, fontWeight:500, borderRadius:6, border:"none", cursor:"pointer",
              background: filter===key ? "#3b82f6" : "#1e293b", color: filter===key ? "#fff" : "#94a3b8" }}>
            {label} ({key==="all" ? events.length : key==="pending" ? events.filter(e=>!e.annotation).length : stats.done})
          </button>
        ))}
        {/* One match at a time; a match with a player outside the predictor's 35 is reviewed but never exported */}
        <select value={match} onChange={e => chooseMatch(e.target.value)}
          style={{ marginLeft:8, padding:"6px 8px", fontSize:12, borderRadius:6, border:"1px solid #334155", background:"#1e293b", color:"#e2e8f0", minWidth:0, maxWidth:"100%" }}>
          <option value="all">All matches</option>
          {matchNames.map(name => {
            const evs = events.filter(e => e.match === name);
            const known = new Set(Object.keys(PLAYER_IDS).map(playerKey));
            const exported = evs[0].players.every(p => known.has(playerKey(p)));
            return (
              <option key={name} value={name}>
                {name} · {evs.filter(e => e.annotation).length}/{evs.length}{exported ? "" : " · not exported"}
              </option>
            );
          })}
        </select>
      </div>

      {/* Export modal */}
      {showExport && (
        <div style={{ background:"#1e293b", borderRadius:8, padding:16, marginBottom:16, border:"1px solid #334155" }}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:8, gap:8, flexWrap:"wrap" }}>
            <span style={{ fontSize:13, fontWeight:600 }}>BadmintonShotPredictor CSV (train.csv format)</span>
            <div style={{ display:"flex", gap:6 }}>
              <button onClick={() => navigator.clipboard.writeText(predictorCsv.csv)} style={btnStyle("#334155")}>Copy</button>
              <button onClick={downloadCsv} style={btnStyle("#334155")}>Download</button>
            </div>
          </div>
          <div style={{ fontSize:11, color:"#64748b", marginBottom:8, lineHeight:1.6 }}>
            <div>{predictorCsv.rallies} rallies · {predictorCsv.shots} shots, from {predictorCsv.reviewed} fully reviewed rallies. Rallies with unreviewed shots are held back.</div>
            <div>{predictorCsv.notShots} false hits dropped, and {predictorCsv.ownHalf} shots landing on their own half left out (fix them with L or X). Rallies are split where a shot is missing, and {predictorCsv.short} pieces under {MIN_PIECE_SHOTS} shots left out (the predictor never scores a rally's first 3 shots).</div>
            <div>{csvSaved?.ok ? <>Auto-saved to <code>{csvSaved.path}</code></> : "Auto-save needs the Vite dev server (npm run dev); use Download instead."}</div>
            {predictorCsv.unknownPlayers.length > 0 && (
              <div style={{ color:"#fbbf24" }}>
                Not among the predictor's 35 players, so their rallies are left out: {predictorCsv.unknownPlayers.join(", ")} ({predictorCsv.outsiders} reviewed rallies held back)
              </div>
            )}
          </div>
          <pre style={{ fontSize:11, color:"#94a3b8", maxHeight:200, overflow:"auto", margin:0, whiteSpace:"pre" }}>{predictorCsv.csv}</pre>
        </div>
      )}

      {!current ? (
        <div style={{ textAlign:"center", padding:60, color:"#64748b" }}>
          {filter === "pending" ? "All shots annotated!" : "No events loaded."}
        </div>
      ) : (
        <div style={{ display:"grid", gridTemplateColumns:"minmax(0, 520px) minmax(0, 1fr)", gap:16, alignItems:"start" }}>
          {/* Left: Court + CV data */}
          <div>
            <div style={{ marginBottom:8, display:"flex", justifyContent:"space-between", alignItems:"center" }}>
              <span style={{ fontSize:11, color:"#64748b" }}>{current.match}</span>
              <span style={{ fontSize:11, color:"#64748b" }}>R{current.rally} · S{current.shot_num} · {current.frame_time}</span>
            </div>
            <Court3D cv={shownCv} shotType={shownShot} hitter={hitterOf(current)} players={current.players} shotColor={SHOT_COLORS[shownShot] || "#fff"}
              editLanding={editLanding} onLanding={setLanding} />
            <div style={{ display:"flex", gap:6, marginTop:6 }}>
              <button onClick={() => setEditLanding(v => !v)}
                style={{ ...btnStyle(editLanding ? "#78350f" : "#1e293b"), color: editLanding ? "#fde68a" : "#e2e8f0" }}>
                {editLanding ? "✓ Done moving landing" : "📍 Move landing"} <span style={{ color:"#94a3b8", fontSize:10 }}>L</span>
              </button>
              {current.landing_fix && (
                <button onClick={() => setLanding(null)} style={btnStyle("#1e293b")}>↺ Reset landing</button>
              )}
            </div>
            <div style={{ marginTop:8, fontSize:11, color:"#64748b", display:"grid", gridTemplateColumns:"1fr 1fr", gap:"4px 12px" }}>
              <span>Landing: ({shownCv.landing_xy[0].toFixed(2)}, {shownCv.landing_xy[1].toFixed(2)}){current.landing_fix
                ? <b style={{ color:"#f59e0b" }}> · moved by you</b> : current.cv.landing_source ? ` · ${current.cv.landing_source.replace("_", " ")}` : ""}</span>
              <span>Speed: {current.cv.speed ?? "—"} km/h{current.cv.landing_source ? " (average)" : ""}</span>
              <span>Player: ({current.cv.player_xy[0].toFixed(2)}, {current.cv.player_xy[1].toFixed(2)}){current.cv.hitter_side ? ` · ${current.cv.hitter_side}` : ""}</span>
              <span>Angle: {current.cv.trajectory_angle ?? "—"}°</span>
            </div>
          </div>

          {/* Right: this shot's contact, the next one's (where it went), then labels + controls */}
          <div>
            {stale && (
              <div style={{ background:"#3b1414", borderRadius:8, padding:"8px 10px", marginBottom:8, border:"1px solid #b91c1c", fontSize:12, color:"#fca5a5",
                display:"flex", justifyContent:"space-between", alignItems:"center", gap:8 }}>
                <span>The review was changed in another tab, so this tab has stopped saving. Reload to continue here.</span>
                <button onClick={() => window.location.reload()} style={{ ...btnStyle("#7f1d1d"), color:"#fff", flexShrink:0 }}>Reload</button>
              </div>
            )}
            {/* The match's first-named player's row always on top, the other player's below: the shot being labelled
                sits on its hitter's row, the next shot (or, after a rally's last shot, the court a moment later) on the other */}
            {(() => {
            const thisShot = (
              <StripPanel key="this" event={current} color="#60a5fa"
                title={`This shot · S${current.shot_num} · ${current.players[hitterOf(current) - 1]}`} />
            );
            const other = nextEvent ? (
              <StripPanel key="next" event={nextEvent} color="#a78bfa"
                title={`Next shot · S${nextEvent.shot_num} · ${nextEvent.players[hitterOf(nextEvent) - 1]}: where this shot went${nextEvent.annotation?.not_shot ? " (marked not a shot)" : ""}`} />
            ) : (
              <div key="after">
                {/* No next hit shows where the last shot went, so 1F shows the court 0.5, 1 and 1.5 s after it */}
                {current.frames_after && (
                  <div style={{ marginBottom:4 }}>
                    <div style={{ fontSize:12, fontWeight:600, color:"#a78bfa", marginBottom:3 }}>After the last shot: where it went</div>
                    <img src={current.frames_after} alt="the court 0.5, 1 and 1.5 s after the last hit"
                      style={{ width:"100%", display:"block", borderRadius:6 }}/>
                  </div>
                )}
                <div style={{ fontSize:12, color:"#64748b", padding:"4px 2px 10px" }}>
                  Last detected shot of the rally: its landing is {current.cv.landing_source === "floor" ? "where the shuttle hit the floor"
                    : current.cv.landing_source === "track_end" ? "where the shuttle's track ends (rough)" : "from the pipeline"}.
                </div>
              </div>
            );
            // The suggestion sits between the two strips, so the frames and the label read in one glance:
            // Claude's label (Phase 1G) if there is one, else the ShuttleSet classifier's
            const suggestion = current.claude_label ? (
              <div key="suggestion" style={{ background:"#1e293b", borderRadius:8, padding:"8px 12px", marginBottom:8, border:"1px solid #334155", textAlign:"center" }}>
                <div style={{ display:"flex", justifyContent:"center", alignItems:"center", gap:8, marginBottom:4 }}>
                  <span style={{ fontSize:11, color:"#64748b", fontWeight:600 }}>CLAUDE LABEL</span>
                  <span style={{ fontSize:10, padding:"2px 8px", borderRadius:10,
                    background: current.claude_label.confidence === "high" ? "#166534" : "#854d0e",
                    color: current.claude_label.confidence === "high" ? "#4ade80" : "#fbbf24"
                  }}>{current.claude_label.confidence}</span>
                </div>
                <div style={{ fontSize:18, fontWeight:700, color: SHOT_COLORS[current.claude_label.shot_type], marginBottom:4 }}>
                  {current.claude_label.shot_type}
                </div>
                <div style={{ fontSize:11, color:"#94a3b8", lineHeight:1.5 }}>{current.claude_label.reasoning}</div>
              </div>
            ) : current.model_label ? (
              <div key="suggestion" style={{ background:"#1e293b", borderRadius:8, padding:"8px 12px", marginBottom:8, border:"1px solid #334155", textAlign:"center" }}>
                <div style={{ display:"flex", justifyContent:"center", alignItems:"center", gap:8, marginBottom:4 }}>
                  <span style={{ fontSize:11, color:"#64748b", fontWeight:600 }}>CLASSIFIER SUGGESTION</span>
                  <span style={{ fontSize:10, padding:"2px 8px", borderRadius:10,
                    background: current.model_label.p >= CONFIDENT_P ? "#166534" : "#854d0e",
                    color: current.model_label.p >= CONFIDENT_P ? "#4ade80" : "#fbbf24"
                  }}>{Math.round(current.model_label.p * 100)}%</span>
                </div>
                <div style={{ fontSize:18, fontWeight:700, color: SHOT_COLORS[current.model_label.shot_type], marginBottom:4 }}>
                  {current.model_label.shot_type}
                </div>
                <div style={{ fontSize:11, color:"#94a3b8", lineHeight:1.5 }}>
                  Else {current.model_label.top.slice(1).map(([t, p]) => `${t} ${Math.round(p * 100)}%`).join(" · ")}.
                  Hit by <b style={{ color:"#e2e8f0" }}>{current.players[hitterOf(current) - 1]}</b>.
                </div>
              </div>
            ) : (
              <div key="suggestion" style={{ background:"#1e293b", borderRadius:8, padding:"8px 12px", marginBottom:8, border:"1px dashed #334155", fontSize:12, color:"#94a3b8", textAlign:"center" }}>
                No suggestion yet (run pipeline/shot_classify.py label). Hit by <b style={{ color:"#e2e8f0" }}>{current.players[hitterOf(current) - 1]}</b>; pick the shot type with 1–0.
              </div>
            );
            return hitterOf(current) === 1 ? [thisShot, suggestion, other] : [other, suggestion, thisShot];
            })()}

            {current.gap_before && (
              <div style={{ background:"#1e1b3a", borderRadius:8, padding:"6px 10px", marginBottom:8, border:"1px solid #7c3aed", fontSize:12, color:"#c4b5fd" }}>
                ⋯ A shot is missing before this one: the export splits the rally here (G to undo)
              </div>
            )}

            {landingProblem && (
              <div style={{ background:"#3b1414", borderRadius:8, padding:"6px 10px", marginBottom:8, border:"1px solid #b91c1c", fontSize:12, color:"#fca5a5" }}>
                {landingProblem === "returned"
                  ? "⚠ Lands on the hitter's own half, which a returned shot can't: this shot or the next is a false hit (X on it), or the landing is wrong (L to move it)."
                  : "⚠ The rally's last shot lands on the hitter's own half: move the landing to where it came down (L), or mark a false hit (X)."}
                {" "}The export leaves this shot out until it's fixed.
              </div>
            )}

            {/* Current annotation status */}
            {current.annotation?.not_shot ? (
              <div style={{ background:"#3b1414", borderRadius:8, padding:"6px 10px", marginBottom:8, border:"1px solid #7f1d1d", fontSize:12 }}>
                ✗ Not a shot: dropped from the export (U to undo)
              </div>
            ) : current.annotation && (
              <div style={{ background: current.annotation.confirmed ? "#14352a" : "#352a14", borderRadius:8, padding:"6px 10px", marginBottom:8,
                border: `1px solid ${current.annotation.confirmed ? "#166534" : "#854d0e"}` }}>
                <span style={{ fontSize:12 }}>
                  {current.annotation.confirmed ? "✓ Confirmed" : `✎ Corrected → `}
                  <b style={{ color: SHOT_COLORS[current.annotation.shot_type] }}>{current.annotation.shot_type}</b>
                </span>
              </div>
            )}

            {/* Shot type buttons: two rows of five, in key order */}
            <div style={{ display:"grid", gridTemplateColumns:"repeat(5, minmax(0, 1fr))", gap:4, marginBottom:6 }}>
              {SHOT_TYPES.map((st, i) => (
                <button key={st} onClick={() => annotate(st, st === suggestionOf(current)?.shot_type)}
                  style={{ padding:"6px 6px", fontSize:11, fontWeight:500, borderRadius:6, border:"1px solid #334155", cursor:"pointer",
                    background: current.annotation?.shot_type === st ? "#334155" : "#1e293b",
                    color: SHOT_COLORS[st], textAlign:"left", display:"flex", justifyContent:"space-between", gap:4 }}>
                  <span style={{ whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>{st}</span>
                  <span style={{color:"#475569", fontSize:10}}>{(i+1) % 10}</span>
                </button>
              ))}
            </div>

            {/* Pipeline mistakes: a false hit, or a shot it missed */}
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:4, marginBottom:8 }}>
              {[["✗ Not a shot", "x", markNotShot, current.annotation?.not_shot, "#f87171"],
                ["⋯ Shot missing before", "g", toggleGap, current.gap_before, "#c4b5fd"]].map(([text, key, onClick, active, color]) => (
                <button key={key} onClick={onClick}
                  style={{ padding:"7px 8px", fontSize:11, fontWeight:500, borderRadius:6, border:"1px solid #334155", cursor:"pointer",
                    background: active ? "#334155" : "#1e293b", color, textAlign:"left", display:"flex", justifyContent:"space-between" }}>
                  <span>{text}</span>
                  <span style={{color:"#475569", fontSize:10}}>{key.toUpperCase()}</span>
                </button>
              ))}
            </div>

            {/* Nav and confirm on one row */}
            <div style={{ display:"flex", gap:6, alignItems:"center" }}>
              <button onClick={() => idx > 0 && setIdx(i=>i-1)} disabled={idx===0}
                style={{...btnStyle("#1e293b"), opacity: idx===0?0.3:1}}>← Back</button>
              <button onClick={() => suggestionOf(current) && annotate(suggestionOf(current).shot_type, true)} disabled={!suggestionOf(current)}
                style={{ flex:1, padding:"8px", fontSize:13, fontWeight:600, borderRadius:8, border:"none", cursor: suggestionOf(current) ? "pointer" : "default",
                  background:"#3b82f6", color:"#fff", opacity: suggestionOf(current) ? 1 : 0.35 }}>
                {suggestionOf(current) ? `⏎ Confirm "${suggestionOf(current).shot_type}"` : "Nothing to confirm yet"}
              </button>
              <button onClick={() => idx < filtered.length-1 && setIdx(i=>i+1)} disabled={idx>=filtered.length-1}
                style={{...btnStyle("#1e293b"), opacity: idx>=filtered.length-1?0.3:1}}>Next →</button>
              <span style={{ fontSize:12, color:"#64748b", minWidth:70, textAlign:"right" }}>{idx+1} / {filtered.length}</span>
            </div>
          </div>
        </div>
      )}

      {/* Keyboard shortcuts */}
      <div style={{ marginTop:20, padding:"10px 14px", background:"#1e293b", borderRadius:8, display:"flex", flexWrap:"wrap", gap:"8px 20px", fontSize:11, color:"#64748b" }}>
        <span><kbd style={kbdStyle}>Enter</kbd> Confirm</span>
        <span><kbd style={kbdStyle}>1-0</kbd> Override shot type</span>
        <span><kbd style={kbdStyle}>←→</kbd> Navigate</span>
        <span><kbd style={kbdStyle}>⌫</kbd> Go back</span>
        <span><kbd style={kbdStyle}>U</kbd> Undo annotation</span>
        <span><kbd style={kbdStyle}>X</kbd> Not a shot</span>
        <span><kbd style={kbdStyle}>G</kbd> Shot missing before this</span>
        <span><kbd style={kbdStyle}>L</kbd> Move landing</span>
      </div>
    </div>
  );
}

const btnStyle = (bg) => ({ padding:"6px 12px", fontSize:12, fontWeight:500, borderRadius:6, border:"1px solid #334155", cursor:"pointer", background:bg, color:"#e2e8f0" });
const kbdStyle = { background:"#334155", padding:"2px 6px", borderRadius:4, fontSize:10, fontFamily:"monospace", color:"#e2e8f0" };
