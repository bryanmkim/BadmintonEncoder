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
const STORAGE_KEY = "annotator:events:v2"; // v2: labels switched to the predictor's classes

// === MOCK DATA (replace with real CV pipeline output) ===
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

const Court3D = ({ cv, shotType, hitter, shotColor }) => {
  const W = 420, H = 440;
  const [view, setView] = useState(VIEWS.Broadcast);
  const [t, setT] = useState(0);
  const drag = useRef(null);
  const traj = useMemo(() => buildTrajectory(cv, shotType), [cv, shotType]);
  const { project: P, size } = useMemo(() => makeProjector(view, W, H), [view]);

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
        <text x={lx} y={ly} textAnchor="middle" fill="#fff" fontSize={10} fontWeight="bold">{label}</text>
      </g>
    ));
  };
  player(cv.player_xy, "#3b82f6", `P${hitter}`, "hitter");
  player(cv.opponent_xy, "#ef4444", `P${hitter === 1 ? 2 : 1}`, "opponent");

  // Trajectory: shadow + landing on the floor, arc split at the net for correct layering
  const [lX, lY] = toWorld(cv.landing_xy);
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

  const onPointerDown = e => { e.currentTarget.setPointerCapture(e.pointerId); drag.current = { x: e.clientX, y: e.clientY }; };
  const onPointerMove = e => {
    if (!drag.current) return;
    const dx = e.clientX - drag.current.x, dy = e.clientY - drag.current.y;
    drag.current = { x: e.clientX, y: e.clientY };
    setView(v => ({ yaw: v.yaw - dx * 0.01, pitch: Math.min(1.54, Math.max(0.08, v.pitch + dy * 0.008)) }));
  };
  const endDrag = () => { drag.current = null; };

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={endDrag} onPointerCancel={endDrag}
        onDoubleClick={() => setView(VIEWS.Broadcast)}
        style={{ width: "100%", background: "#1a2332", borderRadius: 8, cursor: drag.current ? "grabbing" : "grab", touchAction: "none", userSelect: "none", display: "block" }}>
        <polygon points={poly(apron)} fill="#22472a"/>
        <polygon points={poly(surface)} fill="#2d5a27"/>
        {COURT_LINES.map(([a, b], i) => <path key={i} d={path([a, b])} stroke="#fff" strokeWidth={1.2} opacity={0.8}/>)}
        {floor}
        {byDepth(layers.far)}
        {net}
        {byDepth(layers.near)}
        <text x={10} y={14} fill="#475569" fontSize={9}>drag to orbit · double-click to reset</text>
        <text x={W / 2} y={H - 8} textAnchor="middle" fill="#94a3b8" fontSize={10}>
          {cv.speed} km/h · {cv.trajectory_angle}° · contact {traj.contact.toFixed(1)} m · apex {traj.apex[2].toFixed(1)} m
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

// Predictor player ids (0–34). Its CSVs are anonymised, so add names here once you know their ids;
// unlisted players get new ids from 35 up (the predictor's player_embedding must grow to train on them).
const PLAYER_IDS = {};
const FIRST_NEW_PLAYER_ID = 35;

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

const toPredictorCsv = events => {
  const playerIds = { ...PLAYER_IDS };
  let nextId = Math.max(FIRST_NEW_PLAYER_ID, ...Object.values(PLAYER_IDS).map(id => id + 1));
  const idFor = name => (playerIds[name] ??= nextId++);
  events.forEach(e => e.players.forEach(idFor)); // assign over all events so ids stay stable as rallies complete

  const rallies = new Map();
  events.forEach(e => {
    const key = `${e.match}|${e.rally}`;
    if (!rallies.has(key)) rallies.set(key, []);
    rallies.get(key).push(e);
  });

  const rows = [];
  let complete = 0;
  [...rallies.values()].forEach((shots, i) => {
    // Whole rallies only: the model learns shot-to-shot transitions, so a gap would teach a false one
    if (!shots.every(s => s.annotation)) return;
    complete++;
    [...shots].sort((a, b) => a.shot_num - b.shot_num).forEach(s => {
      const [x, y] = toPredictorXY(s.cv.landing_xy);
      const hitter = s.players[s.annotation.hitting_player - 1];
      rows.push([RALLY_ID_OFFSET + i, s.shot_num, idFor(hitter), s.annotation.shot_type, x, y, shots.length]);
    });
  });

  return {
    csv: [PREDICTOR_COLUMNS, ...rows].map(r => r.join(",")).join("\n") + "\n",
    rallies: complete,
    shots: rows.length,
    newPlayers: Object.entries(playerIds).filter(([name]) => !(name in PLAYER_IDS)),
  };
};

// === FRAME STRIP (mock - shows placeholder frames) ===
const FrameStrip = ({ event }) => (
  <div style={{ display:"flex", gap:4, marginBottom:12 }}>
    {[-2,-1,0,1,2].map(offset => (
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

// === MAIN APP ===
export default function BadmintonAnnotator() {
  const [events, setEvents] = useState([]);
  const [idx, setIdx] = useState(0);
  const [filter, setFilter] = useState("all"); // all | pending | done
  const [showExport, setShowExport] = useState(false);
  const containerRef = useRef(null);
  const [csvSaved, setCsvSaved] = useState(null); // result of the last write to disk
  const saveQueue = useRef(Promise.resolve());

  // Load from storage or generate mock
  useEffect(() => {
    (async () => {
      try {
        const saved = await window.storage.get(STORAGE_KEY);
        if (saved) { setEvents(JSON.parse(saved.value)); return; }
      } catch {}
      const mock = generateMockEvents();
      setEvents(mock);
      try { await window.storage.set(STORAGE_KEY, JSON.stringify(mock)); } catch {}
    })();
  }, []);

  // Save on change
  useEffect(() => {
    if (events.length === 0) return;
    (async () => { try { await window.storage.set(STORAGE_KEY, JSON.stringify(events)); } catch {} })();
  }, [events]);

  const filtered = events.filter(e =>
    filter === "all" ? true : filter === "pending" ? !e.annotation : !!e.annotation
  );
  const current = filtered[idx];
  // The 3D court previews the human label once set, otherwise Claude's
  const shownShot = current && (current.annotation?.shot_type || current.claude_label.shot_type);

  const annotate = useCallback((shotType, confirmed = false) => {
    if (!current) return;
    setEvents(prev => prev.map(e => e.id === current.id ? {
      ...e, annotation: { shot_type: shotType, hitting_player: e.claude_label.hitting_player, confirmed, corrected: !confirmed, timestamp: Date.now() }
    } : e));
    if (idx < filtered.length - 1) setIdx(i => i + 1);
  }, [current, idx, filtered.length]);

  const handleKey = useCallback((e) => {
    if (showExport) return;
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); if (current) annotate(current.claude_label.shot_type, true); }
    else if (e.key === "Backspace") { e.preventDefault(); if (idx > 0) setIdx(i => i - 1); }
    else if (e.key === "ArrowRight") { e.preventDefault(); if (idx < filtered.length-1) setIdx(i => i+1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); if (idx > 0) setIdx(i => i-1); }
    else if (SHOT_KEYS[e.key] && current) annotate(SHOT_KEYS[e.key], false);
    else if (e.key === "u" && current) {
      setEvents(prev => prev.map(ev => ev.id === current.id ? {...ev, annotation: null} : ev));
    }
  }, [current, idx, filtered.length, annotate, showExport]);

  useEffect(() => {
    const el = containerRef.current;
    if (el) { el.focus(); }
  }, [idx, filter]);

  const stats = {
    total: events.length,
    done: events.filter(e => e.annotation).length,
    confirmed: events.filter(e => e.annotation?.confirmed).length,
    corrected: events.filter(e => e.annotation?.corrected).length,
  };
  const accuracy = stats.done > 0 ? ((stats.confirmed / stats.done) * 100).toFixed(1) : "—";

  const predictorCsv = useMemo(() => toPredictorCsv(events), [events]);

  // Write the CSV to disk through the dev server (see vite.config.js); chained so writes land in order
  useEffect(() => {
    if (events.length === 0) return;
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

  const resetAll = async () => {
    const mock = generateMockEvents();
    setEvents(mock);
    setIdx(0);
    try { await window.storage.set(STORAGE_KEY, JSON.stringify(mock)); } catch {}
  };

  if (events.length === 0) return <div style={{color:"#94a3b8",padding:40,textAlign:"center",fontFamily:"system-ui"}}>Loading...</div>;

  return (
    <div ref={containerRef} tabIndex={0} onKeyDown={handleKey}
      style={{ fontFamily:"'Inter',system-ui,sans-serif", background:"#0f1729", color:"#e2e8f0", minHeight:"100vh", outline:"none", padding:"16px 20px", maxWidth:900, margin:"0 auto" }}>

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
          <span style={{color:"#34d399"}}>✓ {stats.confirmed}</span>
          <span style={{color:"#fb923c"}}>✎ {stats.corrected}</span>
          <span>Claude acc: <b style={{color: Number(accuracy) > 80 ? "#34d399" : "#fbbf24"}}>{accuracy}%</b></span>
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
            <div>{predictorCsv.rallies} complete rallies · {predictorCsv.shots} shots. Rallies with unlabeled shots are held back so the model never sees a gap.</div>
            <div>{csvSaved?.ok ? <>Auto-saved to <code>{csvSaved.path}</code></> : "Auto-save needs the Vite dev server (npm run dev); use Download instead."}</div>
            {predictorCsv.newPlayers.length > 0 && (
              <div style={{ color:"#fbbf24" }}>
                New player ids (not in the predictor's 0–34): {predictorCsv.newPlayers.map(([name, id]) => `${name} → ${id}`).join(", ")}
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
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16 }}>
          {/* Left: Court + CV data */}
          <div>
            <div style={{ marginBottom:8, display:"flex", justifyContent:"space-between", alignItems:"center" }}>
              <span style={{ fontSize:11, color:"#64748b" }}>{current.match}</span>
              <span style={{ fontSize:11, color:"#64748b" }}>R{current.rally} · S{current.shot_num} · {current.frame_time}</span>
            </div>
            <Court3D cv={current.cv} shotType={shownShot} hitter={current.claude_label.hitting_player} shotColor={SHOT_COLORS[shownShot] || "#fff"} />
            <div style={{ marginTop:8, fontSize:11, color:"#64748b", display:"grid", gridTemplateColumns:"1fr 1fr", gap:"4px 12px" }}>
              <span>Landing: ({current.cv.landing_xy[0].toFixed(2)}, {current.cv.landing_xy[1].toFixed(2)})</span>
              <span>Speed: {current.cv.speed} km/h</span>
              <span>Player: ({current.cv.player_xy[0].toFixed(2)}, {current.cv.player_xy[1].toFixed(2)})</span>
              <span>Angle: {current.cv.trajectory_angle}°</span>
            </div>
          </div>

          {/* Right: Labels + controls */}
          <div>
            {/* Frame strip */}
            <FrameStrip event={current} />

            {/* Claude's label */}
            <div style={{ background:"#1e293b", borderRadius:8, padding:12, marginBottom:12, border:"1px solid #334155" }}>
              <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:6 }}>
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

            {/* Current annotation status */}
            {current.annotation && (
              <div style={{ background: current.annotation.confirmed ? "#14352a" : "#352a14", borderRadius:8, padding:10, marginBottom:12,
                border: `1px solid ${current.annotation.confirmed ? "#166534" : "#854d0e"}` }}>
                <span style={{ fontSize:12 }}>
                  {current.annotation.confirmed ? "✓ Confirmed" : `✎ Corrected → `}
                  <b style={{ color: SHOT_COLORS[current.annotation.shot_type] }}>{current.annotation.shot_type}</b>
                </span>
              </div>
            )}

            {/* Shot type buttons */}
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:4, marginBottom:12 }}>
              {SHOT_TYPES.map((st, i) => (
                <button key={st} onClick={() => annotate(st, st === current.claude_label.shot_type)}
                  style={{ padding:"7px 8px", fontSize:11, fontWeight:500, borderRadius:6, border:"1px solid #334155", cursor:"pointer",
                    background: current.annotation?.shot_type === st ? "#334155" : "#1e293b",
                    color: SHOT_COLORS[st], textAlign:"left", display:"flex", justifyContent:"space-between" }}>
                  <span>{st}</span>
                  <span style={{color:"#475569", fontSize:10}}>{(i+1) % 10}</span>
                </button>
              ))}
            </div>

            {/* Confirm button */}
            <button onClick={() => annotate(current.claude_label.shot_type, true)}
              style={{ width:"100%", padding:"10px", fontSize:14, fontWeight:600, borderRadius:8, border:"none", cursor:"pointer",
                background:"#3b82f6", color:"#fff", marginBottom:8 }}>
              ⏎ Confirm "{current.claude_label.shot_type}"
            </button>

            {/* Nav */}
            <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
              <button onClick={() => idx > 0 && setIdx(i=>i-1)} disabled={idx===0}
                style={{...btnStyle("#1e293b"), opacity: idx===0?0.3:1}}>← Back</button>
              <span style={{ fontSize:12, color:"#64748b" }}>{idx+1} / {filtered.length}</span>
              <button onClick={() => idx < filtered.length-1 && setIdx(i=>i+1)} disabled={idx>=filtered.length-1}
                style={{...btnStyle("#1e293b"), opacity: idx>=filtered.length-1?0.3:1}}>Next →</button>
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
      </div>
    </div>
  );
}

const btnStyle = (bg) => ({ padding:"6px 12px", fontSize:12, fontWeight:500, borderRadius:6, border:"1px solid #334155", cursor:"pointer", background:bg, color:"#e2e8f0" });
const kbdStyle = { background:"#334155", padding:"2px 6px", borderRadius:4, fontSize:10, fontFamily:"monospace", color:"#e2e8f0" };
