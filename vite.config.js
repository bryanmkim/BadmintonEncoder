import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { copyFile, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

// Where annotations are written, in the same 7-column CSV format as BadmintonShotPredictor/train.csv.
// Override with ANNOTATIONS_CSV=/path/to/file.csv npm run dev
const OUTPUT_CSV = process.env.ANNOTATIONS_CSV
  ? resolve(process.env.ANNOTATIONS_CSV)
  : fileURLToPath(new URL("./annotations.csv", import.meta.url));
// The whole review ({ event id: { annotation, gap_before, landing_fix } }), so it doesn't live only in the
// browser's localStorage. Override with REVIEW_JSON=/path/to/file.json
const OUTPUT_REVIEW = process.env.REVIEW_JSON
  ? resolve(process.env.REVIEW_JSON)
  : fileURLToPath(new URL("./review.json", import.meta.url));
// Before the review on disk shrinks by more than this many events (a reset, or a browser that lost its storage),
// the old file is copied to review.<time>.json
const REVIEW_SHRINK = 5;

const readBody = req => new Promise((ok, fail) => {
  let body = "";
  req.setEncoding("utf8");
  req.on("data", chunk => { body += chunk; });
  req.on("end", () => ok(body));
  req.on("error", fail);
});

const backupIfShrinking = async body => {
  let old;
  try { old = JSON.parse(await readFile(OUTPUT_REVIEW, "utf8")); } catch { return; }
  if (Object.keys(JSON.parse(body)).length < Object.keys(old).length - REVIEW_SHRINK) {
    await copyFile(OUTPUT_REVIEW, OUTPUT_REVIEW.replace(/\.json$/, `.${new Date().toISOString().replace(/[:.]/g, "-")}.json`));
  }
};

// PUT /api/annotations.csv or /api/review.json with the body to overwrite that file on disk
const saveFiles = () => {
  const routes = { "/api/annotations.csv": [OUTPUT_CSV, null], "/api/review.json": [OUTPUT_REVIEW, backupIfShrinking] };
  const use = server => Object.entries(routes).forEach(([route, [path, before]]) => server.middlewares.use(route, async (req, res) => {
    if (req.method !== "PUT") { res.statusCode = 405; res.end(); return; }
    try {
      const body = await readBody(req);
      if (before) await before(body);
      await writeFile(path, body);
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ path }));
    } catch (err) {
      res.statusCode = 500;
      res.end(String(err));
    }
  }));
  return { name: "save-annotations", configureServer: use, configurePreviewServer: use };
};

export default defineConfig({
  plugins: [react(), saveFiles()],
  // Keep the dev server's file watcher off annotation output and the Phase 1 video pipeline
  server: { watch: { ignored: ["**/*.csv", "**/review*.json", "**/pipeline/**", "**/data/**"] } },
});
