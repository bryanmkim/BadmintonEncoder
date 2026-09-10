import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

// Where annotations are written, in the same 7-column CSV format as BadmintonShotPredictor/train.csv.
// Override with ANNOTATIONS_CSV=/path/to/file.csv npm run dev
const OUTPUT_CSV = process.env.ANNOTATIONS_CSV
  ? resolve(process.env.ANNOTATIONS_CSV)
  : fileURLToPath(new URL("./annotations.csv", import.meta.url));

// PUT /api/annotations.csv with the CSV body to overwrite the file on disk
const saveAnnotationsCsv = () => {
  const handler = (req, res) => {
    if (req.method !== "PUT") { res.statusCode = 405; res.end(); return; }
    let body = "";
    req.setEncoding("utf8");
    req.on("data", chunk => { body += chunk; });
    req.on("end", async () => {
      try {
        await writeFile(OUTPUT_CSV, body);
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({ path: OUTPUT_CSV }));
      } catch (err) {
        res.statusCode = 500;
        res.end(String(err));
      }
    });
  };
  return {
    name: "save-annotations-csv",
    configureServer: server => { server.middlewares.use("/api/annotations.csv", handler); },
    configurePreviewServer: server => { server.middlewares.use("/api/annotations.csv", handler); },
  };
};

export default defineConfig({
  plugins: [react(), saveAnnotationsCsv()],
  // Keep the dev server's file watcher off annotation output and the Phase 1 video pipeline
  server: { watch: { ignored: ["**/*.csv", "**/pipeline/**", "**/data/**"] } },
});
