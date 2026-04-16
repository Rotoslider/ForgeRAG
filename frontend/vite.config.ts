import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Serve production build under /app so it mounts cleanly from FastAPI at
// http://localhost:8200/app/ — see backend/main.py StaticFiles mount.
export default defineConfig({
  plugins: [react()],
  base: "/app/",
  server: {
    port: 5173,
    proxy: {
      // During dev, proxy API calls to the backend on :8200
      "/health": "http://localhost:8200",
      "/documents": "http://localhost:8200",
      "/categories": "http://localhost:8200",
      "/tags": "http://localhost:8200",
      "/ingest": "http://localhost:8200",
      "/search": "http://localhost:8200",
      "/graph": "http://localhost:8200",
      "/system": "http://localhost:8200",
      "/images": "http://localhost:8200",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
