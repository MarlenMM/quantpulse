import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to the FastAPI backend, so the browser only ever
// talks to one origin in development too. That keeps the CORS allow-list in
// `api/main.py` as a safety net rather than something the app depends on.
export default defineConfig({
  plugins: [react()],
  // A GitHub Pages project site is served from `/<repo>/`, not from the root,
  // so every asset URL and the router's path matching need that prefix. It is
  // an environment variable rather than a constant so the same config produces
  // the root-mounted build that `npm run dev`, `vite preview` and the
  // Playwright suite all expect.
  base: process.env.VITE_BASE ?? "/",
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
  build: { outDir: "dist", sourcemap: true },
});
