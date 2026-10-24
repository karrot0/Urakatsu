import path from "node:path";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the UI runs on :5173 and proxies both the gRPC-web route and
// the JSON API to the Python service (URAKATSU_API, default :8090). In production
// the service serves web/dist itself, so no proxy is needed.
const api = process.env.URAKATSU_API ?? "http://127.0.0.1:8090";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  server: {
    proxy: {
      "/urakatsu.v1.Urakatsu": { target: api, changeOrigin: true },
      "/api": { target: api, changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
  },
});
