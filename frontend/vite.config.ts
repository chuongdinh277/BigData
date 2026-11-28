import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Khi phát triển: /api được chuyển tiếp tới FastAPI ở cổng 8000.
// Khi triển khai: nginx trong container web làm việc này (deploy/nginx.conf).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: process.env.VITE_API_TARGET ?? "http://localhost:8000", changeOrigin: true } },
  },
});
