import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

declare const process: { env: Record<string, string | undefined> }

// Two homes for the same build:
//  - inside the serving container at /dashboard (default base), same origin as the API;
//  - on Cloudflare Pages at the site root, with VITE_BASE=/ and VITE_API_BASE_URL set to the
//    Container App's URL (the pattern the sibling projects use).
// In development, `vite` proxies the API to a local serving app on 8000 (never 5000 on macOS);
// DRIFTWATCH_DEV_API can point the proxy at a deployed API instead, for design work that only
// needs real numbers (the proxy is same-origin to the browser, so CORS never enters into it).
const devApi = process.env.DRIFTWATCH_DEV_API || 'http://localhost:8000'
const proxy = { target: devApi, changeOrigin: true }

export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_BASE || '/dashboard/',
  server: {
    proxy: { '/api': proxy, '/model': proxy, '/health': proxy },
  },
  build: { outDir: 'dist', emptyOutDir: true },
})
