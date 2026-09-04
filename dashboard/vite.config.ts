import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Served by the FastAPI app at /dashboard, so assets resolve under that base. In development,
// `vite` proxies the API to a locally running serving app (uvicorn on 8000; never 5000 on macOS).
export default defineConfig({
  plugins: [react()],
  base: '/dashboard/',
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/model': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
  build: { outDir: 'dist', emptyOutDir: true },
})
