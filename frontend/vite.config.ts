import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Forward the UI API surface to the FastAPI backend.
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        // Pass the browser's address on: the API rate-limits logins per client IP.
        xfwd: true,
        timeout: 600_000,
        proxyTimeout: 600_000,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
});
