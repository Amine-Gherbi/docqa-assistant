import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

// The API is served on 8788 because Windows excludes 8001-8300 from user
// listeners on this machine. Vite proxies /api so the browser sees one origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8788',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  preview: {
    port: 5174,
    strictPort: true,
  },
  build: {
    target: 'es2022',
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    css: false,
  },
});
