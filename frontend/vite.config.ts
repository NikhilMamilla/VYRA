import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Bind on all interfaces so the dev server is reachable from a container.
    host: true,
  },
  preview: {
    port: 4173,
  },
});
