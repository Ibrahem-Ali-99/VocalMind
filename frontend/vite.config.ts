import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';
import { defineConfig } from 'vite';
import istanbul from 'vite-plugin-istanbul';

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    // Instruments client code for Cypress test coverage
    istanbul({
      cypress: true,
      requireEnv: true,
      forceBuildInstrument: true,
    }),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  assetsInclude: ['**/*.svg', '**/*.csv'],
  server: {
    port: 3000,
    host: '0.0.0.0',
    open: false,
    watch: {
      usePolling: true,
    },
  },
});
