import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],

  // Local release uses the root path. Hosted deployments can supply a custom
  // path, for example ELASTIQ_BASE_PATH=/repository/demo/ npm run build.
  base: process.env.ELASTIQ_BASE_PATH || "/",

  // Local development settings
  server: {
    port: 5173,
    strictPort: true,
    open: false,
  },
});
