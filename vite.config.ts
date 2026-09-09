import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';

// MediaPipe's SIMD + non-SIMD WASM runtimes for the vision tasks bundle.
// We only copy the plain (non-ES-module) variants, since FilesetResolver is
// called with useModule left at its default (false) — see src/tracking.ts
// once Phase 1 lands. This also skips @mediapipe/tasks-vision's
// vision_wasm_module_internal.* files, halving the copied WASM payload.
const mediapipeWasmDir = 'node_modules/@mediapipe/tasks-vision/wasm';

export default defineConfig({
  plugins: [
    viteStaticCopy({
      targets: [
        {
          src: `${mediapipeWasmDir}/vision_wasm_internal.*`,
          dest: 'mediapipe/wasm',
          rename: { stripBase: true },
        },
        {
          src: `${mediapipeWasmDir}/vision_wasm_nosimd_internal.*`,
          dest: 'mediapipe/wasm',
          rename: { stripBase: true },
        },
      ],
    }),
  ],
  build: {
    sourcemap: true,
    lib: {
      entry: 'src/main.ts',
      // Global exposed on window when loaded via a plain <script> tag from
      // the Shopify theme, e.g. window.CollectiblissTryOn.openTryOn({...}).
      name: 'CollectiblissTryOn',
      formats: ['iife'],
      fileName: () => 'collectibliss-tryon.js',
    },
    // Vite's lib mode already disables code splitting for a single-format
    // build like this one (confirmed: dist/ contains exactly one JS file),
    // which is what lets this bundle be dropped in via one <script> tag
    // with no bundler on the consuming (Shopify) side — no rolldownOptions
    // override needed to force that.
    //
    // NOTE: Vite 8 (Rolldown) does not polyfill `import.meta.url` in IIFE
    // output — it becomes `undefined`. Don't rely on it for locating
    // assets at runtime; src/assets.ts resolves the bundle's own base URL
    // via `document.currentScript` instead.
  },
});
