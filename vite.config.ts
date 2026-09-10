import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';

// MediaPipe's SIMD + non-SIMD WASM runtimes for the vision tasks bundle.
// We only copy the plain (non-ES-module) variants, since FilesetResolver is
// called with useModule left at its default (false) — see src/tracking.ts
// once Phase 1 lands. This also skips @mediapipe/tasks-vision's
// vision_wasm_module_internal.* files, halving the copied WASM payload.
const mediapipeWasmDir = 'node_modules/@mediapipe/tasks-vision/wasm';

// FaceLandmarker's model file, self-hosted rather than fetched from Google's
// storage URL at runtime — vendored under vendor/ (not npm-managed, not
// Vite's public/ convention) so this one file can be copied the same way as
// the WASM assets above. Fetch/refresh via `npm run fetch:model`.
const mediapipeModelDir = 'vendor/mediapipe';

// three.js's own self-contained gltf Draco decoder set (decoder JS + wasm +
// wrapper). DRACOLoader fetches these by URL at runtime (unlike meshopt
// decoding, which is a self-contained wasm module bundled straight into the
// JS import — see render.ts), so they need to be copied/served like the
// MediaPipe WASM assets above.
const dracoDecoderDir = 'node_modules/three/examples/jsm/libs/draco/gltf';

// Placeholder earring GLBs (Phase 3/4 debug harness only — see
// scripts/generate-placeholder-earring.mjs; the glob covers both the rigid
// hoop and the Phase 4 dangly test model). Real per-product GLBs come from
// Shopify's own CDN at runtime and are never copied here.
const placeholderEarringDir = 'vendor/earrings';

// Curated "Choose a Model" photos (Phase 10) — free-license placeholders,
// see src/ui/modelPhotos.ts for the source/license record. Flagged there
// for replacement with final licensed photos before launch.
const curatedModelPhotosDir = 'vendor/model-photos';

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
        {
          src: `${mediapipeModelDir}/face_landmarker.task`,
          dest: 'mediapipe/models',
          rename: { stripBase: true },
        },
        {
          src: `${dracoDecoderDir}/*`,
          dest: 'draco',
          rename: { stripBase: true },
        },
        {
          src: `${placeholderEarringDir}/placeholder-earring*.glb`,
          dest: 'earrings',
          rename: { stripBase: true },
        },
        {
          src: `${curatedModelPhotosDir}/*`,
          dest: 'model-photos',
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
