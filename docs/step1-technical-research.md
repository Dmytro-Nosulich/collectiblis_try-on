# Step 1 Technical Research — AR Earring Try-On

Research conducted 2026-09-09 to inform the full technical/design document. Builds on `ar-earring-tryon-reference.md` (prior chat) — confirms most of that doc still holds, with specifics below.

## Face/ear tracking
- **MediaPipe Face Landmarker** (`@mediapipe/tasks-vision`, Apache 2.0, free, no usage caps) remains the right choice. Runs fully client-side via WASM, unified web/Android/iOS/Python API. 478 3D landmarks + 52 blendshapes.
- No dedicated ear landmark exists (confirmed still true in 2026) — must approximate from jaw/cheek landmarks + tuned offset. Training data concentrates on frontal face, so accuracy degrades at extreme head angles and with hair occlusion.
- TensorFlow.js `face-landmarks-detection` is effectively a wrapper around the same underlying model — no advantage over using MediaPipe directly.
- Jeeliz FaceFilter (MIT) still viable as a lighter-weight alternative worth prototyping against on low-end phones, but MediaPipe stays primary.

## 3D rendering
- **Three.js** over Babylon.js: smaller footprint, native GLTFLoader/DRACOLoader/meshopt support for compressed GLB, simpler transparent-canvas compositing. Babylon.js is more "batteries included" (physics/GUI) — unnecessary overhead here.
- Both now ship WebGPU renderers, but default to the WebGL2 renderer for v1 (see browser support below); treat WebGPU as a later progressive enhancement, not a v1 dependency.

## Language & tooling
- **TypeScript** — both `@mediapipe/tasks-vision` and OneEuroFilter (via `OneEuroFilter-ts`) have TS-friendly builds; the landmark math / filtering / 3D transform code benefits from type safety.
- **Vite** for build tooling — fast dev loop, tree-shaking, clean static output.
- Ship the widget as one self-contained bundle (JS + wasm + model assets) that drops into a Shopify theme without needing a build step on Shopify's side.

## Browser support
- getUserMedia, WebGL2, and WASM (incl. SIMD) are supported on all current desktop and mobile browsers (Chrome/Edge, Firefox, Safari desktop+iOS, Samsung Internet) — safe v1 baseline.
- WebGPU only recently went universal and mobile coverage is still patchy: Chrome/Edge 113+ desktop, Firefox 141+ desktop, Safari on macOS Tahoe 26 / iOS 26; Android Chrome 121+ (Qualcomm/ARM GPUs only), Android Firefox still WIP. Not ready to depend on for v1.
- iOS Safari specifics to design around: camera access needs HTTPS + a direct user gesture (no autoplay start — must be a "Try it on" tap); doesn't work in a cross-origin iframe without `allow="camera"` (favors injecting the widget directly into the page DOM rather than a sandboxed iframe); WebGL context can be reclaimed more aggressively under memory pressure than desktop — widget should handle context-loss gracefully.
- No mobile browser supports WebXR camera passthrough — confirms the architecture decision to avoid WebXR entirely.

## Mobile optimization
- Throttle MediaPipe's detection loop (e.g. every other frame, or a downscaled ~480p input) separately from the full-res video display and 60fps Three.js render — inference is the expensive part, not rendering.
- Keep earring GLBs low-poly with compressed textures (KTX2/Basis) and Draco/meshopt geometry compression.
- Lazy-load MediaPipe WASM + model files only after the user taps "try on," not on initial page load — protects product page load speed/SEO.
- Detect low-end devices (core count, pixel ratio) and fall back to lower canvas resolution / skip smoothing.

## Shopify integration
- 3D models upload as native product media (Admin → product → Media → Add 3D model); Shopify's Admin GraphQL API exposes them as `Model3d` / `Model3dSource` objects with a direct GLB URL — no separate asset pipeline needed.
- For a non-Plus, non-headless store without an admin configuration UI need, a lightweight custom app or even a maintained JSON manifest (product ID → GLB URL) is enough; no need for a full public Shopify app.
- **Theme App Extension** (app embed block) is the modern supported way to inject the widget's script/CSS into the storefront theme — survives theme updates, merchant can toggle from the theme editor. Preferred over the legacy ScriptTag API pattern.
- Hosting: static CDN (Vercel/Netlify/Cloudflare Pages) is sufficient since all compute is client-side; the theme just loads the widget JS from that URL.

## Net changes vs. the original reference doc
Architecture, stack, and known challenges from the prior doc all hold up. New/refined points from this pass: TypeScript + Vite as concrete tooling choice, Theme App Extension as the specific Shopify embedding mechanism (vs. generic "package for embedding"), a concrete mobile perf plan (frame throttling, lazy-load, compression), and current (2026) browser/WebGPU support data confirming WebGL2 + WASM as the safe v1 baseline.
