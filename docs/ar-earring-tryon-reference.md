# AR Earring Try-On — Project Reference

Browser-based "try it on" feature for a vintage jewelry Shopify store: user picks a piece (starting with earrings), turns on their webcam, and sees it rendered on themselves in real time.

## Decision: build in-house

Vendor SDKs (Banuba, GlamAR, MirrAR, Trillion) exist and work well, but recurring cost ($49–$1,599+/mo, or custom/volume-based quotes) isn't justified for a small store. Building it is a deliberate choice — not just cheaper, but also the more interesting path, and the scope is manageable for an experienced dev using Claude Code. Not aiming for perfection, just a working v1.

## Architecture

Everything runs client-side in the browser — no server-side ML, no backend compute cost.

```
Camera feed (getUserMedia)
        │
        ├──────────────────────────────────┐
        ▼                                   │ (raw video passthrough)
Face & ear detection                        │
(MediaPipe, runs in-browser via WASM)       │
        │                                   │
        ▼                                   │
3D render engine (Three.js)                 │
 - loads earring GLB model                  │
 - anchors it to detected ear point         │
 - transparent background                   │
        │                                   │
        ▼                                   ▼
          Composite output
     (3D overlay merged onto video, drawn to <canvas>)
```

This is **not** WebXR — browser support for camera-passthrough WebXR is inconsistent (notably Safari/desktop). Production web AR try-on tools use video + ML landmarks + WebGL overlay, not true world-anchored AR.

## Recommended free/open-source stack

| Piece | Tool | License | Notes |
|---|---|---|---|
| Camera access | `getUserMedia()` | Native browser API | No library needed |
| Face/ear landmark detection | **MediaPipe Face Landmarker** | Apache 2.0 | 478-point face mesh, runs via WASM. No literal ear/earlobe point — approximate using face-edge landmarks near the jaw + a tuned offset |
| Alt. landmark detection | **Jeeliz FaceFilter** | MIT | Lighter weight, built specifically for WebAR filters, integrates with Three.js/Babylon.js directly. Fewer fine-grained landmarks than MediaPipe — better for head-pose attachment (glasses/hats) than a precise ear point. Worth prototyping against MediaPipe. |
| 3D rendering | **Three.js** | MIT | `GLTFLoader` loads GLB natively |
| Jitter smoothing | **`1eurofilter`** (npm) | MIT | One Euro Filter — adaptive smoothing, doesn't lag on fast movement. Needed because raw landmark output trembles even when the head is still |
| 3D model source | Shopify Admin API / manual download | Free with any Shopify plan | Shopify's product 3D models are natively **GLB** (binary glTF) — no conversion needed before loading into Three.js |
| Hosting | Vercel / Netlify / Cloudflare Pages / GitHub Pages | Free tier | All compute (tracking + rendering) happens in the visitor's browser, so a static host is enough |

## Known technical challenges (earrings specifically)

- **No true ear landmark on the web.** Unlike Snap's Lens Studio (which has a proprietary Ear Binding component with tragus/lobe/helix points, native-app only), there's no equivalent for browser MediaPipe/Jeeliz. Ear position has to be approximated from nearby face-mesh points + a manually tuned offset.
- **Occlusion.** Hair covering the ear breaks tracking much more easily than it breaks face tracking generally.
- **Orientation.** The earring needs to hang naturally as the head tilts, not stay flat — this is the part that takes the most iteration/tuning time.
- **Jitter.** Raw per-frame landmark noise makes jewelry visibly shake without smoothing (see `1eurofilter` above).

## Reference project: OpenMakeupSDK

https://github.com/ehsanwwe/OpenMakeupSDK (MIT license, npm: `open-makeup-sdk`)

A solo-developer, MIT-licensed, real-time AR makeup + face-reshape SDK: MediaPipe FaceMesh + Three.js + custom WebGL/GLSL shaders, 100% client-side, no backend.

**Useful as:** an architecture reference. Validates the same video → MediaPipe → WebGL → composite pattern, and its init/config API design (`video`, `renderCanvas`, `assetsBaseUrl`, `mediapipeBaseUrl`, `mk.apply(category, opts)`) is a clean example worth reading before structuring your own SDK-style API.

**Doesn't transfer directly:** makeup is a shader painting color/texture *onto the tracked face mesh surface itself* (UV-mapped to the existing geometry). An earring is a separate rigid 3D object that must be positioned/rotated/scaled *relative to* a tracked point — a different technique. The tracking+rendering plumbing is a good reference; the attachment logic itself has to be built from scratch.

**Caveat:** brand new (first release ~3 months old at time of writing), zero stars, unproven — reading material, not a dependency to build on top of.

## Vendor landscape (for reference, in case priorities change)

| | Time to launch | Cost | Customization | Ear-tracking quality | Maintenance |
|---|---|---|---|---|---|
| Banuba Virtual Try-On | Days (Shopify extension) | $49–$1,599/mo | Widget-level, limited | Production-tuned | Vendor-handled |
| GlamAR / Trillion / MirrAR (jewelry-specific) | Days, plug-and-play SDK | Custom quote, often volume-based | Widget-level, jewelry-focused UX | Purpose-built for jewelry | Vendor-handled |

Trillion has a live demo store worth trying to benchmark quality: search "Trillion jewelry AR demo store."

## Suggested build order

1. Get MediaPipe Face Landmarker running with raw webcam feed, log landmarks to console.
2. Pick/approximate an ear anchor point from the face mesh; draw a simple debug marker there to validate stability.
3. Add `1eurofilter` smoothing to that anchor point.
4. Set up Three.js scene with transparent background, load one test earring GLB, position it at the anchor each frame.
5. Composite the Three.js canvas over the video feed.
6. Iterate on orientation/hang behavior as the head tilts.
7. Wire up model switching (multiple earrings) and pull real GLBs from the Shopify store.
8. Package for embedding into the Shopify storefront (script/section) and deploy to a free static host.
