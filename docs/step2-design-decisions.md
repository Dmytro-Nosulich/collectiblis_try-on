# Step 2 Design Decisions — AR Earring Try-On

Captured 2026-09-09. Builds on `step1-technical-research.md`. This covers the UX/feature design for v1, informed by reviewing live examples (GlamAR demo store, Trillion demo store, Perfect Corp earrings showcase, Warby Parker, Sephora Virtual Artist, Pandora).

## Entry points & launch mechanism
- "Try on" button appears in two places: the collection/product-list card, and the product detail page (PDP).
- Both entry points call one shared launcher, e.g. `openTryOn({ productId, glbUrl, name })`.
- The GLB URL is attached as a `data-model-url` attribute on the button at template render time (Theme App Extension reads product media directly via Liquid) — no extra API round-trip needed on either page.

## Modal / container behavior
- Desktop / large screens: centered popup over a dark semi-transparent backdrop.
- Mobile: full-screen takeover.
- Body scroll locked while open.
- Camera stream tracks are explicitly stopped the instant the modal/view closes (avoids the browser's persistent "camera in use" indicator and privacy concerns).

## Mode chooser (shown before any camera/photo activity starts)
Three entry tiles, presented up front:
1. **Live Try-On** — real-time camera + AR overlay.
2. **Use a Photo** — user uploads their own photo; AR applied to the static image.
3. **Choose a Model** — pick from a small set of hardcoded model photos; same pipeline as "Use a Photo," for users who don't want to share a personal camera feed or photo.

A short privacy reassurance line is shown near the camera prompt / mode chooser: something like "Nothing is uploaded — this stays on your device." (True, since all processing is client-side; also improves camera-permission opt-in rates and reads well for EU visitors.)

## Mode 1: Live Try-On
- Live camera feed, mirrored (like a real mirror) — matches near-universal user expectation from camera apps, video calls, and other try-on tools. Ear-anchoring math accounts for the horizontal flip.
- Snapshot flow: button triggers a 3-2-1 countdown before capture (not an instant photo), then shows a share sheet.
  - Saved/shared snapshot is the **mirrored** version — matches exactly what the user saw and posed for (consistent with Snapchat/Instagram convention for front camera).
  - Share sheet: Web Share API (`navigator.share` with an image file) on mobile browsers that support it, surfacing Instagram/Facebook/etc. via the OS share sheet. Desktop needs separate handling per platform since Web Share for files isn't reliably supported there: a "download image" fallback, a Pinterest share link (supports direct image URLs), and a Facebook share dialog (requires the image hosted at a URL rather than a local blob).
- Split real-vs-virtual comparison: two side-by-side video panes fed by the same camera stream; only one pane renders the AR overlay. A toggle swaps which pane has AR on. Implemented as two panes rather than literally splitting one frame — keeps total rendered pixel count about the same as a single full-width view (negligible performance cost). On mobile, this control should stack top/bottom rather than left/right, since side-by-side halves get too narrow on phone screens.
- Product switching mid-session: **not included in v1** — one product per try-on session. To try a different earring, the customer closes the modal and launches try-on from a different product.
- No face/ear detected (out of frame, too far, hair covering ear): **no explicit guidance UI for v1** — the earring simply appears once tracking locks on and disappears if lost. (Flagged as something to revisit if user testing shows people get stuck without feedback.)
- Camera permission denied / no camera found: show a friendly error message (worded appropriately for "permission denied" vs. "no camera detected") with buttons to switch to Upload Photo or Choose a Model instead.
- Loading state while MediaPipe + the GLB model download (can take 1-3s, longer on slow mobile connections): centered spinner with a short status message (e.g. "Getting your camera ready...").

## Mode 2: Use a Photo
- User uploads their own photo; single-shot detection (MediaPipe's IMAGE mode, not VIDEO mode — no per-frame smoothing needed) places the earring once.
- Reuses the same rendering/anchoring pipeline as live mode, minus the video loop.

## Mode 3: Choose a Model
- Same pipeline as Mode 2, but with a small curated set of hardcoded model photos instead of a personal upload.
- Model photos should be chosen so the ear is clearly visible and not covered by hair, since occlusion is a known tracking weak point.

## Earring rendering rule
- Every earring renders on **both ears by default** (mirrored) — no per-product "single/pair/left/right" metadata for v1. Simpler rule that fits the common case (most vintage earring stock is sold as pairs); revisit if single statement pieces or mismatched sets are added to the catalog later.

## Suggested build sequencing (not a scope cut — everything above stays in v1, this is the order)
1. Live Try-On end-to-end: camera permission/error states, loading state, core anchoring/orientation quality, snapshot with countdown, save-to-device.
2. Upload Photo mode (reuses most of the live-mode pipeline).
3. Choose a Model mode (near-zero extra work once Upload Photo works).
4. Split real-vs-virtual comparison toggle.
5. Full multi-platform share polish (Web Share, Pinterest, Facebook dialog fallbacks).

## Open items intentionally deferred past v1
- On-screen tracking guidance ("move closer," face outline) if testing shows it's needed.
- Mid-session product switching.
- Per-product single/pair/left-right earring metadata.
