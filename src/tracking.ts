/**
 * Face/landmark tracking: loads MediaPipe's FaceLandmarker
 * (`@mediapipe/tasks-vision`) and runs detection against video/image
 * frames, approximates a left/right earlobe anchor point from the 478
 * returned landmarks (there is no dedicated ear landmark), and smooths
 * that point with a One Euro Filter (`1eurofilter`) to remove per-frame
 * jitter. See CLAUDE.md "Known technical challenges".
 *
 * Phase 1: raw landmark detection + debug visualization.
 * Phase 2: ear anchor approximation + One Euro Filter smoothing, still
 * drawn only as debug dots — no earring model yet.
 * Phase 3: `startTracking` takes an optional per-frame callback
 * (`TrackingFrameCallback`) so render.ts can consume the same smoothed
 * anchors (plus faceScale, for on-screen sizing) this file already computes
 * for its own debug drawing — one detection loop feeds both.
 */

import {
  DrawingUtils,
  FaceLandmarker,
  FilesetResolver,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';
import { OneEuroFilter } from '1eurofilter';
import { getAssetBaseUrl } from './assets.ts';

const WASM_BASE_PATH_SEGMENT = 'mediapipe/wasm';
const MODEL_ASSET_PATH_SEGMENT = 'mediapipe/models/face_landmarker.task';
const FPS_LOG_INTERVAL_FRAMES = 30;

// --- Ear anchor approximation -------------------------------------------
//
// MediaPipe's 478-point face mesh has no dedicated ear/earlobe landmark.
// We approximate one from the nearest cheek landmark plus a small tunable
// offset, expressed as a multiple of interocular (eye-to-eye) distance
// rather than a fixed [0,1]-normalized delta — a fixed normalized offset
// would visually drift off the earlobe as the user moves closer to/
// further from the camera, since the same normalized delta then covers a
// different amount of real face distance. Interocular distance is used as
// the scale reference (rather than jaw/mouth points) because it stays a
// roughly constant proportion of the face regardless of expression, and is
// computed in 3D (x/y/z, not just x/y) so it stays stable across head yaw
// too (a 2D-projected distance foreshortens as the head turns).
//
// The offset direction is NOT fixed to the screen's x/y axes — it rotates
// with the head's roll (derived from the forehead->chin landmarks), so it
// keeps pointing at the earlobe as the head tilts instead of sliding away
// from it. Without this, a straight-down screen-space offset stops meaning
// "down toward the ear" once the head is tilted (confirmed visually: dots
// drifted away from the face as roll increased). The rotation itself is
// done in aspect-ratio-corrected space (see toAspectCorrected below),
// since MediaPipe normalizes x by frame width and y by frame height
// separately — on a 16:9 webcam frame those are different physical units,
// and rotating without correcting for that would skew the direction.
//
// IMPORTANT: MediaPipe's landmark left/right are the SUBJECT's actual
// left/right (computed on the raw, unmirrored camera frame) — NOT
// mirror/screen-relative. In an unmirrored front-camera image the
// subject's actual right side appears on the LEFT of the image (smaller
// normalized x) and vice versa, like looking at a photo of someone facing
// you. This will need reconciling once mirrored display is added in
// Phase 5 (screen-left will then need to show the subject's actual right
// ear, not "right" as computed here).

/** Widest point of the face oval, roughly cheek/ear height (FACEMESH_FACE_OVAL). */
const RIGHT_CHEEK_LANDMARK_INDEX = 234; // subject's actual right cheek
const LEFT_CHEEK_LANDMARK_INDEX = 454; // subject's actual left cheek

/** Outer eye corners — used only as a per-frame face-scale reference. */
const RIGHT_EYE_OUTER_CORNER_LANDMARK_INDEX = 33;
const LEFT_EYE_OUTER_CORNER_LANDMARK_INDEX = 263;

/**
 * Top/bottom of the face oval (FACEMESH_FACE_OVAL) — stay roughly stable
 * relative to each other across normal head movement, so the vector
 * between them is used as a per-frame head-roll reference.
 */
const FOREHEAD_LANDMARK_INDEX = 10;
const CHIN_LANDMARK_INDEX = 152;

export interface EarAnchorTuning {
  outwardFactor: number;
  downFactor: number;
  backFactor: number;
  /**
   * Multiplier on the cheap per-frame yaw-magnitude proxy (see
   * computeYawMagnitude) that scales outwardFactor/downFactor/backFactor
   * down toward 0 when the head faces the camera near straight-on, and
   * back up toward their full values as the head turns. See
   * DEFAULT_EAR_ANCHOR_TUNING's docstring for why this direction (bigger
   * offset the more the head turns) is correct.
   */
  yawSensitivity: number;
  minCutoffHz: number;
  beta: number;
}

/**
 * Starting values, expressed as multiples of interocular distance for the
 * offset factors. `outwardFactor`/`downFactor` were recalibrated from an
 * earlier, too-large pair (0.12/0.55) using anthropometric ranges (outer
 * canthal ~85-95mm avg, bizygomatic breadth ~132-143mm avg — the ear sits
 * only modestly lateral to the cheekbone) plus visual evidence that the
 * anchor was landing well past the visible face silhouette. `downFactor`
 * in particular still has real uncertainty — MediaPipe doesn't document
 * where landmark 234/454 sits relative to true ear height — hence the
 * live-tuning panel in index.html rather than a one-shot guess.
 * `backFactor` (z) is invisible in this file's 2D debug dots (confirmed:
 * DrawingUtils.drawLandmarks only reads x*width/y*height, never z) — it
 * will start mattering once Phase 3 positions the earring in true 3D.
 * mincutoff/beta: mincutoff controls steady-state smoothing (lower = less
 * jitter when still, but more overall lag); beta controls how much the
 * cutoff opens up as the signal speeds up (higher = less lag during fast
 * head motion, but more jitter passes through).
 *
 * yawSensitivity=2.5 is new: after the above were confirmed correct by eye
 * at a yaw large enough to reveal an ear, straight-on testing (yaw≈0)
 * showed the same fixed magnitude overshooting well past the visible ear.
 * Geometrically this is expected, not just an empirical quirk: the
 * cheek->ear offset's out-of-plane ("back"/z) component is maximally
 * foreshortened in the 2D projection at yaw=0 (pointing along the
 * camera's viewing axis, contributing ~nothing to visible x/y) and
 * becomes increasingly visible as the head turns — so a magnitude
 * calibrated by eye at a turned angle will always look oversized
 * head-on. yawSensitivity scales outward/back by a shared multiplier (see
 * computeYawMagnitude/computeEarAnchor) — NOT downFactor: yaw is a
 * rotation about the head's roughly-vertical axis, and a rotation about
 * an axis never changes a vector's component along that same axis, so
 * "down" (the vertical component) is geometrically invariant under yaw
 * while "outward"/"back" (lateral/depth) live in the plane yaw actually
 * rotates. Confirmed empirically too: live testing found downFactor was
 * already correct once yawScaleMultiplier had ramped up to 1, so making
 * it unconditional just keeps it at that already-correct value at every
 * angle instead of collapsing toward 0 head-on. 2.5 means outward/back's
 * multiplier reaches 1 (their full, already-correct magnitude) once
 * computeYawMagnitude's output reaches 0.4 — a rough guess at "turned
 * enough to clearly reveal an ear" (~30-45°, back-derived from the
 * interocular-normalized cheek z-separation, not measured against a
 * protractor); refine live via the tuning panel like everything else
 * here.
 */
export const DEFAULT_EAR_ANCHOR_TUNING: Readonly<EarAnchorTuning> = Object.freeze({
  outwardFactor: 0.08,
  downFactor: 0.55,
  backFactor: 0.05,
  yawSensitivity: 2.5,
  minCutoffHz: 1.0,
  beta: 0.3,
});

/**
 * Live-tunable ear-anchor state, mutated in place by index.html's debug
 * sliders so changes apply instantly to the running camera feed with no
 * rebuild. An imported `let` binding can't be reassigned from outside its
 * defining module, so this uses one mutable object instead of several
 * `export let`s — index.html mutates its properties, which is fine.
 */
export const earAnchorTuning: EarAnchorTuning = { ...DEFAULT_EAR_ANCHOR_TUNING };

// `1eurofilter` is a SCALAR filter (confirmed by reading its source: it
// wraps two plain-number low-pass filters), so each ear anchor needs one
// filter instance per coordinate (x, y, z) — 6 total for two ears.
const EAR_ANCHOR_DCUTOFF_HZ = 1.0; // library default; rarely needs tuning
const EAR_ANCHOR_FILTER_INITIAL_FREQ_HZ = 30; // initial guess only; 1eurofilter recalculates real freq from consecutive call timestamps

// --- Debug overlay colors for ear anchors ---------------------------------
const RAW_EAR_ANCHOR_COLOR = 'rgba(255, 255, 255, 0.55)'; // dim, shared by both ears
const SMOOTHED_RIGHT_EAR_ANCHOR_COLOR = '#ff2d55';
const SMOOTHED_LEFT_EAR_ANCHOR_COLOR = '#2d7bff';

/** Normalized MediaPipe landmark space: x/y in [0,1] (frame width/height), z a relative depth proxy. */
export interface EarAnchorPoint {
  x: number;
  y: number;
  z: number;
}

interface Vector2 {
  x: number;
  y: number;
}

interface EarAnchors {
  left: EarAnchorPoint;
  right: EarAnchorPoint;
  /** Smoothed interocular distance, normalized by frame width (same units as x) — see render.ts's use as a scale reference. */
  faceScale: number;
}

/** One tracking frame's worth of data, handed to render.ts via `onFrame`. */
export interface TrackingFrame {
  left: EarAnchorPoint;
  right: EarAnchorPoint;
  faceScale: number;
  timestampMs: number;
}

/** Called once per detected frame with the smoothed frame data, or with `null` on a frame where no face was detected. */
export type TrackingFrameCallback = (frame: TrackingFrame | null) => void;

interface AxisFilters {
  x: OneEuroFilter;
  y: OneEuroFilter;
  z: OneEuroFilter;
}

interface EarAnchorFilters {
  left: AxisFilters;
  right: AxisFilters;
  faceScale: OneEuroFilter;
}

function distance3D(a: NormalizedLandmark, b: NormalizedLandmark): number {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

/**
 * Converts a landmark's x/y into a common physical unit. MediaPipe
 * normalizes x by frame width and y by frame height, which are different
 * units on a non-square (e.g. 16:9) frame — multiplying x by the frame's
 * aspect ratio (width/height) puts both components on the same
 * height-based scale, so vector math (angles, rotation) on the result is
 * geometrically correct instead of skewed by the frame's aspect ratio.
 */
function toAspectCorrected(point: NormalizedLandmark, aspect: number): Vector2 {
  return { x: point.x * aspect, y: point.y };
}

function rotateVector2(vector: Vector2, angleRadians: number): Vector2 {
  const cos = Math.cos(angleRadians);
  const sin = Math.sin(angleRadians);
  return {
    x: vector.x * cos - vector.y * sin,
    y: vector.x * sin + vector.y * cos,
  };
}

/** Restricts `value` to the closed range [min, max]. */
function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/**
 * Angle (radians) of the forehead->chin vector relative to straight-down,
 * measured in aspect-corrected space so it reflects true on-screen
 * geometry. 0 = head upright; nonzero tracks head roll (tilt).
 */
function computeHeadRollAngle(landmarks: NormalizedLandmark[], aspect: number): number {
  const forehead = toAspectCorrected(landmarks[FOREHEAD_LANDMARK_INDEX], aspect);
  const chin = toAspectCorrected(landmarks[CHIN_LANDMARK_INDEX], aspect);
  return Math.atan2(chin.x - forehead.x, chin.y - forehead.y);
}

/**
 * Cheap, unsigned proxy for how far the head has yawed (turned) away from
 * facing the camera — 0 facing the camera, growing with |yaw| in either
 * direction. Deliberately unsigned: the same magnitude is applied to both
 * ears (see computeEarAnchor), so a sign is never needed — which also
 * sidesteps this file's demonstrated history of left/right/mirroring sign
 * bugs (see the mirroring comment near the top of this file) for this
 * particular calculation.
 *
 * Rationale: yawing the head about its vertical axis moves one cheek
 * landmark closer to the camera (smaller z) and the other farther
 * (larger z). For two bilaterally-symmetric points at head-local
 * coordinates (d, 0) and (-d, 0) in the head's own lateral/depth plane,
 * rotating by yaw angle θ about the vertical axis puts them at depth
 * +/- d*sin(θ) relative to head center — so their z difference is exactly
 * 2*d*sin(θ): 0 at θ=0, monotonically increasing with |θ| up to a full
 * profile turn, never discontinuous. Real faces aren't perfectly
 * symmetric, and MediaPipe's z is inferred depth — noisier and less
 * certain than x/y — so expect a small nonzero reading even at true
 * yaw=0 and extra jitter at high yaw; neither breaks anything downstream,
 * since the result only ever scales a magnitude between 0 and 1x today's
 * calibrated offset (see computeEarAnchor) and flows through the same
 * One Euro filtering every other jitter source in the ear anchor does.
 *
 * Normalized by faceScale for the same reason every other offset in this
 * file is: stays comparable across users/camera distances. Guards
 * against faceScale being ~0 (degenerate/undetected face) instead of
 * dividing by it directly.
 */
function computeYawMagnitude(landmarks: NormalizedLandmark[], faceScale: number): number {
  const cheekZDifference = Math.abs(
    landmarks[RIGHT_CHEEK_LANDMARK_INDEX].z - landmarks[LEFT_CHEEK_LANDMARK_INDEX].z,
  );
  return faceScale > 0 ? cheekZDifference / faceScale : 0;
}

function computeEarAnchor(
  cheekLandmark: NormalizedLandmark,
  outwardSign: 1 | -1,
  faceScale: number,
  rollAngleRadians: number,
  yawMagnitude: number,
  aspect: number,
): EarAnchorPoint {
  // Shrinks the outward/back offset toward 0 as the head approaches facing
  // the camera straight-on, and grows it back up to today's calibrated
  // magnitudes as the head turns — see DEFAULT_EAR_ANCHOR_TUNING's
  // docstring for why this applies to outward/back but NOT down (yaw
  // rotates the lateral/depth plane, not the vertical axis). Clamped to
  // [0, 1] so it can only ever shrink today's calibrated offset, never
  // exceed it (no runaway growth approaching a full 90 degree profile
  // turn) and never flip it negative.
  const yawScaleMultiplier = clamp(yawMagnitude * earAnchorTuning.yawSensitivity, 0, 1);

  // Baseline offset as if the head were perfectly upright (matches the
  // pre-roll-compensation behavior exactly when rollAngleRadians is 0),
  // converted to aspect-corrected space, rotated by the head's actual
  // roll, then converted back — this keeps the anchor's real distance from
  // the cheek landmark constant regardless of roll; only its direction
  // follows the head.
  const baselineOffsetScaled: Vector2 = {
    x: outwardSign * earAnchorTuning.outwardFactor * faceScale * aspect * yawScaleMultiplier,
    // Not yaw-scaled: rotation about the (roughly vertical) yaw axis never
    // changes a vector's component along that same axis, so the vertical
    // "down" offset is geometrically invariant under yaw — unlike
    // outward/back, which live in the plane yaw actually rotates.
    y: earAnchorTuning.downFactor * faceScale,
  };
  // Negate the angle here: rotateVector2 is the textbook (y-up-convention)
  // rotation matrix, while computeHeadRollAngle's atan2(dx, dy) is
  // independently correct for this file's y-down screen convention. Both
  // are individually "standard," but combined without this negation they
  // rotate in opposite directions — verified algebraically:
  // rotateVector2((0,1), -θ) = (sinθ, cosθ), which matches the actual
  // measured forehead->chin direction at roll θ; without the negation it
  // produces the x-mirror of that. Confirmed as the cause of anchors
  // swinging to the wrong side when the head tilted. Left rotateVector2
  // and computeHeadRollAngle themselves untouched since both are correct
  // in isolation — the mismatch only exists at this one seam.
  const rotatedOffsetScaled = rotateVector2(baselineOffsetScaled, -rollAngleRadians);

  return {
    x: cheekLandmark.x + rotatedOffsetScaled.x / aspect,
    y: cheekLandmark.y + rotatedOffsetScaled.y,
    z: cheekLandmark.z + earAnchorTuning.backFactor * faceScale * yawScaleMultiplier,
  };
}

/**
 * Approximates left/right earlobe anchor points from the 478 face
 * landmarks. `aspect` is the video/canvas width divided by height, needed
 * to keep the roll-rotation geometrically correct (see toAspectCorrected).
 */
function computeRawEarAnchors(landmarks: NormalizedLandmark[], aspect: number): EarAnchors {
  const faceScale = distance3D(
    landmarks[RIGHT_EYE_OUTER_CORNER_LANDMARK_INDEX],
    landmarks[LEFT_EYE_OUTER_CORNER_LANDMARK_INDEX],
  );
  const rollAngleRadians = computeHeadRollAngle(landmarks, aspect);
  const yawMagnitude = computeYawMagnitude(landmarks, faceScale);
  return {
    // Right cheek sits on the image's smaller-x side (unmirrored frame), so
    // "outward" (further right, away from face center) means decreasing x
    // before roll rotation is applied.
    right: computeEarAnchor(
      landmarks[RIGHT_CHEEK_LANDMARK_INDEX],
      -1,
      faceScale,
      rollAngleRadians,
      yawMagnitude,
      aspect,
    ),
    left: computeEarAnchor(
      landmarks[LEFT_CHEEK_LANDMARK_INDEX],
      1,
      faceScale,
      rollAngleRadians,
      yawMagnitude,
      aspect,
    ),
    faceScale,
  };
}

function createOneEuroFilter(): OneEuroFilter {
  return new OneEuroFilter(
    EAR_ANCHOR_FILTER_INITIAL_FREQ_HZ,
    earAnchorTuning.minCutoffHz,
    earAnchorTuning.beta,
    EAR_ANCHOR_DCUTOFF_HZ,
  );
}

function createAxisFilters(): AxisFilters {
  return { x: createOneEuroFilter(), y: createOneEuroFilter(), z: createOneEuroFilter() };
}

function createEarAnchorFilters(): EarAnchorFilters {
  return {
    left: createAxisFilters(),
    right: createAxisFilters(),
    faceScale: createOneEuroFilter(),
  };
}

/**
 * Runs one scalar value through its One Euro Filter. Pulls the live tuning
 * values every call (not just at construction) so a tuning-panel change
 * takes effect immediately on filters already mid-session: setMinCutoff/
 * setBeta only change the cutoff going forward, they don't reset smoothing
 * state.
 */
function filterScalar(filter: OneEuroFilter, value: number, timestampSeconds: number): number {
  filter.setMinCutoff(earAnchorTuning.minCutoffHz);
  filter.setBeta(earAnchorTuning.beta);
  return filter.filter(value, timestampSeconds);
}

/**
 * Runs one raw ear anchor point through its x/y/z One Euro Filters.
 * `timestampSeconds` must be strictly increasing across calls for a given
 * `filters` instance — pass the same per-frame timestamp (converted from
 * `performance.now()` ms to seconds) for both ears each frame.
 */
function filterEarAnchor(
  filters: AxisFilters,
  point: EarAnchorPoint,
  timestampSeconds: number,
): EarAnchorPoint {
  return {
    x: filterScalar(filters.x, point.x, timestampSeconds),
    y: filterScalar(filters.y, point.y, timestampSeconds),
    z: filterScalar(filters.z, point.z, timestampSeconds),
  };
}

/** visibility is required by NormalizedLandmark's type but unused for a synthetic drawn point. */
function toDrawableLandmark(point: EarAnchorPoint): NormalizedLandmark {
  return { x: point.x, y: point.y, z: point.z, visibility: 1 };
}

async function createFaceLandmarker(): Promise<FaceLandmarker> {
  const assetBaseUrl = getAssetBaseUrl();
  const vision = await FilesetResolver.forVisionTasks(`${assetBaseUrl}${WASM_BASE_PATH_SEGMENT}`);
  const modelAssetPath = `${assetBaseUrl}${MODEL_ASSET_PATH_SEGMENT}`;

  // MediaPipe has no silent GPU->CPU auto-fallback — `delegate` is an
  // explicit choice, and GPU delegate init can throw depending on
  // browser/hardware support. Try GPU first, fall back to CPU on failure.
  try {
    return await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numFaces: 1,
    });
  } catch (gpuInitError) {
    console.warn(
      '[CollectiblissTryOn] GPU delegate init failed, falling back to CPU delegate.',
      gpuInitError,
    );
    return FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath, delegate: 'CPU' },
      runningMode: 'VIDEO',
      numFaces: 1,
    });
  }
}

function resizeCanvasToVideo(canvas: HTMLCanvasElement, video: HTMLVideoElement): void {
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
}

/**
 * Starts a requestAnimationFrame-driven detection loop against `video`,
 * drawing a dot per landmark onto `canvas` (sized to the video's intrinsic
 * pixel dimensions) so tracking quality/jitter can be visually confirmed.
 * Logs a rolling average frame time/FPS every 30 frames as a performance
 * baseline. Returns a cleanup function that stops the loop and releases the
 * landmarker.
 *
 * `onFrame`, if given, is called once per tick with this frame's smoothed
 * ear anchors + faceScale (or `null` on a frame with no detected face) —
 * this is how render.ts drives the Three.js scene off the same detection
 * loop instead of running a second one.
 */
export async function startTracking(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  onFrame?: TrackingFrameCallback,
): Promise<() => void> {
  const faceLandmarker = await createFaceLandmarker();
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('[CollectiblissTryOn] 2D canvas context unavailable');
  }
  const drawingUtils = new DrawingUtils(ctx);

  // Created once per startTracking() call, alongside faceLandmarker, so
  // filter state persists across frames (that's what smooths anything) but
  // doesn't leak into a future call if this is ever invoked again. No
  // explicit cleanup needed: OneEuroFilter/LowPassFilter hold only plain
  // numeric fields, no external resources.
  const earAnchorFilters = createEarAnchorFilters();

  let rafHandle = 0;
  let frameCount = 0;
  let frameTimeAccumulatorMs = 0;

  const tick = (): void => {
    rafHandle = requestAnimationFrame(tick);

    const frameStartMs = performance.now();
    const result = faceLandmarker.detectForVideo(video, frameStartMs);
    const frameTimeMs = performance.now() - frameStartMs;

    resizeCanvasToVideo(canvas, video);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const landmarks = result.faceLandmarks[0];
    if (landmarks) {
      drawingUtils.drawLandmarks(landmarks, { radius: 1.5, color: '#00ff88' });

      const aspect = canvas.width / canvas.height;
      const rawAnchors = computeRawEarAnchors(landmarks, aspect);
      // 1eurofilter expects seconds; performance.now() is ms.
      const timestampSeconds = frameStartMs / 1000;
      const smoothedRight = filterEarAnchor(
        earAnchorFilters.right,
        rawAnchors.right,
        timestampSeconds,
      );
      const smoothedLeft = filterEarAnchor(
        earAnchorFilters.left,
        rawAnchors.left,
        timestampSeconds,
      );
      const smoothedFaceScale = filterScalar(
        earAnchorFilters.faceScale,
        rawAnchors.faceScale,
        timestampSeconds,
      );

      onFrame?.({
        left: smoothedLeft,
        right: smoothedRight,
        faceScale: smoothedFaceScale,
        timestampMs: frameStartMs,
      });

      // Dim raw anchor dots drawn first, bright smoothed dots on top — lets
      // jitter (raw dot shaking at rest) vs. lag (smoothed dot trailing
      // during motion) be judged in the same view.
      drawingUtils.drawLandmarks([toDrawableLandmark(rawAnchors.right)], {
        radius: 3,
        color: RAW_EAR_ANCHOR_COLOR,
      });
      drawingUtils.drawLandmarks([toDrawableLandmark(rawAnchors.left)], {
        radius: 3,
        color: RAW_EAR_ANCHOR_COLOR,
      });
      drawingUtils.drawLandmarks([toDrawableLandmark(smoothedRight)], {
        radius: 5,
        color: SMOOTHED_RIGHT_EAR_ANCHOR_COLOR,
      });
      drawingUtils.drawLandmarks([toDrawableLandmark(smoothedLeft)], {
        radius: 5,
        color: SMOOTHED_LEFT_EAR_ANCHOR_COLOR,
      });
    } else {
      onFrame?.(null);
    }

    frameCount += 1;
    frameTimeAccumulatorMs += frameTimeMs;
    if (frameCount % FPS_LOG_INTERVAL_FRAMES === 0) {
      const avgFrameTimeMs = frameTimeAccumulatorMs / FPS_LOG_INTERVAL_FRAMES;
      console.log(
        `[CollectiblissTryOn] detectForVideo avg over ${FPS_LOG_INTERVAL_FRAMES} frames: ` +
          `${avgFrameTimeMs.toFixed(2)}ms (${(1000 / avgFrameTimeMs).toFixed(1)} fps)`,
      );
      frameTimeAccumulatorMs = 0;
    }
  };

  rafHandle = requestAnimationFrame(tick);

  return () => {
    cancelAnimationFrame(rafHandle);
    faceLandmarker.close();
  };
}
