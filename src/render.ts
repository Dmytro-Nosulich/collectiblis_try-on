/**
 * 3D rendering: sets up a Three.js scene with a transparent-background
 * WebGLRenderer, loads earring GLB models via GLTFLoader (with
 * DRACOLoader/meshopt for compressed GLB), and positions/rotates the
 * loaded model at the ear anchor from tracking.ts each frame. The
 * resulting canvas is layered on top of the video (Live Try-On) or image
 * (Upload Photo) element it's tracking to composite the AR overlay.
 *
 * Implemented in Phase 3 (anchored placement) and Phase 4 (orientation/hang
 * tuning as the head tilts). Phase 9 widens createEarringScene's source
 * parameter to also accept a decoded `<img>`, for one-shot placement on an
 * uploaded photo instead of a live camera feed.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';
import { getAssetBaseUrl } from './assets.ts';
import type { EarAnchorPoint, HeadRotation, TrackingFrame } from './tracking.ts';

// DRACOLoader fetches its decoder files by URL at runtime (unlike
// meshopt's decoder, which is a self-contained wasm module bundled
// straight into the module import below). LoaderUtils.resolveURL's
// relative-path branch is plain `path + url` string concatenation with no
// separator inserted, so this MUST end in a trailing slash — unlike
// tracking.ts's WASM_BASE_PATH_SEGMENT, which has no trailing slash
// because FilesetResolver joins paths differently. Copy-pasting that other
// convention here would silently produce a broken URL.
const DRACO_DECODER_PATH_SEGMENT = 'draco/';

// Adult average interocular distance, used to convert the tracked
// faceScale (interocular distance normalized by frame width) into a
// pixels-per-meter conversion with no camera calibration step — see
// placeEar's docstring.
const ASSUMED_INTEROCULAR_DISTANCE_METERS = 0.063;

const ORTHOGRAPHIC_NEAR = -2000;
const ORTHOGRAPHIC_FAR = 2000;

// Phase 11: how long to wait after a 'webglcontextlost' event before giving
// up on the browser/three.js's own silent restoration (see
// createEarringScene's context-loss handling) and surfacing an error to the
// caller instead. Unvalidated guess — confirm against a real forced-loss
// test and real iOS Safari memory-pressure conditions.
const CONTEXT_LOST_RECOVERY_TIMEOUT_MS = 3000;

// Phase 4: rotation axes in this file's world space (X=screen-right,
// Y=screen-down, Z=depth — see the OrthographicCamera setup below). Roll
// rotates about the screen-normal (Z), pitch about the horizontal (X), yaw
// about the vertical (Y). Composed as qYaw * qPitch * qRoll (roll
// innermost) — a standard order, but per tracking.ts's RotationTuning
// comment, treat each axis's SIGN as unverified until checked against a
// real webcam, not as a settled derivation (this file has already hit one
// rotation-direction sign bug from assuming instead of testing — see
// tracking.ts's computeEarAnchor).
const ROLL_AXIS = new THREE.Vector3(0, 0, 1);
const PITCH_AXIS = new THREE.Vector3(1, 0, 0);
const YAW_AXIS = new THREE.Vector3(0, 1, 0);

/**
 * Composes one head-rotation quaternion, shared identically by both ears'
 * anchorGroup (see createEarInstance's docstring for why no per-ear mirror
 * flip is needed here: mirroring is a static, CHILD-level transform, so a
 * parent-level rotation like this applies the same way regardless of which
 * ear it's on — physically, two earrings on a tilting head rotate the same
 * absolute direction).
 */
function computeHeadRotationQuaternion(headRotation: HeadRotation): THREE.Quaternion {
  const rollQuaternion = new THREE.Quaternion().setFromAxisAngle(
    ROLL_AXIS,
    headRotation.rollRadians,
  );
  const pitchQuaternion = new THREE.Quaternion().setFromAxisAngle(
    PITCH_AXIS,
    headRotation.pitchRadians,
  );
  const yawQuaternion = new THREE.Quaternion().setFromAxisAngle(YAW_AXIS, headRotation.yawRadians);
  return yawQuaternion.multiply(pitchQuaternion).multiply(rollQuaternion);
}

// Created once and reused across createEarringScene() calls: DRACOLoader
// spins up a worker pool on first decode, and constructing a fresh
// loader per load would leak workers with no corresponding dispose().
let sharedGltfLoader: GLTFLoader | undefined;

function getGltfLoader(): GLTFLoader {
  if (!sharedGltfLoader) {
    const dracoLoader = new DRACOLoader();
    dracoLoader.setDecoderPath(`${getAssetBaseUrl()}${DRACO_DECODER_PATH_SEGMENT}`);
    sharedGltfLoader = new GLTFLoader();
    sharedGltfLoader.setDRACOLoader(dracoLoader);
    sharedGltfLoader.setMeshoptDecoder(MeshoptDecoder);
  }
  return sharedGltfLoader;
}

/**
 * Per-ear wrapper hierarchy:
 *   anchorGroup (per-frame: position + uniform scale + Phase 4's rotation)
 *     -> orientationAdapter (static, set once at load: mirroring + the
 *          camera-convention Y-flip below)
 *          -> modelRoot (a clone of the loaded GLB's scene)
 *
 * Splitting "which ear + fixed axis conventions" (set once) from "where/
 * how big/how rotated this frame" (set every frame) means the per-frame
 * update never has to think about mirroring or the Y-flip — and, per
 * computeHeadRotationQuaternion's docstring, means a single shared rotation
 * quaternion applies correctly to both ears with no per-ear adjustment.
 */
interface EarInstance {
  anchorGroup: THREE.Group;
  /** 1 for one ear's wrapper, -1 for the other's — kept so setEarModel (model swapping) can rebuild orientationAdapter without the caller re-specifying it. */
  mirrorX: 1 | -1;
}

/** Disposes every mesh's geometry/material under `object` (used both for full scene teardown and for swapping a single ear's model). */
function disposeObject3D(object: THREE.Object3D): void {
  object.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) {
      return;
    }
    child.geometry.dispose();
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    for (const material of materials) {
      material.dispose();
    }
  });
}

/**
 * Builds the static orientationAdapter->modelRoot subtree for one ear from
 * a freshly cloned model scene, replacing whatever was previously attached
 * to `instance.anchorGroup` (disposing it first) — used both for initial
 * setup and for live model swapping (see EarringScene.loadModel).
 *
 * `mirrorX` is 1 for one ear's wrapper and -1 for the other's, implementing
 * CLAUDE.md's "every earring renders on both ears by default (mirrored)"
 * rule — which one is "as authored" is arbitrary for a generic earring.
 *
 * The `-1` on Y is NOT arbitrary: render.ts's OrthographicCamera (below) is
 * deliberately set up so world Y is numerically equal to canvas-pixel Y (0
 * at top, increasing downward), matching MediaPipe's own convention. But
 * glTF's own convention is the opposite (+Y = physically up), so without
 * this flip a model authored to "hang down" in its own local space (see
 * scripts/generate-placeholder-earring.mjs) would render pointing toward
 * the top of the canvas instead of hanging below the ear anchor.
 */
function setEarModel(instance: EarInstance, modelScene: THREE.Group): void {
  for (const existingChild of [...instance.anchorGroup.children]) {
    disposeObject3D(existingChild);
    instance.anchorGroup.remove(existingChild);
  }

  const modelRoot = modelScene.clone(true);
  const orientationAdapter = new THREE.Group();
  orientationAdapter.scale.set(instance.mirrorX, -1, 1);
  orientationAdapter.add(modelRoot);
  instance.anchorGroup.add(orientationAdapter);
}

function createEarInstance(modelScene: THREE.Group, mirrorX: 1 | -1): EarInstance {
  const anchorGroup = new THREE.Group();
  // Hidden until the first successful detection — see updateFrame. Directly
  // implements CLAUDE.md's "no face-not-detected UI... earring just
  // appears when tracking locks on."
  anchorGroup.visible = false;

  const instance: EarInstance = { anchorGroup, mirrorX };
  setEarModel(instance, modelScene);
  return instance;
}

/**
 * Positions, scales, and rotates one ear's anchorGroup for this frame.
 * Position is a direct pixel mapping (anchor.x/y are normalized by frame
 * width/height; z uses the same width-based scale as x, per MediaPipe's
 * documented convention that z is roughly on the same scale as x).
 *
 * Scale is deliberately NOT driven by perspective/camera depth — there's
 * no camera intrinsics calibration available, so instead faceScale (the
 * tracked interocular distance, normalized by frame width) is converted
 * through a fixed real-world average interocular distance into a
 * self-calibrating pixels-per-meter factor: as the user's face gets
 * closer to the camera, faceScale grows and so does the rendered size,
 * with no calibration step required. This assumes the GLB is authored in
 * real-world meters (glTF's required unit convention).
 *
 * Rotation (Phase 4) is one shared quaternion computed from headRotation —
 * see computeHeadRotationQuaternion's docstring for why the same quaternion
 * is correct for both ears with no per-ear sign flip.
 */
function placeEar(
  anchorGroup: THREE.Group,
  anchor: EarAnchorPoint,
  faceScale: number,
  headRotation: HeadRotation,
  canvasWidth: number,
  canvasHeight: number,
): void {
  anchorGroup.position.set(anchor.x * canvasWidth, anchor.y * canvasHeight, anchor.z * canvasWidth);
  const pixelsPerMeter = (faceScale * canvasWidth) / ASSUMED_INTEROCULAR_DISTANCE_METERS;
  anchorGroup.scale.setScalar(pixelsPerMeter);
  anchorGroup.quaternion.copy(computeHeadRotationQuaternion(headRotation));
  anchorGroup.visible = true;
}

export interface CreateEarringSceneOptions {
  /**
   * Caps renderer.setPixelRatio() — omitted (or 1) preserves this file's
   * pre-Phase-11 behavior exactly, which is what every index.html
   * debug-harness call site relies on by not passing this option at all.
   * Typically window.devicePixelRatio itself, pre-capped by
   * deviceCapabilities.ts's detectDeviceTier() before being passed in here
   * — see that module for why devicePixelRatio is treated as a workload
   * multiplier to cap, not a device-tier signal on its own.
   */
  pixelRatioCap?: number;
  /**
   * Fires only if a lost WebGL context doesn't restore within
   * CONTEXT_LOST_RECOVERY_TIMEOUT_MS — see the context-loss handling below
   * this function for why most losses need no app-level recovery code at
   * all, and why this is a last-resort signal rather than the primary
   * recovery mechanism.
   */
  onContextLost?: () => void;
}

export interface EarringScene {
  /** Positions both ear instances from this frame's smoothed data (or hides them on `null`) and renders. */
  updateFrame(frame: TrackingFrame | null): void;
  /** Loads a different GLB and swaps it onto both ears in place — anchorGroup's live position/scale/rotation are untouched, so there's no visual jump. Debug-harness-only for now (v1 has no mid-session product switching, per CLAUDE.md). */
  loadModel(glbUrl: string): Promise<void>;
  /** Releases GPU resources (geometries/materials/renderer) and the context-loss listeners below. Does not touch the shared GLTFLoader/DRACOLoader singleton. */
  dispose(): void;
}

/**
 * Sets up a Three.js scene layered on top of `source` (via `canvas`,
 * positioned by the caller the same way the Phase 1/2 debug canvas is) and
 * loads `glbUrl` once, instancing it onto both ears (mirrored — see
 * createEarInstance). `glbUrl` is taken as a fully-resolved absolute URL,
 * not resolved through getAssetBaseUrl() here: in production this is an
 * arbitrary Shopify CDN URL (TryOnOptions.glbUrl) with nothing to do with
 * where this widget's own script was loaded from — only the Draco decoder
 * path above is one of this widget's own bundled runtime assets.
 *
 * `source` is a live `<video>` for Live Try-On (Phases 5-8) or a fully
 * decoded `<img>` for Upload Photo (Phase 9) / Choose a Model (Phase 10) —
 * see syncSizeToSource. Either way the caller must guarantee valid
 * intrinsic dimensions before calling this (a ready camera stream, or an
 * already-`decode()`d image).
 */
export async function createEarringScene(
  canvas: HTMLCanvasElement,
  source: HTMLVideoElement | HTMLImageElement,
  glbUrl: string,
  options: CreateEarringSceneOptions = {},
): Promise<EarringScene> {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    alpha: true,
    antialias: true,
    // Phase 6's snapshot capture reads this canvas's pixels via drawImage
    // well after the render call (after a 3-2-1 countdown, not
    // synchronously inside updateFrame) — without this, the browser is
    // free to clear/swap the drawing buffer once it's done compositing to
    // screen, so the readback isn't guaranteed to see the last frame.
    preserveDrawingBuffer: true,
  });
  // alpha: true alone already defaults WebGLRenderer's clear alpha to 0;
  // set explicitly anyway as documentation against relying on an unstated
  // default.
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  // Phase 11: was deliberately left uncalled through Phase 3-10 (pixelRatio
  // stuck at Three.js's default of 1) specifically so canvas.width/height
  // equaled the WebGL drawing-buffer size, which every position/scale
  // formula in this file used to assume directly. Now that syncSizeToSource
  // and placeEar below key off the closure-scoped currentWidth/currentHeight
  // (the *source's* logical pixel size) instead of canvas.width/height, it's
  // safe to let the drawing buffer be larger than that. Verified against
  // three.js's own WebGLRenderer.setSize source: it derives
  // canvas.width/height = width/height * pixelRatio internally, so passing
  // logical width/height into setSize (below) and letting the renderer
  // derive the (possibly larger) buffer size is exactly the standard
  // three.js high-DPI pattern. options.pixelRatioCap is omitted (defaults to
  // 1) by every index.html debug-harness call site, so this line is a no-op
  // there — behavior changes only for callers that opt in.
  renderer.setPixelRatio(options.pixelRatioCap ?? 1);

  // --- WebGL context-loss handling (Phase 11) -----------------------------
  //
  // Verified directly against node_modules/three/src/renderers/
  // WebGLRenderer.js: THREE.WebGLRenderer already installs its own
  // 'webglcontextlost'/'webglcontextrestored' listeners at construction
  // (before these ones), its own onContextLost already calls
  // event.preventDefault() (permitting browser-side restoration), render()
  // already silently no-ops while lost, and its own onContextRestore already
  // calls initGLContext() to rebuild its internal GPU-resource caches — the
  // next render() call after restore lazily re-uploads geometry/textures
  // from the still-alive JS-side scene graph with no manual re-init needed.
  // So most context losses need zero app code to recover from; what's
  // missing is a signal for the case restoration DOESN'T happen (a real risk
  // on iOS Safari under memory pressure per CLAUDE.md) so the caller can show
  // an error instead of leaving the view silently frozen forever.
  let contextLostTimeoutHandle: ReturnType<typeof setTimeout> | undefined;

  function handleContextLost(): void {
    contextLostTimeoutHandle = setTimeout(() => {
      contextLostTimeoutHandle = undefined;
      options.onContextLost?.();
    }, CONTEXT_LOST_RECOVERY_TIMEOUT_MS);
  }

  function handleContextRestored(): void {
    if (contextLostTimeoutHandle !== undefined) {
      clearTimeout(contextLostTimeoutHandle);
      contextLostTimeoutHandle = undefined;
    }
  }

  canvas.addEventListener('webglcontextlost', handleContextLost);
  canvas.addEventListener('webglcontextrestored', handleContextRestored);

  const scene = new THREE.Scene();

  // Frustum is set in canvas-pixel units (see syncSizeToVideo). top=0,
  // bottom=canvas.height (reversed from the usual top>bottom convention)
  // is what makes world-space Y equal canvas-pixel Y — see
  // createEarInstance's docstring for why that requires a static Y-flip on
  // the loaded model. near/far are a generous, symmetric range around the
  // camera's default position (world origin): MediaPipe's z is a small,
  // uncalibrated depth proxy (roughly faceScale-sized), so there's no real
  // distance to calibrate against — the range just needs to comfortably
  // contain whatever anchor.z * canvasWidth produces.
  const camera = new THREE.OrthographicCamera(
    0,
    canvas.width,
    0,
    canvas.height,
    ORTHOGRAPHIC_NEAR,
    ORTHOGRAPHIC_FAR,
  );

  // MeshStandardMaterial (used by the placeholder earring) renders solid
  // black with no lights in the scene. Real webcam lighting direction is
  // unknowable without environment capture (out of scope) — this is just
  // enough to make the model visible and legible.
  scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.5));
  const keyLight = new THREE.DirectionalLight(0xffffff, 1.5);
  keyLight.position.set(0.5, 1, 1);
  scene.add(keyLight);

  // <video> and <img> expose their intrinsic pixel size under different
  // property names (videoWidth/videoHeight vs. naturalWidth/naturalHeight)
  // — this is the one place that difference matters, everything else below
  // operates purely in canvas.width/height pixel space regardless of source.
  function getSourceDimensions(): { width: number; height: number } {
    if (source instanceof HTMLVideoElement) {
      return { width: source.videoWidth, height: source.videoHeight };
    }
    return { width: source.naturalWidth, height: source.naturalHeight };
  }

  // Logical (CSS-pixel-equivalent) source dimensions — what the
  // OrthographicCamera frustum and placeEar's pixel-space math key off, as
  // opposed to canvas.width/height, which is this same size times
  // renderer.getPixelRatio() once options.pixelRatioCap != 1. Comparing
  // against canvas.width/height directly (as this function used to) would
  // never match again once pixelRatio != 1, since canvas.width becomes
  // width * pixelRatio while `width` here stays the source's logical size —
  // that would silently force a full resize/projection-matrix recompute on
  // every single frame instead of only on real video-resolution changes.
  let currentWidth = 0;
  let currentHeight = 0;

  function syncSizeToSource(): void {
    const { width, height } = getSourceDimensions();
    if (currentWidth === width && currentHeight === height) {
      return;
    }
    currentWidth = width;
    currentHeight = height;
    // Derives canvas.width/height = width/height * renderer.getPixelRatio()
    // internally (verified against three.js's own source) — no longer
    // stamping canvas.width/height manually here, since that would be
    // redundant with (and immediately overwritten by) this call once
    // pixelRatio != 1.
    renderer.setSize(width, height, false);
    camera.right = width;
    camera.bottom = height;
    camera.updateProjectionMatrix();
  }

  // Run once up front too: the caller's precondition (a ready camera stream,
  // or an already-decode()d image) already guarantees valid dimensions, so
  // this avoids briefly building the renderer/camera against the <canvas>
  // element's 300x150 HTML default before the first updateFrame() resize
  // check runs.
  syncSizeToSource();

  // Loaded once (not once per ear) to avoid a duplicate network fetch +
  // duplicate Draco decode; each ear gets its own clone (see
  // createEarInstance). NOTE: this is a placeholder GLB (see
  // scripts/generate-placeholder-earring.mjs) — the real per-product GLB
  // comes from Shopify product media (TryOnOptions.glbUrl) once
  // openTryOn's pipeline is wired up (Phase 5+/12).
  const gltf = await getGltfLoader().loadAsync(glbUrl);
  const rightEar = createEarInstance(gltf.scene, 1);
  const leftEar = createEarInstance(gltf.scene, -1);
  scene.add(rightEar.anchorGroup, leftEar.anchorGroup);

  function updateFrame(frame: TrackingFrame | null): void {
    syncSizeToSource();
    if (frame) {
      placeEar(
        rightEar.anchorGroup,
        frame.right,
        frame.faceScale,
        frame.headRotation,
        currentWidth,
        currentHeight,
      );
      placeEar(
        leftEar.anchorGroup,
        frame.left,
        frame.faceScale,
        frame.headRotation,
        currentWidth,
        currentHeight,
      );
    } else {
      rightEar.anchorGroup.visible = false;
      leftEar.anchorGroup.visible = false;
    }
    renderer.render(scene, camera);
  }

  async function loadModel(nextGlbUrl: string): Promise<void> {
    const nextGltf = await getGltfLoader().loadAsync(nextGlbUrl);
    setEarModel(rightEar, nextGltf.scene);
    setEarModel(leftEar, nextGltf.scene);
  }

  function dispose(): void {
    canvas.removeEventListener('webglcontextlost', handleContextLost);
    canvas.removeEventListener('webglcontextrestored', handleContextRestored);
    if (contextLostTimeoutHandle !== undefined) {
      clearTimeout(contextLostTimeoutHandle);
      contextLostTimeoutHandle = undefined;
    }
    disposeObject3D(scene);
    renderer.dispose();
  }

  return { updateFrame, loadModel, dispose };
}
