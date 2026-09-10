/**
 * 3D rendering: sets up a Three.js scene with a transparent-background
 * WebGLRenderer, loads earring GLB models via GLTFLoader (with
 * DRACOLoader/meshopt for compressed GLB), and positions/rotates the
 * loaded model at the smoothed ear anchor from tracking.ts each frame.
 * The resulting canvas is layered on top of the `<video>` element to
 * composite the AR overlay over the camera feed.
 *
 * Implemented in Phase 3 (anchored placement) and Phase 4 (orientation/hang
 * tuning as the head tilts).
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

export interface EarringScene {
  /** Positions both ear instances from this frame's smoothed data (or hides them on `null`) and renders. */
  updateFrame(frame: TrackingFrame | null): void;
  /** Loads a different GLB and swaps it onto both ears in place — anchorGroup's live position/scale/rotation are untouched, so there's no visual jump. Debug-harness-only for now (v1 has no mid-session product switching, per CLAUDE.md). */
  loadModel(glbUrl: string): Promise<void>;
  /** Releases GPU resources (geometries/materials/renderer). Does not touch the shared GLTFLoader/DRACOLoader singleton. */
  dispose(): void;
}

/**
 * Sets up a Three.js scene layered on top of `video` (via `canvas`,
 * positioned by the caller the same way the Phase 1/2 debug canvas is) and
 * loads `glbUrl` once, instancing it onto both ears (mirrored — see
 * createEarInstance). `glbUrl` is taken as a fully-resolved absolute URL,
 * not resolved through getAssetBaseUrl() here: in production this is an
 * arbitrary Shopify CDN URL (TryOnOptions.glbUrl) with nothing to do with
 * where this widget's own script was loaded from — only the Draco decoder
 * path above is one of this widget's own bundled runtime assets.
 */
export async function createEarringScene(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
  glbUrl: string,
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
  // default. Do NOT call renderer.setPixelRatio(): leaving it at 1 is what
  // makes canvas.width/height equal to the WebGL drawing-buffer size, which
  // every position/scale formula in this file assumes. Higher-DPI
  // sharpness is a real improvement but belongs in the Phase 11
  // performance pass, once those formulas are reworked to account for it.
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

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

  function syncSizeToVideo(): void {
    const { videoWidth, videoHeight } = video;
    if (canvas.width === videoWidth && canvas.height === videoHeight) {
      return;
    }
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    renderer.setSize(videoWidth, videoHeight, false);
    camera.right = videoWidth;
    camera.bottom = videoHeight;
    camera.updateProjectionMatrix();
  }

  // Run once up front too: startCamera() (the caller's precondition)
  // already guarantees valid videoWidth/videoHeight, so this avoids
  // briefly building the renderer/camera against the <canvas> element's
  // 300x150 HTML default before the first updateFrame() resize check runs.
  syncSizeToVideo();

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
    syncSizeToVideo();
    if (frame) {
      placeEar(
        rightEar.anchorGroup,
        frame.right,
        frame.faceScale,
        frame.headRotation,
        canvas.width,
        canvas.height,
      );
      placeEar(
        leftEar.anchorGroup,
        frame.left,
        frame.faceScale,
        frame.headRotation,
        canvas.width,
        canvas.height,
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
    disposeObject3D(scene);
    renderer.dispose();
  }

  return { updateFrame, loadModel, dispose };
}
