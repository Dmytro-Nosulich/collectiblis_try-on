/**
 * Face/landmark tracking: loads MediaPipe's FaceLandmarker
 * (`@mediapipe/tasks-vision`) and runs detection against video/image
 * frames, approximates a left/right earlobe anchor point from the 478
 * returned landmarks (there is no dedicated ear landmark), and smooths
 * that point with a One Euro Filter (`1eurofilter`) to remove per-frame
 * jitter. See CLAUDE.md "Known technical challenges".
 *
 * Phase 1 (this file, so far): raw landmark detection + debug visualization
 * only. Ear anchor approximation + smoothing land in Phase 2.
 */

import { DrawingUtils, FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { getAssetBaseUrl } from './assets.ts';

const WASM_BASE_PATH_SEGMENT = 'mediapipe/wasm';
const MODEL_ASSET_PATH_SEGMENT = 'mediapipe/models/face_landmarker.task';
const FPS_LOG_INTERVAL_FRAMES = 30;

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
 */
export async function startTracking(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
): Promise<() => void> {
  const faceLandmarker = await createFaceLandmarker();
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('[CollectiblissTryOn] 2D canvas context unavailable');
  }
  const drawingUtils = new DrawingUtils(ctx);

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
