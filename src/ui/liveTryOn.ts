/**
 * Live Try-On: wires the Phases 1-4 pipeline (camera.ts + tracking.ts +
 * render.ts) into loading/live/error states inside the modal shell.
 *
 * Mirroring: the video+canvas wrapper gets `transform: scaleX(-1)` in CSS
 * (see styles.css's .mirror-wrapper) — this flips the already-pixel-aligned
 * composite as a single post-render step, so no position/anchor math in
 * tracking.ts/render.ts needs to change, and the two stay aligned
 * automatically. This is the resolution to tracking.ts's Phase 2 comment
 * flagging mirrored display as something to "reconcile" later: nothing
 * there needed to change. `left`/`right` in TrackingFrame stay the
 * subject's actual anatomical ears throughout — only the on-screen
 * *position* is mirrored, which happens after render.ts has already placed
 * both ears correctly relative to each other and to the (unmirrored) video
 * pixels.
 */

import type { TryOnOptions } from '../main.ts';
import { startCamera, stopCamera, type CameraSession } from '../camera.ts';
import {
  startTracking,
  detectionCadenceTuning,
  DEFAULT_DETECTION_CADENCE_TUNING,
  type TrackingFrame,
} from '../tracking.ts';
import { createEarringScene, type EarringScene } from '../render.ts';
import { detectDeviceTier } from '../deviceCapabilities.ts';
import { captureMirroredComposite } from './capture.ts';
import { renderCountdown } from './countdownOverlay.ts';
import { createReviewScreen, type ReviewScreen } from './reviewScreen.ts';

/** Detection cadence dropped to every-other-frame on detected low-end devices only — see deviceCapabilities.ts and tracking.ts's detectionCadenceTuning docs. */
const LOW_END_DETECTION_INTERVAL_FRAMES = 2;

export interface LiveTryOnHandlers {
  onBackToChooser(): void;
}

const STATUS_GETTING_CAMERA_READY = 'Getting your camera ready...';

function cameraErrorMessage(error: unknown): string {
  if (error instanceof DOMException) {
    if (error.name === 'NotAllowedError') {
      return 'Camera access was denied. You can allow it in your browser settings, or try one of the options below.';
    }
    if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
      return "We couldn't find a camera on this device. Try one of the options below instead.";
    }
  }
  return 'Something went wrong starting your camera. Try one of the options below instead.';
}

function createLoadingState(): HTMLElement {
  const wrap = document.createElement('div');
  wrap.className = 'state-centered';

  const spinner = document.createElement('div');
  spinner.className = 'spinner';

  const status = document.createElement('p');
  status.className = 'state-status';
  status.textContent = STATUS_GETTING_CAMERA_READY;

  wrap.append(spinner, status);
  return wrap;
}

/**
 * `onRetry`, when given, prepends a primary "Try again" button ahead of the
 * two existing mode-switch buttons — used by Phase 11's WebGL-context-lost
 * recovery (see handleContextLost), which has something worth retrying
 * unlike a camera-permission/no-camera error (that call site passes nothing
 * for this param, so its behavior is unchanged).
 */
function createErrorState(
  message: string,
  onBackToChooser: () => void,
  onRetry?: () => void,
): HTMLElement {
  const wrap = document.createElement('div');
  wrap.className = 'state-centered';

  const messageEl = document.createElement('p');
  messageEl.className = 'error-message';
  messageEl.textContent = message;

  const actions = document.createElement('div');
  actions.className = 'error-actions';

  if (onRetry) {
    const retryButton = document.createElement('button');
    retryButton.type = 'button';
    retryButton.className = 'button button--primary';
    retryButton.textContent = 'Try again';
    retryButton.addEventListener('click', onRetry);
    actions.append(retryButton);
  }

  for (const label of ['Try uploading a photo instead', 'Choose a model photo instead']) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'button button--secondary';
    button.textContent = label;
    button.addEventListener('click', onBackToChooser);
    actions.append(button);
  }

  wrap.append(messageEl, actions);
  return wrap;
}

interface LiveViewElements {
  root: HTMLElement;
  video: HTMLVideoElement;
  canvas: HTMLCanvasElement;
  liveStage: HTMLElement;
  captureButton: HTMLButtonElement;
  compareToggleButton: HTMLButtonElement;
  swapButton: HTMLButtonElement;
}

function createLiveViewElements(): LiveViewElements {
  const root = document.createElement('div');
  root.className = 'live-view';

  // .live-stage is a plain (non-mirrored) positioning context, separate
  // from .mirror-wrapper, specifically so content mounted here (the
  // countdown overlay, and the compare-view divider below) doesn't inherit
  // mirrorWrapper's scaleX(-1) and render backwards/asymmetrically.
  const liveStage = document.createElement('div');
  liveStage.className = 'live-stage';

  const mirrorWrapper = document.createElement('div');
  mirrorWrapper.className = 'mirror-wrapper';

  const video = document.createElement('video');
  video.className = 'live-video';
  video.playsInline = true;
  video.muted = true;

  const canvas = document.createElement('canvas');
  canvas.className = 'live-canvas';

  mirrorWrapper.append(video, canvas);

  // Compare view (Phase 8): one continuous video/canvas — the same pair
  // used for normal Live Try-On — with a CSS clip-path on `canvas` masking
  // off one half, plus this divider line drawn over the seam. See
  // styles.css's .is-comparing/.is-ar-left rules for the clip-path values
  // (and the mirroring-math comment there before touching them) and
  // render.ts's docs for why rendering both ears unconditionally, every
  // frame, regardless of compare mode, is intentional and cheap — compare
  // mode only ever changes which half of the already-rendered canvas is
  // visible, never what gets rendered. The divider sits at the exact
  // center, a fixed point of mirrorWrapper's scaleX(-1) reflection, so
  // unlike the canvas clip-path it needs no mirror-side reasoning; it's a
  // liveStage sibling of mirrorWrapper (not a child) purely to match the
  // countdown overlay's existing precedent.
  const compareDivider = document.createElement('div');
  compareDivider.className = 'compare-divider';

  liveStage.append(mirrorWrapper, compareDivider);

  const captureButton = document.createElement('button');
  captureButton.type = 'button';
  captureButton.className = 'button button--primary capture-button';
  captureButton.textContent = 'Capture';

  const compareToggleButton = document.createElement('button');
  compareToggleButton.type = 'button';
  compareToggleButton.className = 'button button--secondary';
  compareToggleButton.textContent = 'Compare view';
  compareToggleButton.setAttribute('aria-pressed', 'false');

  const swapButton = document.createElement('button');
  swapButton.type = 'button';
  swapButton.className = 'button button--secondary';
  swapButton.textContent = 'Swap';
  // Only meaningful once compare mode is on — see handleCompareToggle.
  swapButton.hidden = true;

  const liveControls = document.createElement('div');
  liveControls.className = 'live-controls';
  liveControls.append(captureButton, compareToggleButton, swapButton);

  root.append(liveStage, liveControls);

  return {
    root,
    video,
    canvas,
    liveStage,
    captureButton,
    compareToggleButton,
    swapButton,
  };
}

/**
 * Renders into `container`, moving loading -> live (or loading -> error).
 * Returns a cleanup function that releases whatever pipeline resources are
 * currently live — safe to call from any state, including mid-loading
 * (e.g. the modal was closed before the camera/tracking/scene resolved).
 */
export function renderLiveTryOn(
  container: HTMLElement,
  options: Pick<TryOnOptions, 'glbUrl' | 'name'>,
  handlers: LiveTryOnHandlers,
): () => void {
  let cameraSession: CameraSession | undefined;
  let trackingCleanup: (() => void) | undefined;
  let earringScene: EarringScene | undefined;
  let cancelled = false;
  let cancelCountdown: (() => void) | undefined;
  let reviewScreen: ReviewScreen | undefined;

  function handleFrame(frame: TrackingFrame | null): void {
    earringScene?.updateFrame(frame);
  }

  /**
   * Stops/disposes whatever pipeline resources are currently live and clears
   * their refs, so it's safe to call more than once (real dispose(), or a
   * WebGL-context-lost retry that's about to call start() again from
   * scratch) without double-stopping an already-stopped camera track or
   * double-closing an already-closed landmarker.
   */
  function teardownPipeline(): void {
    cancelCountdown?.();
    cancelCountdown = undefined;
    reviewScreen?.dispose();
    reviewScreen = undefined;
    trackingCleanup?.();
    trackingCleanup = undefined;
    earringScene?.dispose();
    earringScene = undefined;
    if (cameraSession) {
      stopCamera(cameraSession);
      cameraSession = undefined;
    }
  }

  /**
   * Fires only if render.ts's WebGL-context-lost recovery timeout elapses
   * without the browser restoring the context on its own (see render.ts's
   * context-loss handling for why most losses need no recovery code at
   * all). Tears down the now-unrecoverable pipeline and offers a real retry
   * — distinct from the camera-permission error state below, which has
   * nothing to retry.
   */
  function handleContextLost(): void {
    if (cancelled) {
      return;
    }
    teardownPipeline();
    container.replaceChildren(
      createErrorState(
        'Something went wrong with the camera view. Please try again.',
        handlers.onBackToChooser,
        () => start(),
      ),
    );
  }

  async function start(): Promise<void> {
    container.replaceChildren(createLoadingState());

    const { root, video, canvas, liveStage, captureButton, compareToggleButton, swapButton } =
      createLiveViewElements();

    function handleCompareToggle(): void {
      const enabled = root.classList.toggle('is-comparing');
      compareToggleButton.classList.toggle('button--primary', enabled);
      compareToggleButton.classList.toggle('button--secondary', !enabled);
      compareToggleButton.setAttribute('aria-pressed', String(enabled));
      swapButton.hidden = !enabled;
    }

    function handleSwap(): void {
      root.classList.toggle('is-ar-left');
    }

    compareToggleButton.addEventListener('click', handleCompareToggle);
    swapButton.addEventListener('click', handleSwap);

    function handleRetake(): void {
      reviewScreen?.dispose();
      reviewScreen = undefined;
      root.classList.remove('is-reviewing');
      captureButton.hidden = false;
    }

    function startCountdownAndCapture(): void {
      captureButton.hidden = true;
      cancelCountdown = renderCountdown(liveStage, () => {
        cancelCountdown = undefined;
        captureMirroredComposite(video, canvas).then(
          (blob) => {
            // Covers the race where the modal was closed in the gap
            // between the countdown completing and this promise
            // resolving — renderCountdown's own cancel() only stops a
            // *pending* timer, it can't un-fire a callback that already
            // ran.
            if (cancelled) {
              return;
            }
            reviewScreen = createReviewScreen(blob, options.name, { onRetake: handleRetake });
            root.classList.add('is-reviewing');
            root.append(reviewScreen.root);
          },
          (error: unknown) => {
            if (cancelled) {
              return;
            }
            console.error('[CollectiblissTryOn] snapshot capture failed', error);
            captureButton.hidden = false;
          },
        );
      });
    }

    captureButton.addEventListener('click', startCountdownAndCapture);

    let session: CameraSession;
    try {
      session = await startCamera(video);
    } catch (error) {
      if (!cancelled) {
        container.replaceChildren(
          createErrorState(cameraErrorMessage(error), handlers.onBackToChooser),
        );
      }
      return;
    }
    if (cancelled) {
      stopCamera(session);
      return;
    }
    cameraSession = session;

    // Device tier recomputed fresh on every start() call (including a
    // context-lost retry) — cheap (two navigator property reads), so no
    // need to cache it across the mode's lifetime. Low-end devices drop
    // detection cadence to every-other-frame (tracking.ts's
    // detectionCadenceTuning) and cap the WebGL drawing-buffer resolution
    // (render.ts's pixelRatioCap); see deviceCapabilities.ts for why
    // hardwareConcurrency (not devicePixelRatio) is the tier signal.
    const tier = detectDeviceTier();
    detectionCadenceTuning.intervalFrames = tier.isLowEnd
      ? LOW_END_DETECTION_INTERVAL_FRAMES
      : DEFAULT_DETECTION_CADENCE_TUNING.intervalFrames;

    // No debug canvas passed to startTracking — the visible canvas here is
    // render.ts's WebGL surface, and a canvas can only host one kind of
    // rendering context. Runs in parallel with createEarringScene since
    // neither depends on the other, only on the now-ready video.
    const [cleanup, scene] = await Promise.all([
      startTracking(video, null, handleFrame),
      createEarringScene(canvas, video, options.glbUrl, {
        pixelRatioCap: tier.pixelRatioCap,
        onContextLost: handleContextLost,
      }),
    ]);

    if (cancelled) {
      cleanup();
      scene.dispose();
      stopCamera(session);
      return;
    }
    trackingCleanup = cleanup;
    earringScene = scene;
    container.replaceChildren(root);
  }

  start();

  return function dispose(): void {
    cancelled = true;
    teardownPipeline();
  };
}
