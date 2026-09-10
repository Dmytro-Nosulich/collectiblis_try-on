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
import { startTracking, type TrackingFrame } from '../tracking.ts';
import { createEarringScene, type EarringScene } from '../render.ts';

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

function createErrorState(message: string, onBackToChooser: () => void): HTMLElement {
  const wrap = document.createElement('div');
  wrap.className = 'state-centered';

  const messageEl = document.createElement('p');
  messageEl.className = 'error-message';
  messageEl.textContent = message;

  const actions = document.createElement('div');
  actions.className = 'error-actions';

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
}

function createLiveViewElements(): LiveViewElements {
  const root = document.createElement('div');
  root.className = 'live-view';

  const mirrorWrapper = document.createElement('div');
  mirrorWrapper.className = 'mirror-wrapper';

  const video = document.createElement('video');
  video.className = 'live-video';
  video.playsInline = true;
  video.muted = true;

  const canvas = document.createElement('canvas');
  canvas.className = 'live-canvas';

  mirrorWrapper.append(video, canvas);
  root.append(mirrorWrapper);

  return { root, video, canvas };
}

/**
 * Renders into `container`, moving loading -> live (or loading -> error).
 * Returns a cleanup function that releases whatever pipeline resources are
 * currently live — safe to call from any state, including mid-loading
 * (e.g. the modal was closed before the camera/tracking/scene resolved).
 */
export function renderLiveTryOn(
  container: HTMLElement,
  options: Pick<TryOnOptions, 'glbUrl'>,
  handlers: LiveTryOnHandlers,
): () => void {
  let cameraSession: CameraSession | undefined;
  let trackingCleanup: (() => void) | undefined;
  let earringScene: EarringScene | undefined;
  let cancelled = false;

  function handleFrame(frame: TrackingFrame | null): void {
    earringScene?.updateFrame(frame);
  }

  async function start(): Promise<void> {
    const { root, video, canvas } = createLiveViewElements();

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

    // No debug canvas passed to startTracking — the visible canvas here is
    // render.ts's WebGL surface, and a canvas can only host one kind of
    // rendering context. Runs in parallel with createEarringScene since
    // neither depends on the other, only on the now-ready video.
    const [cleanup, scene] = await Promise.all([
      startTracking(video, null, handleFrame),
      createEarringScene(canvas, video, options.glbUrl),
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

  container.replaceChildren(createLoadingState());
  start();

  return function dispose(): void {
    cancelled = true;
    trackingCleanup?.();
    earringScene?.dispose();
    if (cameraSession) {
      stopCamera(cameraSession);
    }
  };
}
