/**
 * Upload Photo mode (Phase 9): a file input / drag-and-drop alternative to
 * Live Try-On for shoppers without (or who don't want to use) a camera.
 * Runs MediaPipe's FaceLandmarker once in IMAGE mode against the uploaded
 * photo (no live video, no One Euro smoothing — a single static detection
 * has no per-frame jitter to smooth against) and reuses tracking.ts's
 * ear-anchor math and render.ts's Three.js scene to place the earring
 * once, then reuses the Phases 6-7 review/share screen for the result —
 * skipping the countdown, since there's no live pose moment to count down
 * to.
 *
 * Structurally mirrors liveTryOn.ts: a `cancelled` flag checked after every
 * `await`, and a returned `dispose()` safe to call from any state.
 */

import type { TryOnOptions } from '../main.ts';
import type { FaceLandmarker } from '@mediapipe/tasks-vision';
import { createImageFaceLandmarker, detectEarAnchorsForImage } from '../tracking.ts';
import { createEarringScene, type EarringScene } from '../render.ts';
import { compositeEarringOntoImage } from './capture.ts';
import { createReviewScreen, type ReviewScreen } from './reviewScreen.ts';

export interface UploadPhotoHandlers {
  onBackToChooser(): void;
}

const STATUS_PLACING_EARRING = 'Placing the earring...';

// Same copy/class as modeChooser.ts's own privacy line — duplicated rather
// than exported/shared, since it's one string constant and this file
// otherwise has no dependency on modeChooser.ts.
const PRIVACY_LINE = 'Nothing is uploaded — this stays on your device.';

function createProcessingState(): HTMLElement {
  const wrap = document.createElement('div');
  wrap.className = 'state-centered';

  const spinner = document.createElement('div');
  spinner.className = 'spinner';

  const status = document.createElement('p');
  status.className = 'state-status';
  status.textContent = STATUS_PLACING_EARRING;

  wrap.append(spinner, status);
  return wrap;
}

/** Shared by every failure case (bad file, undecodable file, no face detected, model load failure) — only the message differs. */
function createErrorState(
  message: string,
  onTryAgain: () => void,
  onBackToChooser: () => void,
): HTMLElement {
  const wrap = document.createElement('div');
  wrap.className = 'state-centered';

  const messageEl = document.createElement('p');
  messageEl.className = 'error-message';
  messageEl.textContent = message;

  const actions = document.createElement('div');
  actions.className = 'error-actions';

  const tryAgainButton = document.createElement('button');
  tryAgainButton.type = 'button';
  tryAgainButton.className = 'button button--primary';
  tryAgainButton.textContent = 'Try a different photo';
  tryAgainButton.addEventListener('click', onTryAgain);

  const backButton = document.createElement('button');
  backButton.type = 'button';
  backButton.className = 'button button--secondary';
  backButton.textContent = 'Back to options';
  backButton.addEventListener('click', onBackToChooser);

  actions.append(tryAgainButton, backButton);
  wrap.append(messageEl, actions);
  return wrap;
}

function createPickerState(
  onFileSelected: (file: File) => void,
  onBackToChooser: () => void,
): HTMLElement {
  const wrap = document.createElement('div');
  wrap.className = 'upload-picker';

  const heading = document.createElement('h2');
  heading.className = 'upload-picker-title';
  heading.textContent = 'Upload a photo';

  const dropzone = document.createElement('label');
  dropzone.className = 'upload-dropzone';

  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/*';
  input.className = 'upload-dropzone-input';
  input.addEventListener('change', () => {
    const file = input.files?.[0];
    if (file) {
      onFileSelected(file);
    }
  });

  const dropzoneText = document.createElement('p');
  dropzoneText.textContent = 'Click to choose a photo, or drag one here';

  const dropzoneHint = document.createElement('p');
  dropzoneHint.className = 'upload-dropzone-hint';
  dropzoneHint.textContent = 'A clear photo with your face and ear visible works best.';

  dropzone.append(input, dropzoneText, dropzoneHint);

  // dragDepth (rather than toggling the class directly on every enter/leave)
  // avoids flicker as the pointer crosses the dropzone's own child elements
  // (the hidden input, the two <p>s) — dragenter/dragleave bubble up from
  // whichever child is directly under the pointer, so a naive toggle would
  // rapidly add/remove the class while dragging across them.
  let dragDepth = 0;
  dropzone.addEventListener('dragenter', (event) => {
    event.preventDefault();
    dragDepth += 1;
    dropzone.classList.add('upload-dropzone--drag-over');
  });
  dropzone.addEventListener('dragover', (event) => {
    event.preventDefault();
  });
  dropzone.addEventListener('dragleave', () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) {
      dropzone.classList.remove('upload-dropzone--drag-over');
    }
  });
  dropzone.addEventListener('drop', (event) => {
    event.preventDefault();
    dragDepth = 0;
    dropzone.classList.remove('upload-dropzone--drag-over');
    const file = event.dataTransfer?.files[0];
    if (file) {
      onFileSelected(file);
    }
  });

  const privacyLine = document.createElement('p');
  privacyLine.className = 'privacy-line';
  privacyLine.textContent = PRIVACY_LINE;

  const backButton = document.createElement('button');
  backButton.type = 'button';
  backButton.className = 'button button--secondary';
  backButton.textContent = 'Back to options';
  backButton.addEventListener('click', onBackToChooser);

  wrap.append(heading, dropzone, privacyLine, backButton);
  return wrap;
}

/**
 * Decodes `file` into a ready-to-draw <img>. Uses decode() rather than the
 * load/error events: it resolves only once the bitmap is fully decoded and
 * safe for drawImage()/MediaPipe's detect() (more robust than `onload`,
 * which some engines have historically fired slightly early for large
 * images), and rejects cleanly on a corrupt/undecodable file instead of
 * needing separate error wiring. The object URL only needs to live long
 * enough for decode() to finish, so it's revoked immediately after, in both
 * the resolve and reject path — a materially shorter lifetime than
 * reviewScreen.ts's own blob URL, which must stay alive for as long as its
 * <img> is displayed.
 */
function loadImageFromFile(file: File): Promise<HTMLImageElement> {
  const objectUrl = URL.createObjectURL(file);
  const image = new Image();
  image.src = objectUrl;
  return image.decode().then(
    () => {
      URL.revokeObjectURL(objectUrl);
      return image;
    },
    (error: unknown) => {
      URL.revokeObjectURL(objectUrl);
      return Promise.reject(error);
    },
  );
}

/**
 * Renders into `container`, moving picker -> processing -> review (or
 * picker -> processing -> error, with "try a different photo" looping back
 * to picker). Returns a cleanup function safe to call from any state.
 */
export function renderUploadPhoto(
  container: HTMLElement,
  options: Pick<TryOnOptions, 'glbUrl' | 'name'>,
  handlers: UploadPhotoHandlers,
): () => void {
  let cancelled = false;
  let reviewScreen: ReviewScreen | undefined;
  // Lazily created on first upload and reused across retries within this
  // session — avoids repaying MediaPipe's WASM/model init cost on every
  // retry — and closed in dispose().
  let imageLandmarker: FaceLandmarker | undefined;

  function showPicker(): void {
    container.replaceChildren(createPickerState(handleFileSelected, handlers.onBackToChooser));
  }

  function showError(message: string): void {
    container.replaceChildren(createErrorState(message, showPicker, handlers.onBackToChooser));
  }

  async function handleFileSelected(file: File): Promise<void> {
    if (!file.type.startsWith('image/')) {
      showError("That file doesn't look like a photo. Please choose an image file.");
      return;
    }

    container.replaceChildren(createProcessingState());

    let image: HTMLImageElement;
    try {
      image = await loadImageFromFile(file);
    } catch (error) {
      if (cancelled) {
        return;
      }
      console.error('[CollectiblissTryOn] could not decode the selected photo', error);
      showError("We couldn't read that photo. Please try a different one.");
      return;
    }
    if (cancelled) {
      return;
    }

    if (!imageLandmarker) {
      try {
        imageLandmarker = await createImageFaceLandmarker();
      } catch (error) {
        if (cancelled) {
          return;
        }
        console.error('[CollectiblissTryOn] failed to initialize photo face detection', error);
        showError('Something went wrong getting ready to place the earring. Please try again.');
        return;
      }
    }
    if (cancelled) {
      return;
    }

    const frame = detectEarAnchorsForImage(imageLandmarker, image);
    if (!frame) {
      showError(
        "We couldn't find a face in that photo. Please try a different photo where your face and ear are clearly visible.",
      );
      return;
    }

    // Never appended to the DOM — a one-shot off-screen WebGL render
    // target, read back via compositeEarringOntoImage below and then
    // discarded.
    const earringCanvas = document.createElement('canvas');
    let scene: EarringScene;
    try {
      scene = await createEarringScene(earringCanvas, image, options.glbUrl);
    } catch (error) {
      if (cancelled) {
        return;
      }
      console.error('[CollectiblissTryOn] failed to load earring model', error);
      showError('Something went wrong placing the earring. Please try again.');
      return;
    }
    if (cancelled) {
      scene.dispose();
      return;
    }

    // One updateFrame() call renders the earring once — no live-loop, no
    // smoothing, matching build-plan.md's "composite the earring onto the
    // static image once."
    scene.updateFrame(frame);
    let blob: Blob;
    try {
      blob = await compositeEarringOntoImage(image, earringCanvas);
    } catch (error) {
      if (!cancelled) {
        console.error('[CollectiblissTryOn] failed to composite the final photo', error);
        showError('Something went wrong placing the earring. Please try again.');
      }
      return;
    } finally {
      // One scene per upload attempt: a fresh photo (or a retry after a
      // failure) always starts a brand-new scene, so there's nothing to
      // keep this one alive for. GLB reload cost on retry is an accepted
      // v1 tradeoff (the browser's HTTP cache absorbs the network fetch;
      // only Draco/meshopt decode repeats) — Phase 11 is the dedicated
      // perf pass, not this one.
      scene.dispose();
    }
    if (cancelled) {
      return;
    }

    reviewScreen = createReviewScreen(blob, options.name, {
      onRetake: showPicker,
      retakeLabel: 'Choose a different photo',
    });
    container.replaceChildren(reviewScreen.root);
  }

  showPicker();

  return function dispose(): void {
    cancelled = true;
    reviewScreen?.dispose();
    reviewScreen = undefined;
    imageLandmarker?.close();
    imageLandmarker = undefined;
  };
}
