/**
 * Choose a Model mode (Phase 10): a small curated set of bundled model
 * photos, for shoppers who don't want to use their own camera or photo.
 * Reuses the entire static-photo pipeline built for Upload Photo (Phase 9)
 * unchanged — MediaPipe IMAGE-mode detection, tracking.ts's ear-anchor
 * math, render.ts's Three.js scene, capture.ts's compositeEarringOntoImage,
 * and the Phases 6-7 review/share screen. The only real difference is where
 * the source image comes from: a picker grid of bundled photos (see
 * modelPhotos.ts) instead of a user-uploaded file, so there's no file-type
 * check and no object-URL lifecycle to manage.
 *
 * Structurally mirrors uploadPhoto.ts: a `cancelled` flag checked after
 * every `await`, and a returned `dispose()` safe to call from any state.
 */

import type { TryOnOptions } from '../main.ts';
import type { FaceLandmarker } from '@mediapipe/tasks-vision';
import { createImageFaceLandmarker, detectEarAnchorsForImage } from '../tracking.ts';
import { createEarringScene, type EarringScene } from '../render.ts';
import { detectDeviceTier } from '../deviceCapabilities.ts';
import { compositeEarringOntoImage } from './capture.ts';
import { createReviewScreen, type ReviewScreen } from './reviewScreen.ts';
import { createProcessingState, createErrorState } from './staticPhotoStates.ts';
import { getAssetBaseUrl } from '../assets.ts';
import { CURATED_MODEL_PHOTOS, type ModelPhoto } from './modelPhotos.ts';

export interface ChooseModelHandlers {
  onBackToChooser(): void;
}

function resolveModelPhotoUrl(filename: string): string {
  return `${getAssetBaseUrl()}model-photos/${filename}`;
}

function createPickerState(
  onModelSelected: (photo: ModelPhoto) => void,
  onBackToChooser: () => void,
): HTMLElement {
  const wrap = document.createElement('div');
  wrap.className = 'model-picker';

  const heading = document.createElement('h2');
  heading.className = 'model-picker-title';
  heading.textContent = 'Choose a model';

  const grid = document.createElement('div');
  grid.className = 'model-grid';

  for (const photo of CURATED_MODEL_PHOTOS) {
    const tile = document.createElement('button');
    tile.type = 'button';
    tile.className = 'model-tile';

    const thumb = document.createElement('img');
    thumb.className = 'model-tile-thumb';
    thumb.src = resolveModelPhotoUrl(photo.filename);
    thumb.alt = photo.alt;

    tile.append(thumb);
    tile.addEventListener('click', () => {
      onModelSelected(photo);
    });
    grid.append(tile);
  }

  const backButton = document.createElement('button');
  backButton.type = 'button';
  backButton.className = 'button button--secondary';
  backButton.textContent = 'Back to options';
  backButton.addEventListener('click', onBackToChooser);

  // No privacy line here, unlike the mode chooser and Upload Photo: its
  // claim ("nothing is uploaded, stays on your device") is about the
  // user's own data, and this mode never touches any user data at all —
  // showing it would be a category error, not just redundant.
  wrap.append(heading, grid, backButton);
  return wrap;
}

/**
 * Loads a bundled model photo by URL into a ready-to-draw <img>. Simpler
 * than uploadPhoto.ts's loadImageFromFile — no object-URL create/revoke,
 * since it's a static asset URL, not a user-supplied Blob. crossOrigin is
 * set because getAssetBaseUrl() can resolve to a different origin than the
 * embedding page once a CDN is wired in (Phase 12+); an un-flagged
 * cross-origin <img> would taint the canvas compositeEarringOntoImage reads
 * from, silently breaking toBlob() — same risk class already flagged for
 * GLB textures in Phase 6's build-progress.md entry. Harmless no-op today,
 * since the dev build is same-origin.
 */
function loadImageFromUrl(url: string): Promise<HTMLImageElement> {
  const image = new Image();
  image.crossOrigin = 'anonymous';
  image.src = url;
  return image.decode().then(() => image);
}

/**
 * Renders into `container`, moving picker -> processing -> review (or
 * picker -> processing -> error, with "choose a different model" looping
 * back to picker). Returns a cleanup function safe to call from any state.
 */
export function renderChooseModel(
  container: HTMLElement,
  options: Pick<TryOnOptions, 'glbUrl' | 'name'>,
  handlers: ChooseModelHandlers,
): () => void {
  let cancelled = false;
  let reviewScreen: ReviewScreen | undefined;
  // Lazily created on first selection and reused across the rest of this
  // session (browsing several curated photos is the common case here, even
  // more so than upload's retries) — closed in dispose(). Independent from
  // uploadPhoto.ts's own instance: the two modes are mutually exclusive UI
  // states, never active at the same time.
  let imageLandmarker: FaceLandmarker | undefined;

  function showPicker(): void {
    container.replaceChildren(createPickerState(handleModelSelected, handlers.onBackToChooser));
  }

  function showError(message: string, tryAgainLabel?: string): void {
    container.replaceChildren(
      createErrorState(message, showPicker, handlers.onBackToChooser, tryAgainLabel),
    );
  }

  async function handleModelSelected(photo: ModelPhoto): Promise<void> {
    container.replaceChildren(createProcessingState());

    let image: HTMLImageElement;
    try {
      image = await loadImageFromUrl(resolveModelPhotoUrl(photo.filename));
    } catch (error) {
      if (cancelled) {
        return;
      }
      console.error('[CollectiblissTryOn] could not load the selected model photo', error);
      showError(
        "We couldn't load that photo. Please choose a different one.",
        'Choose a different model',
      );
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
        showError(
          'Something went wrong getting ready to place the earring. Please try again.',
          'Choose a different model',
        );
        return;
      }
    }
    if (cancelled) {
      return;
    }

    const frame = detectEarAnchorsForImage(imageLandmarker, image);
    if (!frame) {
      showError(
        "We couldn't find a face in that photo. Please choose a different model.",
        'Choose a different model',
      );
      return;
    }

    // Never appended to the DOM — a one-shot off-screen WebGL render
    // target, read back via compositeEarringOntoImage below and then
    // discarded.
    const earringCanvas = document.createElement('canvas');
    // No detection-cadence change (this is a one-shot detect(), not a RAF
    // loop) — only caps the WebGL drawing-buffer resolution, same reasoning
    // as liveTryOn.ts's use of this. See deviceCapabilities.ts.
    const { pixelRatioCap } = detectDeviceTier();
    let scene: EarringScene;
    try {
      scene = await createEarringScene(earringCanvas, image, options.glbUrl, { pixelRatioCap });
    } catch (error) {
      if (cancelled) {
        return;
      }
      console.error('[CollectiblissTryOn] failed to load earring model', error);
      showError(
        'Something went wrong placing the earring. Please try again.',
        'Choose a different model',
      );
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
        showError(
          'Something went wrong placing the earring. Please try again.',
          'Choose a different model',
        );
      }
      return;
    } finally {
      // One scene per selection: a fresh photo (or a retry after a
      // failure) always starts a brand-new scene, so there's nothing to
      // keep this one alive for — same accepted v1 tradeoff as
      // uploadPhoto.ts (GLB reload cost on retry, absorbed by the
      // browser's HTTP cache; only Draco/meshopt decode repeats).
      scene.dispose();
    }
    if (cancelled) {
      return;
    }

    reviewScreen = createReviewScreen(blob, options.name, {
      onRetake: showPicker,
      retakeLabel: 'Choose a different model',
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
