/**
 * The post-capture review screen: shows the captured photo with a Retake
 * option and the real share buttons (Phase 7 — see share.ts for the actual
 * Web Share / download / Pinterest / Facebook logic).
 *
 * A passive factory, not a render*() that owns a container — liveTryOn.ts
 * needs to append this into its already-live root and later remove it on
 * retake, rather than wholesale-replacing a container the way
 * renderModeChooser/renderLiveTryOn do.
 */

import { createShareSection } from './share.ts';

export interface ReviewScreenHandlers {
  onRetake(): void;
  /** Defaults to 'Retake'. Upload Photo (Phase 9) passes 'Choose a different photo' instead, since "retake" doesn't describe re-uploading. */
  retakeLabel?: string;
}

export interface ReviewScreen {
  root: HTMLElement;
  /** Revokes the photo's object URL and removes the DOM node. Safe to call more than once. */
  dispose(): void;
}

export function createReviewScreen(
  photoBlob: Blob,
  productName: string,
  handlers: ReviewScreenHandlers,
): ReviewScreen {
  const objectUrl = URL.createObjectURL(photoBlob);

  const root = document.createElement('div');
  root.className = 'review-screen';

  const photo = document.createElement('img');
  photo.className = 'review-photo';
  photo.src = objectUrl;
  photo.alt = 'Your captured photo';

  const actions = document.createElement('div');
  actions.className = 'review-actions';

  const retakeButton = document.createElement('button');
  retakeButton.type = 'button';
  retakeButton.className = 'button button--secondary';
  retakeButton.textContent = handlers.retakeLabel ?? 'Retake';
  retakeButton.addEventListener('click', () => handlers.onRetake());

  actions.append(retakeButton);
  root.append(photo, actions, createShareSection(photoBlob, productName));

  let disposed = false;
  function dispose(): void {
    if (disposed) {
      return;
    }
    disposed = true;
    URL.revokeObjectURL(objectUrl);
    root.remove();
  }

  return { root, dispose };
}
