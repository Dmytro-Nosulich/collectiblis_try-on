/**
 * The post-capture review screen: shows the captured photo with a Retake
 * option and a placeholder for real share buttons (Phase 7).
 *
 * A passive factory, not a render*() that owns a container — liveTryOn.ts
 * needs to append this into its already-live root and later remove it on
 * retake, rather than wholesale-replacing a container the way
 * renderModeChooser/renderLiveTryOn do.
 */

export interface ReviewScreenHandlers {
  onRetake(): void;
}

export interface ReviewScreen {
  root: HTMLElement;
  /** Revokes the photo's object URL and removes the DOM node. Safe to call more than once. */
  dispose(): void;
}

/**
 * Phase 7's extension point: replace this function's body with the real
 * Web Share / download / Pinterest / Facebook buttons described in
 * build-plan.md's Phase 7 prompt (they'll need the photoBlob passed in via
 * createReviewScreen's closure). Kept as a named function rather than bare
 * markup inlined into createReviewScreen, so Phase 7 has one obvious place
 * to extend without touching liveTryOn.ts's state machine.
 */
function createSharePlaceholder(): HTMLElement {
  const placeholder = document.createElement('div');
  placeholder.className = 'review-share-placeholder';
  placeholder.textContent = 'Share options coming soon';
  return placeholder;
}

export function createReviewScreen(photoBlob: Blob, handlers: ReviewScreenHandlers): ReviewScreen {
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
  retakeButton.textContent = 'Retake';
  retakeButton.addEventListener('click', () => handlers.onRetake());

  actions.append(retakeButton);
  root.append(photo, actions, createSharePlaceholder());

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
