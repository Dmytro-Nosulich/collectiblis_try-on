/**
 * UI states shared by every mode that runs the static-photo try-on pipeline
 * (single-shot MediaPipe IMAGE-mode detection against a decoded <img>, no
 * live video loop): Upload Photo (Phase 9) and Choose a Model (Phase 10).
 * Extracted out of uploadPhoto.ts once Choose a Model needed the identical
 * processing spinner and near-identical error screen — see
 * .claude/build-progress.md's Phase 9 entry, which flagged this exact split
 * as the likely outcome once a second real consumer showed up.
 */

const STATUS_PLACING_EARRING = 'Placing the earring...';

export function createProcessingState(): HTMLElement {
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

/** Shared by every failure case across both static-photo modes — only the message and (optionally) the retry button's label differ. */
export function createErrorState(
  message: string,
  onTryAgain: () => void,
  onBackToChooser: () => void,
  tryAgainLabel = 'Try a different photo',
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
  tryAgainButton.textContent = tryAgainLabel;
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
