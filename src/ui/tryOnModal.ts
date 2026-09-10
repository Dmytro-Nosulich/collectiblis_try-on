/**
 * Top-level orchestrator: opens the modal shell, shows the mode chooser
 * first, and swaps in Live Try-On when picked. This is what main.ts's real
 * openTryOn() calls.
 */

import type { TryOnOptions } from '../main.ts';
import { createModalShell } from './modal.ts';
import { renderModeChooser } from './modeChooser.ts';
import { renderLiveTryOn } from './liveTryOn.ts';

// A second openTryOn() call while one is already open is a no-op rather
// than stacking modals — v1 has no defined behavior for concurrent try-on
// sessions, and this is the simplest safe default.
let modalOpen = false;

export function openTryOnModal(options: TryOnOptions): void {
  if (modalOpen) {
    return;
  }
  modalOpen = true;

  let liveTryOnCleanup: (() => void) | undefined;

  function handleClose(): void {
    liveTryOnCleanup?.();
    liveTryOnCleanup = undefined;
    shell.destroy();
    modalOpen = false;
  }

  function showModeChooser(): void {
    liveTryOnCleanup?.();
    liveTryOnCleanup = undefined;
    renderModeChooser(shell.contentEl, {
      onLiveTryOn: () => {
        liveTryOnCleanup = renderLiveTryOn(shell.contentEl, options, {
          onBackToChooser: showModeChooser,
        });
      },
    });
  }

  const shell = createModalShell(handleClose);
  showModeChooser();
}
