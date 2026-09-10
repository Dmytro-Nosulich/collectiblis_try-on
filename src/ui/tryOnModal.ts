/**
 * Top-level orchestrator: opens the modal shell, shows the mode chooser
 * first, and swaps in Live Try-On or Upload Photo when picked. This is what
 * main.ts's real openTryOn() calls.
 */

import type { TryOnOptions } from '../main.ts';
import { createModalShell } from './modal.ts';
import { renderModeChooser } from './modeChooser.ts';
import { renderLiveTryOn } from './liveTryOn.ts';
import { renderUploadPhoto } from './uploadPhoto.ts';

// A second openTryOn() call while one is already open is a no-op rather
// than stacking modals — v1 has no defined behavior for concurrent try-on
// sessions, and this is the simplest safe default.
let modalOpen = false;

export function openTryOnModal(options: TryOnOptions): void {
  if (modalOpen) {
    return;
  }
  modalOpen = true;

  // Only one mode is ever active at a time (the mode chooser is always the
  // step in between), so one cleanup slot is enough regardless of which
  // mode is currently live.
  let activeModeCleanup: (() => void) | undefined;

  function handleClose(): void {
    activeModeCleanup?.();
    activeModeCleanup = undefined;
    shell.destroy();
    modalOpen = false;
  }

  function showModeChooser(): void {
    activeModeCleanup?.();
    activeModeCleanup = undefined;
    renderModeChooser(shell.contentEl, {
      onLiveTryOn: () => {
        activeModeCleanup = renderLiveTryOn(shell.contentEl, options, {
          onBackToChooser: showModeChooser,
        });
      },
      onUploadPhoto: () => {
        activeModeCleanup = renderUploadPhoto(shell.contentEl, options, {
          onBackToChooser: showModeChooser,
        });
      },
    });
  }

  const shell = createModalShell(handleClose);
  showModeChooser();
}
