/**
 * Top-level orchestrator: opens the modal shell, shows the mode chooser
 * first, and swaps in Live Try-On, Upload Photo, or Choose a Model when
 * picked. This is what main.ts's real openTryOn() calls.
 *
 * Phase 11 lazy-loading note: nothing AR-related fetches before a mode is
 * picked. Importing renderLiveTryOn/renderUploadPhoto/renderChooseModel
 * above only costs JS parse/eval — unavoidable given Phase 0's
 * single-IIFE-bundle decision (no dynamic import() anywhere in this repo),
 * since the whole widget ships as one <script> tag for simple Shopify theme
 * embedding. It never costs a network request: neither tracking.ts nor
 * render.ts has any top-level (module-eval-time) side effects, and the
 * actual MediaPipe WASM/model fetch (tracking.ts's createFaceLandmarker) and
 * GLB fetch (render.ts's getGltfLoader().loadAsync) both live inside
 * function bodies that are only reachable once one of this file's three
 * onLiveTryOn/onUploadPhoto/onChooseModel handlers below actually runs —
 * i.e. only after a mode tile is clicked in renderModeChooser, which itself
 * imports none of this fetch logic. Verified via direct call-site tracing,
 * not assumed — confirm via a real browser's Network panel too (see
 * build-progress.md's Phase 11 entry).
 */

import type { TryOnOptions } from '../main.ts';
import { createModalShell } from './modal.ts';
import { renderModeChooser } from './modeChooser.ts';
import { renderLiveTryOn } from './liveTryOn.ts';
import { renderUploadPhoto } from './uploadPhoto.ts';
import { renderChooseModel } from './chooseModel.ts';

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
      onChooseModel: () => {
        activeModeCleanup = renderChooseModel(shell.contentEl, options, {
          onBackToChooser: showModeChooser,
        });
      },
    });
  }

  const shell = createModalShell(handleClose);
  showModeChooser();
}
