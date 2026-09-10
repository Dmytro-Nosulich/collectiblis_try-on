/**
 * Generic modal shell: a host element with an isolated shadow root
 * (backdrop + panel + close button + an empty content container), appended
 * directly to document.body — not a cross-origin iframe, per CLAUDE.md's
 * iOS Safari camera-access note. Knows nothing about mode chooser / live
 * try-on / any specific content; callers render into `contentEl`.
 *
 * Shadow DOM, not prefixed classes: isolates this widget's CSS from
 * whatever an arbitrary Shopify theme injects globally, in both directions.
 * The host element itself can't be styled from inside the shadow root (a
 * shadow root only styles its own content), so its positioning is set
 * directly via inline styles here instead of through styles.css.
 */

import stylesCss from './styles.css?inline';

// Effectively unbounded — the widget must sit above anything a Shopify
// theme could plausibly stack, and there's nothing else on the page for it
// to conflict with (max signed 32-bit int, the practical CSS z-index ceiling).
const HOST_Z_INDEX = 2147483647;

export interface ModalShell {
  readonly contentEl: HTMLElement;
  destroy(): void;
}

function lockBodyScroll(): () => void {
  const previousOverflow = document.body.style.overflow;
  document.body.style.overflow = 'hidden';
  return () => {
    document.body.style.overflow = previousOverflow;
  };
}

/**
 * Builds and mounts the modal shell, wired to call `onClose` on any of:
 * the header close button, a click directly on the backdrop (not the
 * panel), or Escape. Does NOT call onClose itself when torn down via
 * `destroy()` — that's the caller's own teardown path, calling onClose
 * again would be circular.
 */
export function createModalShell(onClose: () => void): ModalShell {
  const host = document.createElement('div');
  host.style.cssText = `position:fixed;inset:0;z-index:${HOST_Z_INDEX};`;
  const shadowRoot = host.attachShadow({ mode: 'open' });

  const styleEl = document.createElement('style');
  styleEl.textContent = stylesCss;

  const overlay = document.createElement('div');
  overlay.className = 'overlay';

  const panel = document.createElement('div');
  panel.className = 'panel';
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-modal', 'true');

  const header = document.createElement('div');
  header.className = 'panel-header';

  const closeButton = document.createElement('button');
  closeButton.type = 'button';
  closeButton.className = 'close-button';
  closeButton.setAttribute('aria-label', 'Close');
  closeButton.textContent = '×';
  closeButton.addEventListener('click', () => onClose());

  const contentEl = document.createElement('div');
  contentEl.className = 'content';

  header.append(closeButton);
  panel.append(header, contentEl);
  overlay.append(panel);
  shadowRoot.append(styleEl, overlay);

  overlay.addEventListener('click', (event) => {
    if (event.target === overlay) {
      onClose();
    }
  });

  function handleKeydown(event: KeyboardEvent): void {
    if (event.key === 'Escape') {
      onClose();
    }
  }
  document.addEventListener('keydown', handleKeydown);

  const unlockBodyScroll = lockBodyScroll();
  document.body.append(host);

  function destroy(): void {
    document.removeEventListener('keydown', handleKeydown);
    unlockBodyScroll();
    host.remove();
  }

  return { contentEl, destroy };
}
