/**
 * The 3-2-1 countdown overlay shown before a snapshot capture, giving the
 * user a moment to pose. Mounts into whatever container it's given and
 * returns a cancel() closure — used by liveTryOn.ts's dispose() to clear a
 * pending timer (and remove the overlay) if the modal closes mid-countdown,
 * without ever calling onComplete for a session that's being torn down.
 */

const COUNT_FROM = 3;
const STEP_MS = 1000;

/**
 * Renders into `container` and starts counting down immediately. Calls
 * `onComplete` once, after the last number has been shown for its full
 * second, then removes itself. Returns `cancel()`, which clears the
 * pending timer and removes the overlay without calling `onComplete`.
 */
export function renderCountdown(container: HTMLElement, onComplete: () => void): () => void {
  const overlay = document.createElement('div');
  overlay.className = 'countdown-overlay';

  const number = document.createElement('span');
  number.className = 'countdown-number';

  overlay.append(number);
  container.append(overlay);

  let remaining = COUNT_FROM;
  let timeoutId: ReturnType<typeof setTimeout> | undefined;

  function showCurrentNumber(): void {
    number.textContent = String(remaining);
    // Force a reflow between removing and re-adding the pulse class so the
    // CSS animation restarts on every digit instead of no-opping on repeat.
    number.classList.remove('countdown-number--pulse');
    void number.offsetWidth;
    number.classList.add('countdown-number--pulse');
  }

  function tick(): void {
    if (remaining <= 0) {
      overlay.remove();
      onComplete();
      return;
    }
    showCurrentNumber();
    remaining -= 1;
    timeoutId = setTimeout(tick, STEP_MS);
  }

  tick();

  return function cancel(): void {
    if (timeoutId !== undefined) {
      clearTimeout(timeoutId);
    }
    overlay.remove();
  };
}
