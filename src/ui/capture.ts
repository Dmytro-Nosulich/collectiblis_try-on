/**
 * Composites the current video frame + Three.js earring overlay into a
 * single mirrored PNG blob — what the snapshot button in liveTryOn.ts
 * actually captures.
 *
 * There is no canvas anywhere in the pipeline that already holds mirrored
 * pixels: the on-screen "mirror" effect is purely a CSS `transform:
 * scaleX(-1)` on `.mirror-wrapper` (see liveTryOn.ts's header comment), so
 * this function has to redo that mirroring itself when drawing into an
 * offscreen canvas, to produce an exported image matching what the user
 * actually saw and posed for (the project's explicit decision, not a "true"
 * unflipped orientation).
 */

const ERROR_PREFIX = '[CollectiblissTryOn]';

export function captureMirroredComposite(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
): Promise<Blob> {
  const width = video.videoWidth;
  const height = video.videoHeight;

  const output = document.createElement('canvas');
  output.width = width;
  output.height = height;
  const ctx = output.getContext('2d');
  if (!ctx) {
    return Promise.reject(new Error(`${ERROR_PREFIX} could not get a 2D context for capture`));
  }

  // Mirror first, so both layers land in the same mirrored space and stay
  // pixel-aligned as a unit — matching how .mirror-wrapper mirrors its
  // video+canvas children together as one, rather than each independently.
  ctx.translate(width, 0);
  ctx.scale(-1, 1);

  // Same stacking order as the live DOM: video first, then the
  // transparent-background WebGL earring canvas on top. Canvas 2D's
  // drawImage respects the source canvas's alpha channel under the default
  // source-over composite operation, so this alpha-blends only the earring
  // pixels over the video, same as the browser already does by layering
  // .live-canvas over .live-video on screen.
  ctx.drawImage(video, 0, 0, width, height);
  ctx.drawImage(canvas, 0, 0, width, height);

  return new Promise<Blob>((resolve, reject) => {
    output.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error(`${ERROR_PREFIX} failed to export capture as a PNG blob`));
      }
    }, 'image/png');
  });
}
