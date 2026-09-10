/**
 * Composites a source (live video frame, or a static uploaded photo) +
 * Three.js earring overlay into a single PNG blob.
 *
 * captureMirroredComposite (Live Try-On, Phase 6) manually mirrors before
 * drawing: there is no canvas anywhere in the live pipeline that already
 * holds mirrored pixels, since the on-screen "mirror" effect is purely a
 * CSS `transform: scaleX(-1)` on `.mirror-wrapper` (see liveTryOn.ts's
 * header comment) — this redoes that mirroring when drawing into an
 * offscreen canvas, to produce an exported image matching what the user
 * actually saw and posed for (the project's explicit decision, not a "true"
 * unflipped orientation).
 *
 * compositeEarringOntoImage (Upload Photo, Phase 9) has no such mirror
 * step: an uploaded photo has no mirror-display convention anywhere in its
 * pipeline (mirroring is specific to Live Try-On's self-view convention),
 * so mirroring here would flip the output backwards relative to the photo
 * the user actually chose.
 */

const ERROR_PREFIX = '[CollectiblissTryOn]';

function canvasToPngBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error(`${ERROR_PREFIX} failed to export capture as a PNG blob`));
      }
    }, 'image/png');
  });
}

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

  return canvasToPngBlob(output);
}

/**
 * Static-photo counterpart to captureMirroredComposite (Phase 9) — see this
 * file's header comment for why there's deliberately no mirror step here.
 */
export function compositeEarringOntoImage(
  image: HTMLImageElement,
  earringCanvas: HTMLCanvasElement,
): Promise<Blob> {
  const width = image.naturalWidth;
  const height = image.naturalHeight;

  const output = document.createElement('canvas');
  output.width = width;
  output.height = height;
  const ctx = output.getContext('2d');
  if (!ctx) {
    return Promise.reject(new Error(`${ERROR_PREFIX} could not get a 2D context for compositing`));
  }

  ctx.drawImage(image, 0, 0, width, height);
  ctx.drawImage(earringCanvas, 0, 0, width, height);

  return canvasToPngBlob(output);
}
