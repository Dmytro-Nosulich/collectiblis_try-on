/**
 * Camera access: requests the webcam via `getUserMedia` (video only, no
 * audio) and attaches the stream to a `<video>` element, per CLAUDE.md's
 * architecture. Also responsible for explicitly stopping all tracks when
 * the try-on modal closes (see step2-design-decisions.md — avoids the
 * browser's persistent "camera in use" indicator).
 */

export interface CameraSession {
  readonly video: HTMLVideoElement;
  readonly stream: MediaStream;
}

// Desktop-first debug default: front camera, 720p ideal (not exact — lets
// the browser pick the closest supported mode instead of failing outright
// on hardware that can't hit exactly 1280x720).
const DEFAULT_CONSTRAINTS: MediaStreamConstraints = {
  video: {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    facingMode: 'user',
  },
  audio: false,
};

function waitForLoadedMetadata(video: HTMLVideoElement): Promise<void> {
  if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    video.addEventListener('loadedmetadata', () => resolve(), { once: true });
  });
}

/**
 * Requests the webcam and attaches it to `video`. Resolves once the video's
 * intrinsic dimensions (`videoWidth`/`videoHeight`) are available, since
 * callers (e.g. tracking.ts's canvas sizing) need those before they can do
 * anything useful.
 */
export async function startCamera(
  video: HTMLVideoElement,
  constraints: MediaStreamConstraints = DEFAULT_CONSTRAINTS,
): Promise<CameraSession> {
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;

  // Attach the listener before awaiting play() to avoid a race where
  // loadedmetadata fires before we're listening for it.
  const metadataLoaded = waitForLoadedMetadata(video);
  await video.play();
  await metadataLoaded;

  return { video, stream };
}

/** Stops all tracks on the session's stream, releasing the camera. */
export function stopCamera(session: CameraSession): void {
  for (const track of session.stream.getTracks()) {
    track.stop();
  }
}
