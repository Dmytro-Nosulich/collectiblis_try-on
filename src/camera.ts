/**
 * Camera access: requests the webcam via `getUserMedia` (video only, no
 * audio) and attaches the stream to a `<video>` element, per CLAUDE.md's
 * architecture. Also responsible for explicitly stopping all tracks when
 * the try-on modal closes (see step2-design-decisions.md — avoids the
 * browser's persistent "camera in use" indicator).
 *
 * Implemented in Phase 1.
 */
export {};
