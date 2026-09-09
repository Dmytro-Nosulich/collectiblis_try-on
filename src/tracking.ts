/**
 * Face/landmark tracking: loads MediaPipe's FaceLandmarker
 * (`@mediapipe/tasks-vision`) and runs detection against video/image
 * frames, approximates a left/right earlobe anchor point from the 478
 * returned landmarks (there is no dedicated ear landmark), and smooths
 * that point with a One Euro Filter (`1eurofilter`) to remove per-frame
 * jitter. See CLAUDE.md "Known technical challenges".
 *
 * Implemented in Phase 1 (raw landmark detection) and Phase 2 (ear anchor
 * approximation + smoothing).
 */
export {};
