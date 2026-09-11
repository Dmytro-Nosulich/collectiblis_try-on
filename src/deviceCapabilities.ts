/**
 * Phase 11: a lightweight, one-time-per-session device-tier check, used to
 * scale back detection cadence (tracking.ts's detectionCadenceTuning) and
 * WebGL render resolution (render.ts's CreateEarringSceneOptions.
 * pixelRatioCap) on detected low-end devices.
 *
 * navigator.hardwareConcurrency (logical core count) is the CPU-power proxy
 * used for isLowEnd — a genuine, if imperfect, signal. window.devicePixelRatio
 * is deliberately NOT treated as a tier signal: a device isn't low-end just
 * because it has a dense screen (many budget Android phones report DPR 3 on
 * modest silicon; a high-end iPad reports DPR 2 on a very powerful chip).
 * DPR is used purely as a workload multiplier — rendering fills DPR^2 more
 * physical pixels — so it's capped at UNCAPPED_PIXEL_RATIO_MAX for everyone
 * (a visually-negligible-loss fill-rate guard) and capped harder at 1 for
 * detected low-end devices specifically, matching today's exact pre-Phase-11
 * behavior there.
 *
 * Thresholds below are unvalidated starting guesses, same disclaimer class
 * as tracking.ts's tuned constants — flagged for real-device confirmation,
 * not made live-tunable via index.html's debug sliders (unlike
 * earAnchorTuning/rotationTuning/detectionCadenceTuning) since this is a
 * one-time per-session decision, not a continuously-applied per-frame value.
 * The ?forceDeviceTier query override serves the same "exercise both paths
 * easily" purpose instead, without needing to own literally low-end hardware.
 */

const LOW_END_HARDWARE_CONCURRENCY_MAX = 4;
const UNCAPPED_PIXEL_RATIO_MAX = 2;

export interface DeviceTier {
  isLowEnd: boolean;
  /** Pass straight through to render.ts's CreateEarringSceneOptions.pixelRatioCap. */
  pixelRatioCap: number;
}

function forcedTierOverride(): DeviceTier | null {
  const forced = new URLSearchParams(location.search).get('forceDeviceTier');
  if (forced === 'low') {
    return { isLowEnd: true, pixelRatioCap: 1 };
  }
  if (forced === 'high') {
    return {
      isLowEnd: false,
      pixelRatioCap: Math.min(window.devicePixelRatio || 1, UNCAPPED_PIXEL_RATIO_MAX),
    };
  }
  return null;
}

export function detectDeviceTier(): DeviceTier {
  const forced = forcedTierOverride();
  if (forced) {
    return forced;
  }

  // undefined (older/privacy-restricted browsers) is treated as "unknown,"
  // not "low-end" — don't punish devices that simply don't report this.
  const hardwareConcurrency = navigator.hardwareConcurrency;
  const isLowEnd =
    typeof hardwareConcurrency === 'number' &&
    hardwareConcurrency > 0 &&
    hardwareConcurrency <= LOW_END_HARDWARE_CONCURRENCY_MAX;

  const rawPixelRatio = window.devicePixelRatio || 1;
  const pixelRatioCap = isLowEnd ? 1 : Math.min(rawPixelRatio, UNCAPPED_PIXEL_RATIO_MAX);

  return { isLowEnd, pixelRatioCap };
}
