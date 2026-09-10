/**
 * Curated photo list for Choose a Model (Phase 10) — see chooseModel.ts.
 *
 * TEMPORARY — these are free-license stock photos, sourced as placeholders
 * pending final licensed model photos before launch (per build-plan.md's
 * Phase 10 instructions). All three are the free "Unsplash License"
 * (unsplash.com/license): free for commercial and non-commercial use, no
 * permission or attribution required. This comment block is the tracked,
 * authoritative record of where each photo came from — .claude/ is
 * gitignored in this repo, so build-progress.md's own note on this is a
 * secondary session-continuity copy, not the source of truth.
 *
 * model-01.jpg — photographer Aleksandar Komnenovic (@alexkomnenovic),
 *   https://unsplash.com/photos/fvnyj_yTfWc, downloaded 2026-09-10.
 * model-02.jpg — photographer Brooke Balentine,
 *   https://unsplash.com/photos/a-young-woman-smiles-with-head-tilted-gqCTggvUhZg,
 *   downloaded 2026-09-10.
 * model-03.jpg — photographer David Sedrakyan,
 *   https://unsplash.com/photos/VuEvTn-Ulhw, downloaded 2026-09-10.
 *
 * Chosen for a clearly visible, unobstructed ear (CLAUDE.md flags hair
 * occlusion as a known tracking weak point) across three different
 * hairstyles/angles rather than one easy repeated case.
 */

export interface ModelPhoto {
  id: string;
  filename: string;
  alt: string;
}

export const CURATED_MODEL_PHOTOS: ModelPhoto[] = [
  {
    id: 'model-01',
    filename: 'model-01.jpg',
    alt: 'Woman with light brown hair pulling it back to reveal her pierced ear, looking toward the camera.',
  },
  {
    id: 'model-02',
    filename: 'model-02.jpg',
    alt: 'Woman with dark hair, head tilted, smiling with her ear and a small stud earring visible.',
  },
  {
    id: 'model-03',
    filename: 'model-03.jpg',
    alt: 'Woman with an undercut hairstyle in profile, her ear clearly visible.',
  },
];
