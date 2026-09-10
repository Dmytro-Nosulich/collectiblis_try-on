import { openTryOnModal } from './ui/tryOnModal.ts';

/**
 * Options passed to `openTryOn`, shared by both entry points (product-list
 * card and product detail page) per CLAUDE.md. `glbUrl` comes from the
 * product's Shopify media (`Model3d`/`Model3dSource`) at template render
 * time — no extra API call needed to fetch it.
 */
export interface TryOnOptions {
  /** Shopify product id, used for analytics/debugging context. */
  productId: string;
  /** Direct URL to the product's earring GLB model. */
  glbUrl: string;
  /** Product display name, used in UI copy (e.g. snapshot/share text). */
  name: string;
  /**
   * Overrides where this widget's own runtime assets (MediaPipe WASM +
   * model file) are fetched from. Defaults to the directory this script was
   * loaded from — see src/assets.ts.
   *
   * NOT YET WIRED THROUGH to tracking.ts/render.ts as of Phase 5:
   * createFaceLandmarker() and getGltfLoader() both call getAssetBaseUrl()
   * with no argument, so they always use the fallback chain regardless of
   * what's passed here. Harmless today (nothing passes this yet), but flag
   * for Phase 12/13 — threading an override through matters once the
   * widget's assets live on a different domain than the Shopify page
   * embedding it.
   */
  assetsBaseUrl?: string;
}

/**
 * Launches the try-on experience for a single product. This is the one
 * shared entry point called from both the product-list card and the
 * product detail page (see CLAUDE.md "Entry points").
 */
export function openTryOn(options: TryOnOptions): void {
  openTryOnModal(options);
}
