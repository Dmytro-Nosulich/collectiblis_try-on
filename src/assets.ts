/**
 * Resolves the base URL this widget should fetch its own runtime assets
 * from (MediaPipe's WASM runtime + model file, GLB earring models, etc).
 *
 * Vite 8 (Rolldown) does not polyfill `import.meta.url` in IIFE output — it
 * comes out as `undefined` — so we can't use the usual
 * `new URL('./x', import.meta.url)` trick to locate assets relative to the
 * bundle. Instead:
 *
 * 1. An explicit `assetsBaseUrl` (passed to `openTryOn`) always wins. This
 *    lets the Shopify integration point at a CDN path that differs from
 *    wherever the <script> tag itself was loaded from.
 * 2. Otherwise, fall back to the directory containing the currently
 *    executing <script> tag, captured once at module-evaluation time via
 *    `document.currentScript`. This works because the production build is
 *    an IIFE loaded as a classic `<script src="...">` — execution is
 *    synchronous, so `document.currentScript` still points at this script
 *    while the module's top-level code runs. It would NOT work if this were
 *    captured lazily inside `openTryOn()`, since `currentScript` is only
 *    valid during synchronous script execution, not later in response to a
 *    click handler — hence capturing it below at import time.
 * 3. Final fallback: the page's own base URI. This is what the Vite dev
 *    server harness (index.html) hits, since it loads main.ts as a
 *    same-origin ES module rather than a built <script> tag, so
 *    `document.currentScript` is null there.
 */

const capturedScriptSrc =
  document.currentScript instanceof HTMLScriptElement ? document.currentScript.src : undefined;

const scriptDirectoryUrl = capturedScriptSrc
  ? capturedScriptSrc.slice(0, capturedScriptSrc.lastIndexOf('/') + 1)
  : undefined;

function withTrailingSlash(url: string): string {
  return url.endsWith('/') ? url : `${url}/`;
}

export function getAssetBaseUrl(override?: string): string {
  if (override) {
    return withTrailingSlash(override);
  }
  if (scriptDirectoryUrl) {
    return scriptDirectoryUrl;
  }
  // document.baseURI is the page's own full URL (query string, hash, and
  // any filename included, e.g. ".../?forceDeviceTier=low") — NOT already a
  // directory-only URL like scriptDirectoryUrl above. Resolving '.' against
  // it strips the filename/query/hash down to the containing directory
  // (with a trailing slash), matching what every caller concatenates
  // relative asset paths onto. Without this, any query string on the dev
  // harness page (e.g. this file's own `?forceDeviceTier=` debug override)
  // corrupted every asset URL built from this fallback — found via Phase 11
  // testing, not a Phase 11 regression (this fallback predates it).
  return new URL('.', document.baseURI).href;
}
