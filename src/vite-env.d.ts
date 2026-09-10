/// <reference types="vite/client" />

// Not declared by this Vite version's own client.d.ts (only plain `*.css`,
// `*.module.css`, etc. are) — needed by src/ui/styles.css's `?inline` import
// so the widget's CSS ships embedded in the single JS bundle rather than as
// a separate .css file a Shopify <script>-tag consumer would have no way to
// auto-load. See build-progress.md's Phase 5 entry.
declare module '*.css?inline' {
  const css: string;
  export default css;
}
