// One-off generator for a placeholder earring GLB used by Phase 3's debug
// harness. Not part of the build; run manually via
// `npm run generate:placeholder-earring` and commit the resulting file under
// vendor/earrings/ (same vendoring convention as vendor/mediapipe/). A real
// Shopify product GLB replaces this once the real pipeline is wired up
// (Phase 5+/12) — this only exists to validate GLTFLoader/DRACOLoader wiring
// and the render.ts anchoring math with something to look at.
//
// Plain Node ESM (not TypeScript): tsconfig.json's `include` is ["src"]
// only, and this is a one-off dev-time tool, not shipped in the bundle.

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as THREE from 'three';

// GLTFExporter's binary (.glb) export path reads its merged buffer via the
// browser FileReader API, which doesn't exist in plain Node — even for a
// texture-free export like this one. Node's global Blob (present since
// Node 18) already implements arrayBuffer(), so a minimal shim is enough;
// no jsdom or other DOM polyfill package needed for this one call site.
globalThis.FileReader = class {
  readAsArrayBuffer(blob) {
    blob
      .arrayBuffer()
      .then((buffer) => {
        this.result = buffer;
        this.onloadend?.();
      })
      .catch((error) => {
        this.onerror?.(error);
      });
  }
};

const { GLTFExporter } = await import('three/examples/jsm/exporters/GLTFExporter.js');

// Authored in meters (glTF's required unit) so render.ts's
// faceScale-to-real-world-size conversion produces a plausible on-screen
// size. A small huggie-hoop scale.
const HOOP_OUTER_RADIUS_M = 0.008; // 8mm ring radius (~16mm diameter)
const TUBE_RADIUS_M = 0.0008; // 0.8mm wire thickness
const RADIAL_SEGMENTS = 16;
const TUBULAR_SEGMENTS = 48;
const GOLD_COLOR_HEX = 0xd4af37;

const outputPath = join(
  dirname(fileURLToPath(import.meta.url)),
  '..',
  'vendor',
  'earrings',
  'placeholder-earring.glb',
);

const geometry = new THREE.TorusGeometry(
  HOOP_OUTER_RADIUS_M,
  TUBE_RADIUS_M,
  RADIAL_SEGMENTS,
  TUBULAR_SEGMENTS,
);
// TorusGeometry is centered on its own geometric center by default. Shift
// every vertex so local (0,0,0) sits at the ring's top instead — that's the
// point render.ts anchors to the ear, so the rest of the ring should hang
// below it in local space (down = negative local Y, matching render.ts's
// glTF-Y-up convention).
geometry.translate(0, -(HOOP_OUTER_RADIUS_M + TUBE_RADIUS_M), 0);

const material = new THREE.MeshStandardMaterial({
  color: GOLD_COLOR_HEX,
  metalness: 0.9,
  roughness: 0.3,
});

const scene = new THREE.Scene();
scene.add(new THREE.Mesh(geometry, material));

const glb = await new GLTFExporter().parseAsync(scene, { binary: true });

mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(outputPath, Buffer.from(glb));

console.log(`Wrote ${outputPath} (${Buffer.from(glb).byteLength} bytes)`);
