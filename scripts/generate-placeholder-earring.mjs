// One-off generator for the placeholder earring GLBs used by the Phase
// 3/4 debug harness. Not part of the build; run manually via
// `npm run generate:placeholder-earring` and commit the resulting files
// under vendor/earrings/ (same vendoring convention as vendor/mediapipe/).
// Real Shopify product GLBs replace these once the real pipeline is wired
// up (Phase 5+/12) — these only exist to validate GLTFLoader/DRACOLoader
// wiring and the render.ts anchoring/rotation math with something to look
// at, including a rigid-vs-dangly comparison for Phase 4's orientation
// tuning (a longer drop sweeps a visibly larger arc for the same head
// rotation than a rigid hoop does, with no joints/rigging needed).
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

const EARRINGS_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', 'vendor', 'earrings');
const GOLD_COLOR_HEX = 0xd4af37;
const GOLD_MATERIAL_PARAMS = { color: GOLD_COLOR_HEX, metalness: 0.9, roughness: 0.3 };

/**
 * Rigid gold hoop, authored in meters (glTF's required unit) so render.ts's
 * faceScale-to-real-world-size conversion produces a plausible on-screen
 * size. A small huggie-hoop scale.
 */
function buildHoopEarring() {
  const outerRadiusM = 0.008; // 8mm ring radius (~16mm diameter)
  const tubeRadiusM = 0.0008; // 0.8mm wire thickness
  const radialSegments = 16;
  const tubularSegments = 48;

  const geometry = new THREE.TorusGeometry(
    outerRadiusM,
    tubeRadiusM,
    radialSegments,
    tubularSegments,
  );
  // TorusGeometry is centered on its own geometric center by default. Shift
  // every vertex so local (0,0,0) sits at the ring's top instead — that's
  // the point render.ts anchors to the ear, so the rest of the ring should
  // hang below it in local space (down = negative local Y, matching
  // render.ts's glTF-Y-up convention).
  geometry.translate(0, -(outerRadiusM + tubeRadiusM), 0);

  const scene = new THREE.Scene();
  scene.add(new THREE.Mesh(geometry, new THREE.MeshStandardMaterial(GOLD_MATERIAL_PARAMS)));
  return scene;
}

/**
 * "Dangly" test piece: a small post at the anchor, a thin chain dropping
 * ~15mm below it, and a larger bead at the bottom — a rigid single Group,
 * no joints/skinning (animating a hinge is out of scope for Phase 4).
 * Local origin (0,0,0) is the top of the post, same "anchor = attachment
 * point" convention as the hoop. Deliberately much longer than the hoop's
 * ~16mm diameter: the point of this model is to visibly sweep a larger arc
 * than the hoop for the same head-rotation angle, since a longer lever arm
 * amplifies a given angle with zero extra physics.
 */
function buildDanglyEarring() {
  const postRadiusM = 0.0015;
  const postLengthM = 0.003;
  const chainRadiusM = 0.0004;
  const chainLengthM = 0.015;
  const beadRadiusM = 0.003;

  const material = new THREE.MeshStandardMaterial(GOLD_MATERIAL_PARAMS);
  const group = new THREE.Group();

  const post = new THREE.Mesh(
    new THREE.CylinderGeometry(postRadiusM, postRadiusM, postLengthM, 12),
    material,
  );
  post.position.y = -postLengthM / 2;
  group.add(post);

  const chain = new THREE.Mesh(
    new THREE.CylinderGeometry(chainRadiusM, chainRadiusM, chainLengthM, 8),
    material,
  );
  chain.position.y = -(postLengthM + chainLengthM / 2);
  group.add(chain);

  const bead = new THREE.Mesh(new THREE.SphereGeometry(beadRadiusM, 16, 16), material);
  bead.position.y = -(postLengthM + chainLengthM + beadRadiusM);
  group.add(bead);

  const scene = new THREE.Scene();
  scene.add(group);
  return scene;
}

async function exportGlb(scene, outputPath) {
  const glb = await new GLTFExporter().parseAsync(scene, { binary: true });
  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, Buffer.from(glb));
  console.log(`Wrote ${outputPath} (${Buffer.from(glb).byteLength} bytes)`);
}

await exportGlb(buildHoopEarring(), join(EARRINGS_DIR, 'placeholder-earring.glb'));
await exportGlb(buildDanglyEarring(), join(EARRINGS_DIR, 'placeholder-earring-dangly.glb'));
