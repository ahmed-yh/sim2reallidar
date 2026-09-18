/* The spine of the demo: site metadata, the shared class palette, and the
 * ordered list of sections.
 *
 * Adding a section is an edit here, as long as its `type` already exists in
 * assets/js/sections.js SECTION_RENDERERS. Adding a model to an existing
 * section is an edit here plus one entry in metrics.js.
 *
 * `classes` is the single source of the class palette for the whole page --
 * table swatches, player legends and the 3D viewer all read it, and the
 * exporter writes it from viz/playback_common.CLASS_COLORS so it can't drift
 * from what the rendered frames actually use.
 */
DEMO.register("manifest", "main", {
  "schemaVersion": 1,
  "generatedAt": "2026-09-19",
  "generatedBy": "hand-written (A1); demo/build_demo.py takes over at B5",

  "site": {
    "navBrand": "sim2real lidar",
    "title": "Sim2Real LiDAR Perception for Autonomous Navigation",
    "subtitle": "Three encoder candidates trained on simulated LiDAR, compared on one shared held-out test set, then run against real Ouster OS1-64 recordings from a Clearpath Jackal.",
    "author": "TODO — your name",
    "degree": "Bachelor Thesis",
    "university": "TODO — university",
    "department": "TODO — department",
    "supervisors": ["TODO — supervisor"],
    "date": "2026",
    "robot": "Clearpath Jackal · Ouster OS1-64",
    "repoUrl": "https://github.com/ahmed-yh/sim2reallidar",
    "disclaimer": "Every result on this page was rendered offline by the project's own scripts and then embedded here. No model runs in your browser. Each playback states the checkpoint it came from."
  },

  "classes": [
    { "id": 0, "key": "environment", "name": "Environment", "color": "#11151a", "object": false },
    { "id": 4, "key": "sphere",      "name": "Sphere",      "color": "#f2a65a", "object": true },
    { "id": 5, "key": "cylinder",    "name": "Cylinder",    "color": "#4fd1c5", "object": true },
    { "id": 6, "key": "box2",        "name": "Box 2",       "color": "#f0d65a", "object": true },
    { "id": 7, "key": "box1",        "name": "Box 1",       "color": "#eb6e5a", "object": true }
  ],

  "sections": [
    {
      "id": "title",
      "type": "hero",
      "nav": false
    },

    {
      "id": "problem",
      "type": "prose",
      "nav": "Problem",
      "title": "Compressing a LiDAR scan into something a policy can use",
      "html": "<p>A navigation policy cannot consume 65,536 raw LiDAR returns per scan. It needs a compact latent vector. The question this thesis asks is which encoder architecture produces the best one — and, critically, whether an encoder trained entirely in simulation still behaves sensibly on a real sensor.</p><p>Three candidates were trained on the same 7,600-scan simulated dataset and scored on the same held-out split: a point-based <strong>PointNet++</strong>, a range-image <strong>CNN autoencoder</strong>, and <strong>SalsaNext</strong>, a published range-image segmentation network. Two of them also predict a semantic class per point, which is what makes the sim-to-real gap visible rather than just numerical.</p>"
    },

    {
      "id": "candidates",
      "type": "comparison",
      "nav": "Candidates",
      "title": "Three candidates, one shared test set",
      "blurb": "<p>Reconstruction is the one task all three are actually trying to do, so it is the axis they are compared on — in identical normalized units, on identical held-out samples, using the same normalization constant.</p>"
    },

    {
      "id": "perclass",
      "type": "table",
      "nav": "Per-class",
      "title": "Where PointNet++ actually finds the objects",
      "blurb": "<p>Overall point accuracy is a misleading headline here: about 99% of every scan is environment, so a model that predicted \"environment\" everywhere would still score ~99%. The per-class numbers are the real signal.</p>",
      "path": "perClass"
    }
  ]
});
