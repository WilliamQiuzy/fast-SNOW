# Fast-SNOW

Fast-SNOW is a faster 4D scene-graph pipeline for video spatial reasoning.

This repository is currently implementation-oriented: we are standardizing a practical route that targets **higher speed than SNOW** with **comparable reasoning quality**.

## 1) What We Are Building

Given a video (and optional question), Fast-SNOW builds a 4D scene graph and uses it as structured context for a VLM.

Planned route:
1. `RAM++` on each frame for object discovery tags.
2. `SAM3` text-prompted segmentation + cross-frame propagation.
3. `DA3` depth/pose estimation per frame.
4. Mask-depth backprojection to 3D object states.
5. Cross-run global ID fusion.
6. SNOW-aligned `STEP` token construction.
7. 4DSG assembly (spatial graph + temporal tracks).
8. JSON serialization + keyframe visual anchors for VLM inference.

## 2) Core Design Decisions

- Speed-first but quality-preserving: remove expensive 3D clustering loop from SNOW-style discovery.
- `STEP` compatibility: node representation follows SNOW’s `S_t^k = {tau, c, s, theta}` format.
- Open-world reasoning: RAM++ tags are used for discovery, not mandatory SG semantic labels.
- Streaming-friendly runtime: per-frame DA3/RAM++ + continuous SAM3 propagation.

## 3) Output Format (Planned)

- **Nodes**: per-object STEP tokens across time (`F_k`).
- **Edges**:
  - ego-object relations (bearing/elevation/distance/motion trend),
  - object-object spatial relations (direction/elevation/distance).
- **Artifacts for VLM**:
  - 4DSG JSON,
  - keyframe images (visual anchors).

Detailed spec: `docs/roadmap/Fast-SNOW_IMPLEMENTATION.md`

## 4) Repository Structure

```text
fast_snow/   core package (engine / vision / reasoning / data)
docs/        roadmap and technical docs
scripts/     runnable entrypoints and tests
assets/      examples and static assets
```

## 5) Environment and Weights

```bash
git clone <your_repo_url> fast-SNOW
cd fast-SNOW
conda create -n fast_snow python=3.10 -y
conda activate fast_snow
pip install -U pip
```

All checkpoints must be placed under:

```bash
fast_snow/models/
```

Example local cache setup:

```bash
export HF_HOME=$(pwd)/fast_snow/models/hf_cache
mkdir -p "$HF_HOME"
```

## 6) Current Status

- Roadmap is defined and being refined in `docs/roadmap/Fast-SNOW_IMPLEMENTATION.md`.
- `SNOWPipeline` (legacy) is still available for baseline reproduction.
- New `FastSNOWPipeline` is available at `fast_snow.engine.pipeline.fast_snow_pipeline`:
  - consumes per-frame DA3 outputs (`depth_t`, `K_t`, `T_wc_t`) + SAM3 detections,
  - implements Step 4/5/6/7/8 (3D state, global ID fusion, STEP, 4DSG, strict JSON).
