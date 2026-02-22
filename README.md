# Fast-SNOW

Fast-SNOW is a faster 4D scene-graph pipeline for video spatial reasoning. Given a video and a question, it builds a **4D Scene Graph (4DSG)** and uses it as structured context for a Vision-Language Model (VLM).

## Pipeline Overview

```
Video → Frame Sampling → DA3 (depth + poses) → FastSAM + SAM3 (segmentation + tracking) → 4DSG → VLM → Answer
```

1. **DA3** batch inference → metric depth maps + temporally consistent camera poses.
2. **FastSAM** class-agnostic instance segmentation (open-world, no class labels).
3. **SAM3** video object tracking with two-pass architecture:
   - Phase 2a: Init + full propagation from frame 0 detections.
   - Phase 2b: Per-frame discovery of new objects via IoU comparison.
   - Phase 2c: Partial propagation to merge new objects with cached masks.
4. **3D back-projection**: mask + depth → world-coordinate 3D centroids and shapes.
5. **STEP tokens**: per-object per-frame `S_t^k = {τ, c, s, θ}` (SNOW paper §3.2).
6. **4DSG assembly**: temporal tracks `F_k` + ego poses → structured JSON.
7. **VLM inference**: 4DSG text + keyframe images → spatial reasoning answers.

## Design Decisions

- **Speed-first**: skip SNOW's expensive 3D clustering; use FastSAM + SAM3 for direct mask-based discovery and tracking.
- **Open-world**: no class labels — VLM infers semantics from visual anchors + geometric tokens.
- **STEP-compatible**: node representation strictly follows SNOW's `S_t^k = {τ, c, s, θ}` format.
- **Online scene graph construction**: frames processed incrementally via producer-consumer GPU/CPU pipeline.
- **GPU memory management**: models unloaded between phases to fit on a single GPU.

## Repository Structure

```
fast_snow/
  engine/         pipeline orchestration and config
  vision/         model wrappers (DA3, FastSAM, SAM3)
  reasoning/      STEP token encoding and patch tokenization
docs/
  roadmap/        implementation spec
  bugs/           known issues and analysis
scripts/          tests and benchmarks
assets/           visualization scripts and examples
benchmark/        VLM4D evaluation scripts and results
test/             integration tests
```

## Setup

```bash
git clone https://github.com/WilliamQiuzy/fast-SNOW.git
cd fast-SNOW
conda create -n snow python=3.10 -y
conda activate snow
pip install -U pip
```

### Model Weights

All checkpoints go under `fast_snow/models/`:

```
fast_snow/models/
  sam3/           # SAM3 checkpoint
  fastsam/        # FastSAM-s.pt
  da3-small/      # DA3 small variant
```

### Clone Vision Model Source Code

```bash
# SAM3
git clone https://github.com/anthropics-ai/sam3.git fast_snow/vision/sam3

# DA3 (Video-Depth-Anything v3)
git clone https://github.com/anthropics-ai/da3.git fast_snow/vision/da3
```

### Environment Variables

```bash
# For VLM inference
export OPENAI_API_KEY="your-key"       # for GPT-5.2
export GOOGLE_API_KEY="your-key"       # for Gemini
```

## Quick Start

```python
from fast_snow.engine.config import FastSNOWConfig
from fast_snow.engine.pipeline.fast_snow_e2e import FastSNOWEndToEnd

config = FastSNOWConfig()
e2e = FastSNOWEndToEnd(config)
result = e2e.process_video("video.mp4", "What is in front of the camera?")
print(result.answer)
```

## Tests

```bash
# Unit tests (no GPU required)
python scripts/test_fast_snow_smoke.py
python scripts/test_fast_snow_step01.py

# GPU integration test
python test/test_fastsam_sam3_integration.py
```

## Documentation

- [Implementation Spec](docs/roadmap/Fast-SNOW_IMPLEMENTATION.md) — full pipeline architecture and hyperparameters
- [Known Issues](docs/bugs/) — SAM3 OOM, track re-identification, silent object loss, etc.

## Known Issues

| Issue | Status | Doc |
|-------|--------|-----|
| SAM3 OOM on V100-32GB | Open | [SAM3_V100_OOM.md](docs/bugs/SAM3_V100_OOM.md) |
| Archived tracks not re-identified | Open | [TRACK_NO_REIDENTIFICATION.md](docs/bugs/TRACK_NO_REIDENTIFICATION.md) |
| SAM3 silent object loss | Mitigated | [SAM3_SILENT_OBJECT_LOSS.md](docs/bugs/SAM3_SILENT_OBJECT_LOSS.md) |
| CPU worker delayed error detection | Open | [CPU_WORKER_DELAYED_ERROR.md](docs/bugs/CPU_WORKER_DELAYED_ERROR.md) |
