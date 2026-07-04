# ROSE

ROSE is a faster 4D scene-graph pipeline for video spatial reasoning. Given a video and a question, it builds a **4D Scene Graph (4DSG)** and uses it as structured context for a Vision-Language Model (VLM).

## Pipeline

```
Video → Frame Sampling → DA3 (depth + poses) → FastSAM + SAM3 (segmentation + tracking)
      → 3D back-projection → STEP tokens → 4DSG → VLM → Answer
```

1. **DA3** batch inference → metric depth maps + temporally consistent camera poses.
2. **FastSAM** class-agnostic instance segmentation (open-world, no class labels).
3. **SAM3** video object tracking with two-pass architecture (init + propagation, per-frame discovery, partial propagation).
4. **3D back-projection**: mask + depth → world-coordinate centroids and shapes.
5. **STEP tokens**: per-object per-frame `S_t^k = {τ, c, s, θ}` (SNOW paper §3.2).
6. **4DSG assembly**: temporal tracks `F_k` + ego poses → structured JSON.
7. **VLM inference**: 4DSG text + keyframe images → spatial reasoning answers.

## Quick Start (Docker — recommended)

Requires Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed and a CUDA 12.x GPU (H100 / H200 tested).

```bash
git clone https://github.com/WilliamQiuzy/ROSE.git
cd ROSE
docker build -t rose .

# Run with GPU passthrough; mount a host dir for model weights so they
# persist across container runs.
mkdir -p ./weights
docker run --gpus all -it --rm \
    -v $(pwd)/weights:/workspace/rose/rose/models \
    rose
```

On first launch the entrypoint will warn that `rose/models/` is empty. Inside the container:

```bash
huggingface-cli login          # one-time, for gated models (Gemma, SAM3)
bash scripts/setup.sh --core   # downloads DA3-Small, SAM3, FastSAM-s, Gemma-3-4B-IT
```

After this completes, the weights live in your host's `./weights/` and you can re-run the container without re-downloading.

### Smoke test inside the container

```bash
python -c "import rose; from rose.engine.config import ROSEConfig; print('ok', ROSEConfig)"
python scripts/test_rose_smoke.py
```

## Quick Start (manual / conda)

If you prefer not to use Docker:

```bash
git clone https://github.com/WilliamQiuzy/ROSE.git
cd ROSE
conda create -n rose python=3.11 -y
conda activate rose
pip install --upgrade pip
pip install -r requirements.txt

# Third-party vision deps (not on PyPI):
git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git rose/vision/da3
pip install -e rose/vision/da3
git clone --depth 1 https://github.com/facebookresearch/sam2.git rose/vision/sam3
pip install -e rose/vision/sam3

# Weights:
huggingface-cli login
bash scripts/setup.sh --core
```

## Programmatic use

```python
from rose.engine.config import ROSEConfig
from rose.engine.pipeline.rose_e2e import ROSEEndToEnd

config = ROSEConfig()
e2e = ROSEEndToEnd(config)
result = e2e.process_video("video.mp4", "What is in front of the camera?")
print(result.answer)
```

Set `OPENAI_API_KEY` and/or `GOOGLE_API_KEY` for the VLM step (or use the bundled local Gemma-3-4B).

## Repository layout

```
rose/
  engine/         pipeline orchestration and config
  vision/         model wrappers (DA3, FastSAM, SAM3, RAM++, YOLO)
  reasoning/      STEP token encoding and patch tokenization
  models/         per-model checkpoint dirs (gitignored content; populated by setup.sh)
configs/          default + experiment YAML configs
scripts/          tests, benchmarks, setup.sh
test/, tests/     integration tests
benchmark/        VLM4D / DSR-Bench / RoboSpatial eval scripts
docs/
  roadmap/        implementation spec
  bugs/           known issues
assets/           visualization scripts and example videos
Dockerfile        reproducible build (CUDA 12.4 + torch 2.4)
requirements.txt  Python deps (excluding third-party DA3/SAM3 git installs)
```

## Tests

```bash
python scripts/test_rose_smoke.py        # fast, no model loads
python scripts/test_rose_step01.py
python test/test_fastsam_sam3_integration.py   # GPU integration
```

## Documentation

- [Implementation Spec](docs/roadmap/ROSE_IMPLEMENTATION.md) — full pipeline architecture and hyperparameters.
- [Known Issues](docs/bugs/) — SAM3 OOM, track re-identification, silent object loss, etc.

## Known issues

| Issue | Status | Doc |
|-------|--------|-----|
| SAM3 OOM on V100-32GB | Open | [SAM3_V100_OOM.md](docs/bugs/SAM3_V100_OOM.md) |
| Archived tracks not re-identified | Open | [TRACK_NO_REIDENTIFICATION.md](docs/bugs/TRACK_NO_REIDENTIFICATION.md) |
| SAM3 silent object loss | Mitigated | [SAM3_SILENT_OBJECT_LOSS.md](docs/bugs/SAM3_SILENT_OBJECT_LOSS.md) |
| CPU worker delayed error detection | Open | [CPU_WORKER_DELAYED_ERROR.md](docs/bugs/CPU_WORKER_DELAYED_ERROR.md) |
