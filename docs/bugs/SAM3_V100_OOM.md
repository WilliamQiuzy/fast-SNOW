# SAM3 在 V100-32GB 上 Propagation OOM

## 状态：Open

## 问题描述

SAM3 的 `propagate_in_video` 在 V100-SXM2-32GB 上触发 `torch.OutOfMemoryError`。
OOM 发生在 SAM2 tracker 的 cross-attention 层，需要分配 ~13 GiB，
但 SAM3 模型自身已占用 ~23 GiB，剩余空间不足。

```
torch.OutOfMemoryError: Tried to allocate 13.22 GiB.
GPU 0 has a total capacity of 31.73 GiB of which 8.77 GiB is free.
This process has 22.96 GiB memory in use.
```

Peak 显存需求 ~29 GiB（实测成功运行时 `max over time: 28780 MiB`），
几乎吃满 V100 的 32 GiB。任何额外的显存占用（其他进程、未释放的模型）
都会导致 OOM。

## 复现条件

- GPU: V100-SXM2-32GB（或任何 <40GB 显存的 GPU）
- 视频: bear.mp4 (854×480)，3 帧
- FastSAM: `max_det=200`（默认），frame 0 检测到 ~20+ 个 instance
- SAM3: `image_size=1008`（硬编码默认值）

```bash
conda activate snow
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python test/test_fastsam_sam3_integration.py
```

## 根因分析

### OOM 调用链

```
fast_snow_e2e.py: _run_pipeline()
  → sam3_shared_session_wrapper.py: propagate_all()
    → sam3_video_predictor.py: propagate_in_video()
      → sam3_video_inference.py: propagate_in_video() [full mode]
        → sam3_video_inference.py: _run_single_frame_inference()
          → sam3_video_base.py:203 _det_track_one_frame()
            → sam3_video_base.py:415 run_tracker_propagation()      ← Step 2
              → sam3_video_base.py:1118 _propogate_tracker_one_frame_local_gpu()
                → sam3_tracking_predictor.py:847 propagate_in_video()   ← SAM2 tracker
                  → sam3_tracker_base.py:970 track_step()
                    → sam3_tracker_base.py:782 _prepare_memory_conditioned_features()
                      → decoder.py:697 transformer.encoder()
                        → decoder.py:909 cross_attn_image()
                          → transformer.py:354 F.scaled_dot_product_attention()  ← OOM
```

OOM 不在 SAM3 的检测器（Step 1），而在 **SAM2 tracker 的 cross-attention**（Step 2）。

### 显存去向

SAM3 的 `_det_track_one_frame` 每帧执行 5 步流水线，其中 Step 2（tracker propagation）
调用内嵌的 SAM2 tracker。SAM2 tracker 用 memory bank 做 cross-attention 预测 mask。

| 组件 | 估算显存 | 说明 |
|------|---------|------|
| SAM3 detection model 权重 | ~7 GiB | 3.4GB safetensors + 运行时 buffer |
| SAM2 tracker model 权重 | ~8 GiB | SAM3 内嵌的 SAM2 tracker |
| Video features cache | ~4 GiB | 5 帧 × 1008² 的 backbone features |
| Memory bank + tracker state | ~3 GiB | per-object memory tokens |
| **Cross-attention 中间计算** | **~13 GiB** | **Q × K^T attention matrix** |
| **Total peak** | **~29 GiB** | |

### 为什么 cross-attention 需要 13 GiB

SAM3 将所有视频帧 resize 到 **1008×1008** 处理（硬编码在 `sam3_video_inference.py:36`）。

- patch_size = 16 → 每帧 63×63 = **3,969 个空间 token**
- memory bank 默认存 7 帧 → K/V 约 3,969 × 7 = **27,783 token**
- cross-attention: Q(3969) × K(27783) × num_heads × sizeof(float32)
- 加上 softmax 中间值、V projection → **~13 GiB**

### 已有的缓解措施

Pipeline 在各阶段之间释放不需要的模型：

```python
# fast_snow_e2e.py
da3_results = self._da3.infer_batch(frames)
self._da3.unload()              # Phase 1 后释放 DA3 (~1 GiB)

fastsam_dets_0 = self._fastsam.detect(frames[0])
self._fastsam.unload()          # Phase 2a init 后释放 FastSAM (~0.5 GiB)
# → SAM3 full propagation

# Phase 2b: FastSAM 重新 lazy load 做 discovery
self._fastsam.unload()          # Phase 2b 后再次释放
# → SAM3 partial propagation
```

这些 unload 释放了 ~1.5 GiB，但不够——SAM3 自身需要 ~29 GiB peak。

### 在 max_det=5 时能跑通的原因

限制 FastSAM 到 5 个 instance 后，SAM3 只追踪 5 个物体。
tracker 的 memory bank 更小，cross-attention 的 K/V 规模降低，
peak 从 ~29 GiB 降到刚好能塞进 32 GiB。

## 解决方案分析

### 方案 A：float16 推理

SAM3 默认使用 float32。改为 `model.half()` 或 `torch.autocast(dtype=torch.float16)`
可以将几乎所有 tensor 大小减半。

- **预估效果**：peak 从 ~29 GiB 降到 ~15 GiB，V100 32GB 完全充裕
- **修改范围**：`sam3_shared_session_wrapper.py` 的 `load()` 中加 `.half()`
  或在 `propagate_all` / `propagate_new_objects` 中包裹 `torch.autocast`
- **风险**：
  - SAM2 官方支持 fp16（SAM2 demo 默认用 `torch.bfloat16`），SAM3 未明确文档化
  - 检测器的分数精度可能略有变化（score_threshold 可能需要微调）
  - V100 不支持 bfloat16，必须用 float16

### 方案 B：降低 SAM3 处理分辨率

将 `image_size` 从 1008 降到 512 或 672。

```
image_size=1008: 63×63=3969 tokens → ~29 GiB peak
image_size=672:  42×42=1764 tokens → ~15 GiB peak (估算)
image_size=512:  32×32=1024 tokens → ~10 GiB peak (估算)
```

- **修改范围**：
  1. `SAM3Config` 加 `image_size: int = 1008`
  2. `Sam3VideoPredictor.__init__` 需要接受 `image_size` 参数并传给
     `build_sam3_video_model()`
  3. 或直接改 `model_builder.py` 中的默认值
- **风险**：
  - SAM3 在低分辨率上的分割精度未知（mask 边界可能更粗糙）
  - `model_builder.py` 中的 1008 可能与预训练权重的 positional encoding 绑定，
    降分辨率可能需要 interpolate positional embeddings

### 方案 C：A + B 组合

float16 + 降分辨率。V100 32GB 上可以同时支持高 max_det 和较多帧。

### 方案 D：只降 tracker 分辨率

SAM3 内部 detection 和 tracker 是两个独立模型。OOM 在 tracker 侧。
如果能只降 tracker 的 `image_size`（detection 保持 1008 高质量），
可以在不损失检测精度的情况下降低 tracker 的显存占用。

- **风险**：需要深入修改 SAM3 源码，detection 和 tracker 的 feature map 对齐可能受影响

## 推荐优先级

1. **方案 A（float16）**——改动最小，不损分辨率，V100 有 fp16 Tensor Core 加速
2. **方案 B（降分辨率）**——如果 A 不够或 fp16 不稳定，作为备选
3. **方案 C（组合）**——在更小的 GPU（T4 16GB）上需要

## 相关文件

- `fast_snow/vision/perception/sam3_shared_session_wrapper.py:71-116` — SAM3 模型加载
- `fast_snow/vision/perception/sam3_shared_session_wrapper.py:282-361` — propagate_all() OOM 发生处
- `fast_snow/vision/sam3/sam3/model/sam3_video_inference.py:36` — `image_size=1008` 硬编码
- `fast_snow/vision/sam3/sam3/model/model_builder.py:453,740,767` — `image_size=1008` 在 builder 中
- `fast_snow/vision/sam3/sam3/model/sam3_tracker_base.py:782` — `_prepare_memory_conditioned_features` OOM 具体位置
- `fast_snow/vision/sam3/sam3/sam/transformer.py:354` — `F.scaled_dot_product_attention` OOM 最终 call site
- `fast_snow/engine/pipeline/fast_snow_e2e.py:204-229` — model unload 缓解措施
- `fast_snow/vision/perception/da3_wrapper.py:82-91` — DA3 unload
- `fast_snow/vision/perception/fastsam_wrapper.py:55-64` — FastSAM unload
- `test/test_fastsam_sam3_integration.py` — GPU 集成测试（复现 OOM）
- [DA3_BATCH_OOM.md](DA3_BATCH_OOM.md) — 相关：DA3 的 OOM 问题（已修复）
