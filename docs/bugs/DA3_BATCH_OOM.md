# DA3 Batch Inference OOM → Streaming Pipeline 重构

## 状态：Phase 1 已修复（Chunked Batch + SIM3），Phase 2 待实现（Streaming Pipeline）

## 问题描述

当视频帧数较多时（实测 51 帧 / 504px），DA3 的 `infer_batch` 在 V100-32GB 上触发 `torch.OutOfMemoryError`。

错误信息：
```
torch.OutOfMemoryError: Tried to allocate 30.22 GiB
```

## 背景：为什么必须 batch 推理

Fast-SNOW 的 4DSG 需要**跨帧一致的 3D 世界坐标系**。这依赖 DA3 提供的帧间相对位姿 `T_wc_t`：

1. DA3 batch 推理时，camera decoder 同时处理所有帧，输出帧间一致的外参 `T_wc`。
2. `da3_wrapper.py` 对 batch 结果做归一化：`T_wc[0] = I`（第 0 帧作为世界原点），其余帧表示相对于第 0 帧的相机运动。
3. 这些 `T_wc` 被用于 Step 4 的像素 → 世界坐标回投影：`p_world = inv(T_wc) @ p_cam`。
4. **所有物体的 centroid (c) 和 shape (s) 都在同一个世界坐标系下表达**，才能构建有意义的 4DSG 轨迹。

**如果改为逐帧推理**（`infer` 而非 `infer_batch`），每帧的 `T_wc` 是独立估计的，帧间位姿不一致，导致：
- 同一物体在不同帧的 centroid 无法对齐
- ego pose 序列无意义（每帧都是独立的世界坐标系）
- 4DSG 的空间推理完全失效

## 根因分析

### 1. GPU 内存模型

DA3 的 `inference()` 将所有帧**打包为单个 batch tensor** 一次性送入 GPU：

```python
# da3/src/depth_anything_3/api.py, _prepare_model_inputs()
imgs = imgs_cpu.to(device, non_blocking=True)[None].float()  # shape: (1, N, 3, H, W)
```

其中 `N` = 帧数。整个 batch 在 forward pass 期间常驻 GPU。

### 2. 内存公式

```
per_frame_mem ≈ (process_res / 504)² × 0.5 GB
total ≈ base(2GB) + N × per_frame_mem × overhead_factor
```

实际值（process_res=504）：

| 帧数 N | 估算 GPU 显存 | V100-32GB |
|--------|--------------|-----------|
| 10     | ~7 GB        | OK        |
| 20     | ~12 GB       | OK        |
| 30     | ~17 GB       | 临界      |
| 51     | ~28 GB       | OOM       |

### 3. DA3 混合注意力机制

DA3 的 backbone 使用混合注意力：
- **Layer 0-7**：局部注意力（per-frame 独立处理）
- **Layer 8+**（奇数层）：**全局注意力**（所有帧互相 attend，位姿一致性的来源）

因此 **chunk 内全局注意力 + chunk 间 SIM3 对齐** 是可行的拆分策略。

## 已完成：Chunked Batch + SIM3 对齐

### 实现概要

将 N 帧分为多个 overlapping chunk，每 chunk 独立 DA3 batch 推理，通过 SIM3 点云匹配将各 chunk 对齐到统一坐标系。

```
51 帧, chunk_size=20, overlap=5, step=15

Chunk 0: frames [0, 20)  ──DA3 batch──→ depth/K/T_wc (frame 0 = identity)
Chunk 1: frames [15, 35) ──DA3 batch──→ depth/K/T_wc (独立坐标系)
  重叠帧 15-19:
    backproject(chunk0[15:20]) vs backproject(chunk1[0:5])
      ↓ Robust SIM3 (Umeyama + IRLS Huber)
    chunk1 全部帧 × SIM3₀₁ → 对齐到 chunk 0 坐标系
Chunk 2: frames [30, 51) ──DA3 batch──→ 同上
    SIM3₀₂ = SIM3₀₁ ∘ SIM3₁₂ (累积链接)
```

### 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `fast_snow/vision/perception/da3_wrapper.py` | 新增 `infer_batch_chunked()`, `compute_chunks()`, `backproject_depth()`, `estimate_sim3()`, `estimate_sim3_robust()`, `_accumulate_sim3()`, `_apply_sim3_to_result()`, `_align_overlap()` |
| `fast_snow/engine/config/fast_snow_config.py` | `DA3Config` 新增 `chunk_size: int = 0` 和 `chunk_overlap: int = 5` |
| `test/test_da3_chunked.py` | 27 个测试 (24 单元 + 3 GPU 集成) |

### 自动路由

`infer_batch()` 根据配置自动选择路径：

```python
def infer_batch(self, images):
    if chunk_size > 0 and len(images) > chunk_size:
        return self.infer_batch_chunked(images)  # 分 chunk + SIM3
    return self._infer_batch_core(images)          # 原始全 batch
```

默认 `chunk_size=0`，向后兼容。

### 测试结果

V100-32GB 上全部通过：

| 测试 | 结果 |
|------|------|
| 单元测试：chunk 划分、SIM3 数学、backproject、result 变换 | 24/24 pass |
| GPU 集成：chunked vs full-batch 轨迹一致性 (horsing.mp4, 12 帧) | pass |
| GPU 集成：depth 有效性检查 (12 帧) | pass |
| GPU 集成：30 帧 chunk_size=8 无 OOM + 轨迹平滑 | pass |

---

## 实测延迟数据（V100-32GB, horsing.mp4 1920×1080, 30帧 @10fps）

```bash
# 复现: conda activate snow && PYTHONPATH=. python scripts/benchmark_latency.py
```

### DA3 — 深度 + 位姿估计

| 配置 | 总耗时 | 每帧耗时 |
|------|--------|---------|
| batch 5 帧 | 0.46s | 92ms |
| batch 8 帧 | 0.44s | 55ms |
| batch 10 帧 | 0.74s | 74ms |
| batch 15 帧 | 0.70s | 47ms |
| batch 20 帧 | **1.0s** | **50ms** |
| chunked 20帧 (chunk=10, overlap=3) | 4.6s | 232ms |
| chunked 30帧 (chunk=10, overlap=3) | 6.9s | 231ms |
| chunked 30帧 (chunk=15, overlap=5) | 7.4s | 246ms |

DA3 batch 本身很快（20帧 ~1s）。chunked 模式的开销主要来自**多次模型 forward + SIM3 对齐**，每帧有效耗时翻了 ~5 倍。

### FastSAM — 目标检测

| 指标 | 值 |
|------|------|
| 每帧耗时 | **~15ms** |
| 每帧检测数 | 取决于场景，class-agnostic |

**极快，可忽略。**

### SAM3 — 分割 + 跟踪（最大瓶颈）

| 阶段 | 耗时 |
|------|------|
| 模型加载 | 14.4s (一次性) |
| init (add_prompt, bbox) | **6.4s** (包含视频帧加载 1.2s) |
| **propagate 29帧** | **63.7s total** |
| **每帧 propagate** | **2.2s/frame** |
| cache lookup (已 propagate 的帧) | 0.001ms |

**SAM3 是压倒性的瓶颈：每帧 2.2 秒，比 DA3 慢 44 倍，比 FastSAM 慢 ~150 倍。**

> **注意**：SAM3 在 V100-32GB 上还有 OOM 问题，详见 [SAM3_V100_OOM.md](SAM3_V100_OOM.md)。

### CPU Pipeline — backproject + STEP + fusion

| 阶段 | 耗时 |
|------|------|
| process_frame (5 objects) | **57ms/frame** |
| 其中 backproject 单个 mask | 10.5ms |
| 其中 STEP patch tokenize | 2.3ms |

### 各模块耗时占比（30帧视频）

```
SAM3 propagate:  63.7s  ████████████████████████████████████████  88.5%
SAM3 init:        6.4s  ████                                       8.9%
DA3 batch:        1.0s  █                                          1.4%
CPU pipeline:     0.9s  █                                          1.2%
FastSAM:          0.4s                                             0.0%
                 ─────
总计:            ~72.4s
```

---

## 待实现：Streaming Pipeline 重构

### 动机

当前 pipeline 是 **全量 batch 架构**，即使有 chunk 拆分，仍然是：

```
[收完所有帧] → [DA3 全部 chunk 一口气跑完] → [FastSAM+SAM3 全部帧] → [4DSG] → [VLM]
```

对于 20 秒 @10fps = 200 帧的视频：
- 收帧 20s + DA3 全部 chunk ~7s + SAM3 propagate **~440s** = **~467s 才能得到 SG**
- 前 20 秒收帧时 GPU 完全空闲
- DA3 跑完之前 YOLO/SAM3 无法开始

### SAM3 的根本限制：propagate_in_video 只能调用一次

SAM3 内部有 `action_history` 机制，限制 `propagate_in_video` 的调用模式：
- **第一次调用**：正常 forward propagation
- **第二次调用**：SAM3 假设这是 backward pass，进入 fetch-only 模式
- **此后**：不再真正执行推理，只返回缓存结果

当前 wrapper 的应对方式（`sam3_shared_session_wrapper.py:274-353`）：**一次性 propagate 所有剩余帧**，将每帧结果缓存在 `_propagation_cache` 中，后续帧从 cache 返回。

```python
# 当前代码：第一次调用 propagate_all 时
for out in self._predictor.propagate_in_video(
    session_id=...,
    start_frame_idx=start_idx,
    max_frame_num_to_track=None,   # ← 一次性跑完 ALL 剩余帧
):
    self._propagation_cache[fid] = frame_masks
```

**这意味着 SAM3 无法按 DA3 chunk 逐步 propagate。** 如果只喂 chunk 0 的 20 帧，SAM3 会一次性 propagate 这 20 帧。当 chunk 1 的帧到达时，再次调用 `propagate_in_video` 会进入 fetch-only 模式，**不会真正跑推理**。

### 实际可行的 streaming 架构

考虑到 SAM3 的限制，streaming 不能在 SAM3 层面分 chunk，但可以在 **DA3 层面** 和 **帧收集层面** streaming。SAM3 仍然是一次性 propagate 所有帧，但我们可以把 "等所有帧收完" 这个瓶颈去掉。

#### 方案 A：DA3 streaming + SAM3 batch（推荐）

```
[收 chunk 0] → [DA3 chunk 0 | 收 chunk 1]
             → [DA3 chunk 1 | 收 chunk 2]
             → ...
             → [DA3 全部 chunk 完成，SIM3 对齐]
             → [SAM3 一次性 propagate 所有帧]  ← 仍是瓶颈
             → [SG 构建（逐帧 CPU）]
```

改进点：
- 帧收集与 DA3 推理并行（收 chunk N+1 的同时跑 DA3 chunk N）
- DA3 不再等所有帧到齐才开始

**实际时间线（20秒视频 @10fps = 200帧, chunk_size=20, overlap=5）**：

```
时段              耗时    模块
────────────────  ─────  ────────
收 chunk 0        2.0s   帧解码 (CPU)
DA3 chunk 0       1.0s   DA3 batch (GPU)     ← 同时收 chunk 1
DA3 chunk 1       1.0s   DA3 batch (GPU)     ← 同时收 chunk 2
...
DA3 chunk 12      1.0s   DA3 batch (GPU)     ← 最后一个 chunk
SIM3 全局对齐     ~0.5s  CPU
────────────────
DA3 总计:         ~13s   (13 chunks, 流水线重叠后 ≈ 帧收集20s 内完成)
────────────────
SAM3 init:         6.4s  GPU
SAM3 propagate:  440.0s  GPU  ← 200帧 × 2.2s/frame
────────────────
CPU pipeline:     ~11.4s CPU  (200帧 × 57ms, 与 SAM3 并行)
────────────────
总端到端:        ~460s   (SAM3 是绝对瓶颈)
```

DA3 streaming 把 DA3 阶段从等待 20s（收帧）+ 13s（串行 DA3）= 33s 缩短到 ~20s（与收帧重叠）。**但对端到端影响微乎其微，因为 SAM3 占了 95%+ 的时间。**

#### 方案 B：SAM3 分段重启（牺牲跟踪连续性）

为每个 DA3 chunk 独立启动一个 SAM3 session：

```
Chunk 0:  [DA3] → [FastSAM frame 0 → SAM3 init] → [SAM3 propagate frames 1-14]
Chunk 1:  [DA3] → [YOLO frame 15 → SAM3 init] → [SAM3 propagate frames 16-29]
...
→ pipeline 层面合并不同 session 的 mask（通过 cross-run fusion）
```

优点：
- 真正的 streaming：每个 chunk 独立处理，可以边收边出结果
- SAM3 propagate 只需跑 chunk 内的帧（15帧 × 2.2s = 33s per chunk）

缺点：
- **每个 chunk 的物体需要重新初始化**（FastSAM bbox → SAM3 add_prompt）
- 跨 chunk 的物体 ID 不连续，需要 cross-run fusion 重新关联
- SAM3 init 每次 ~6s 开销
- 如果 FastSAM 在某个 chunk 首帧漏检某物体，该物体在该 chunk 内完全丢失

**时间线（chunk_size=20, step=15, 200帧 = 13个 chunk）**：

```
每 chunk 延迟:
  DA3 batch:         1.0s
  FastSAM frame 0:      0.012s
  SAM3 init:         6.4s
  SAM3 propagate:   33.0s  (15帧 × 2.2s)
  ──────
  chunk 总计:       ~40.4s

首次 SG 可用: 收帧(2s) + chunk处理(40s) ≈ 42s
后续每 chunk: max(收帧 1.5s, chunk处理 40s) ≈ 40s
```

#### 方案 C：替换 SAM3（根本解决）

SAM3 的 2.2s/frame propagation 是整个 pipeline 的绝对瓶颈。如果替换为更轻量的跟踪方案：

| 方案 | 预估 per-frame | 来源 |
|------|---------------|------|
| SAM3 (当前) | 2200ms | 实测 |
| SAM2 | ~100-200ms | 公开 benchmark |
| FastSAM + ByteTrack | ~15ms | FastSAM 已有 |
| Cutie / XMem | ~50-100ms | 视频分割 |

如果 mask tracking 降到 ~100ms/frame，整个 streaming 方案就成立了：

```
首次 SG 可用:  收帧(2s) + DA3(1s) + 15帧×100ms(1.5s) ≈ 4.5s
后续每 chunk:  max(收帧 1.5s, DA3 1s) + 15帧×100ms ≈ 3s
```

### 结论与优先级

1. **SAM3 是唯一真正的瓶颈**（占 pipeline 95%+ 时间）
2. **DA3 streaming 技术上可行但收益极小**（从 33s → 20s，但总时间 460s 几乎不变）
3. **真正的 streaming 需要解决 SAM3 的两个问题**：
   - 延迟：2.2s/frame 太慢
   - action_history：限制了只能调用一次 propagate_in_video
4. **推荐路径**：先调研 SAM3 替代方案或参数调优（降低分辨率、减少物体数），再做 streaming 重构

### 与已完成工作的关系

Chunked batch + SIM3 对齐（已实现）是 streaming 的 **必要基础设施**：

```
Chunked Batch (已完成)          Streaming Pipeline (待实现)
─────────────────────          ──────────────────────────
compute_chunks()          →    帧 generator 按 chunk 产出
_infer_batch_core()       →    每 chunk 即时调用
_align_overlap()          →    即时 SIM3 对齐上一个 chunk
_accumulate_sim3()        →    增量累积（无需全部 chunk 到齐）
_apply_sim3_to_result()   →    对齐后立即推入 ready queue
```

所有 SIM3 数学和 chunk 管理代码直接复用。但 **streaming 的收益取决于 SAM3 瓶颈是否先被解决**。

### 需要修改的文件

| 文件 | 修改 | 工作量 | 前置条件 |
|------|------|--------|---------|
| `fast_snow/engine/pipeline/fast_snow_e2e.py` | `_run_pipeline()` 重构为 streaming loop | 主要工作 | SAM3 瓶颈解决后 |
| `fast_snow/vision/perception/da3_wrapper.py` | 不改（已有 `_infer_batch_core` + SIM3 工具函数） | 无 | — |
| `fast_snow/engine/pipeline/fast_snow_pipeline.py` | 不改（已是逐帧 streaming） | 无 | — |
| `fast_snow/vision/perception/sam3_shared_session_wrapper.py` | propagate 分段 或 替换 SAM3 | 重大 | 需要先调研 |
| `fast_snow/engine/config/fast_snow_config.py` | 可能新增 `streaming: bool = False` 开关 | 极小 | — |

### 复现步骤

OOM 复现（已修复）：
```bash
# 51 帧 @504px on V100-32GB, chunk_size=0 (no chunking) → OOM
PYTHONPATH=. python -c "
from fast_snow.engine.config.fast_snow_config import DA3Config
from fast_snow.vision.perception.da3_wrapper import DA3Wrapper
# ... load 51 frames, call infer_batch with chunk_size=0
"
```

Chunked batch 验证：
```bash
# 27 tests, including 3 GPU integration tests on horsing.mp4
conda activate snow
python -m pytest test/test_da3_chunked.py -v
```

延迟 benchmark：
```bash
conda activate snow
PYTHONPATH=. python scripts/benchmark_latency.py
```

## 相关文件

- `fast_snow/vision/perception/da3_wrapper.py` — `infer_batch()`, `infer_batch_chunked()`, SIM3 工具函数
- `fast_snow/engine/pipeline/fast_snow_e2e.py` — `_run_pipeline()`（当前 batch 架构，待重构为 streaming）
- `fast_snow/engine/pipeline/fast_snow_pipeline.py` — `process_frame()`（已是 streaming，无需改动）
- `fast_snow/vision/perception/sam3_shared_session_wrapper.py` — SAM3 propagate + action_history 限制
- `fast_snow/engine/config/fast_snow_config.py` — `DA3Config.chunk_size`, `DA3Config.chunk_overlap`
- `fast_snow/vision/da3/src/depth_anything_3/api.py` — DA3 底层 `inference()`
- `fast_snow/vision/da3/src/depth_anything_3/model/dinov2/vision_transformer.py` — 混合注意力
- `test/test_da3_chunked.py` — 27 个测试（chunk 划分、SIM3 数学、GPU 集成）
- `scripts/benchmark_latency.py` — 延迟实测脚本
