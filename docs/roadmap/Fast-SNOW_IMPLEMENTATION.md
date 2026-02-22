# Fast-SNOW Implementation Spec (v2 — FastSAM + SAM3 Two-Pass)

## 1. Objective

目标：比 SNOW 更快，同时保持可接受准确率。

核心思路：用 FastSAM（class-agnostic 分割）+ SAM3（视频追踪）的 **two-pass 架构**替代 SNOW 的 HDBSCAN 聚类 + SAM2 点提示分割 + 匈牙利匹配追踪，跳过昂贵的 3D 点云聚类步骤。

### 版本历史

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-02-20 | v1 | YOLO11n bbox → SAM3 单活跃 run，per-frame 串行推理 |
| 2026-02-22 | **v2** | FastSAM 替代 YOLO，two-pass 架构，DA3 batch 推理，mid-video discovery |

### v2 决策要点

- **FastSAM 替代 YOLO**：FastSAM 输出 class-agnostic instance masks（不限于 COCO 类别），天然适配 SAM3 的 bbox prompt。
- **Two-pass 替代 re-anchor**：v1 依赖破坏性 re-anchor（`add_prompt()` 触发 `reset_state()`），v2 使用 `add_tracker_new_points()` 在不清空 tracker 状态的情况下添加新物体。
- **DA3 batch 推理**：DA3 一次性处理所有帧，产出帧间一致的位姿和深度图（`consistent video poses`），不再逐帧串行。
- **GPU 显存分时管理**：各阶段之间 unload 不再需要的模型，避免 DA3 + FastSAM + SAM3 同时驻留。

约束：
- 以跟踪稳定性和速度优先，不追求完备语义。
- 节点存紧凑观测，不存整帧点云。
- 不依赖 class label 驱动分割；语义由 VLM 在推理时联合 visual_anchor + STEP token 推断。
- SG 节点严格遵循 SNOW 论文的 STEP token set 定义 `S_t^k = {τ, c, s, θ}`（见 §4.1）：
  - **τ（patch token）= `(row, col, iou)` 网格坐标元数据**，表示该对象 mask 与 16×16 网格 cell 的重叠程度。
  - c, s, θ 作为 text token 序列化。
  - **节点 = F_k = 跨帧 STEP 序列**（论文 Eq. 7），不是单帧快照。
  - F_k 窗口内全帧保留，float 保持原始精度。
- SG（4DSG）**不构建 edge**（ego-object / object-object）。空间关系由 VLM 从 ego_poses + centroid 轨迹推导。

与 SNOW 的对比：

| | SNOW | Fast-SNOW v2 |
|---|------|------|
| object discovery | HDBSCAN 3D 聚类 → 采样点 | FastSAM class-agnostic instance segmentation |
| 分割 prompt | SAM2 point prompt | SAM3 bbox prompt (init) + point prompt (discovery) |
| 跨帧追踪 | 匈牙利匹配（STEP geometric + semantic cost） | SAM3 内置 memory bank + two-pass discovery |
| 新物体发现 | 每帧重新聚类 | Phase 2b: FastSAM per-frame vs SAM3 cache IoU 比较 |
| 3D 来源 | LiDAR 点云 / MapAnything 重建 | DA3 单目深度估计（batch，帧间一致位姿） |
| 迭代细化 | N_iter 轮 HDBSCAN + H_hop 几何校验 | 不做（SAM3 mask 质量更稳定） |
| SG 节点 | STEP token set `S_t^k = {τ, c, s, θ}` | 完全相同 |
| 4DSG 节点 | `F_k = {S_{t-T}^k, ..., S_t^k}` 跨帧堆叠 | 完全相同 |

---

## 2. Pipeline Architecture

### 整体流程

```
Video
 │
 ├─ Step 0: Frame Sampling (target_fps + max_frames → N RGB frames)
 │
 ├─ Phase 1 [GPU, batch]: DA3 → Metric Depth Maps + Consistent Video Poses + K
 │  └─ DA3 unload after completion
 │
 ├─ Phase 2 [GPU, two-pass]:
 │  ├─ 2a: FastSAM(frame 0) → bbox prompts → SAM3 init → full propagation (cache all frames)
 │  │      └─ FastSAM unload before SAM3 propagation
 │  ├─ 2b: FastSAM(frame 1..N) → discover new objects via IoU vs SAM3 cache
 │  │      → SAM3.add_object_point(centroid) for each new object
 │  │      └─ FastSAM unload after discovery
 │  ├─ 2c: SAM3 partial propagation (new objects only, merge with cache)
 │  └─ 2d: Read merged cache → build FastFrameInput → CPU worker queue
 │
 └─ Phase 3 [CPU, streaming, parallel with Phase 2d]:
    ├─ Step 4: Mask + Depth → 3D back-projection
    ├─ Step 5: Candidate Fusion (global ID management)
    ├─ Step 6: Build STEP Tokens
    ├─ Step 7: Build 4DSG (temporal tracks F_k)
    └─ Step 8: Serialize + VLM Inference (on demand)
```

### GPU 显存分时管理

各模型不同时驻留 GPU，通过 `unload()` 释放：

```
时间 →
┌──────────┐
│   DA3    │ Phase 1: batch inference
└──────────┘
     unload()
         ┌────────┐
         │ FastSAM │ Phase 2a: detect frame 0
         └────────┘
              unload()
              ┌─────────────────────────────────────────┐
              │              SAM3                         │ Phase 2a: full propagation
              │              (cache all frames)           │
              └─────────────────────────────────────────┘
                   ┌────────┐
                   │ FastSAM │ Phase 2b: per-frame discovery
                   └────────┘
                        unload()
                        ┌──────────────────────┐
                        │ SAM3 partial propagate│ Phase 2c
                        └──────────────────────┘
                        ┌──────────────────────┐
                        │ SAM3 cache read       │ Phase 2d
                        └──────────────────────┘
```

### CPU Worker 线程模型

Phase 2d 和 Phase 3 通过 producer-consumer 队列并行：

```
GPU 线程 (Phase 2d):  [read cache f0] [read cache f1] ... [read cache fN] [sentinel]
                       put(fi_0)       put(fi_1)            put(fi_N)
                          ↓               ↓                     ↓
CPU Worker (Phase 3):  [process f0]   [process f1]   ...   [process fN]
```

已知问题：CPU worker 异常延迟检测（见 `docs/bugs/CPU_WORKER_DELAYED_ERROR.md`）。

---

## 3. Runtime Pipeline

### Step 0. Sampling & Scheduling
- 输入：视频文件。
- 默认按 `target_fps=10.0`（10 Hz）做时间采样。
- `max_frames` 可选：用于截断窗口长度。
- 输出：`N` 帧 RGB 图像 + 对应时间戳 + source frame indices。
- 同时将每帧保存为 JPEG 到临时目录（供 SAM3 `set_video_dir` 使用）。

### Phase 1. DA3 Batch Inference

DA3（Video-Depth-Anything v3）对所有帧做 **batch 推理**：

- 输入：`N` 帧 RGB 图像
- 输出（每帧）：
  - `depth_t`：`(H, W)` 深度图
  - `K_t`：`(3, 3)` 相机内参
  - `T_wc_t`：`(4, 4)` world-to-camera 外参
  - `depth_conf_t`：`(H, W)` 逐像素置信度
- **Batch 的优势**：帧间位姿一致（`consistent video poses`），DA3 在 batch 模式下利用帧间关系产出时序一致的相机轨迹。
- **OOM 防护**：当帧数超过 `chunk_size` 时，自动分 chunk 推理并用 SIM3 对齐（见 `docs/bugs/DA3_BATCH_OOM.md`）。
- **Phase 1 完成后**：调用 `da3.unload()` 释放 GPU 显存。

坐标变换约定：
- `T_wc_t`（DA3 直接输出）：world → camera，即 `p_cam = T_wc_t @ p_world`。
- `T_cw_t = inv(T_wc_t)`：camera → world，即 `p_world = T_cw_t @ p_cam`。
- **Step 4 回投影使用 `T_cw_t`**（先像素→相机坐标，再 camera→world）。
- **Ego pose** = `T_cw_t`（相机在世界坐标系中的位姿）。

### Phase 2. Two-Pass FastSAM + SAM3

#### Phase 2a: Init + Full Propagation

1. **FastSAM 检测 frame 0**：`FastSAM.detect(frames[0])` → class-agnostic instance masks + bboxes。
2. **SAM3 初始化**：将所有 FastSAM bbox 一次性传入 `create_run_with_initial_bboxes(boxes_xywh=[...])` 创建一个 run。SAM3 为每个 bbox 生成 mask 并分配独立 `obj_id`。
3. **FastSAM unload**：释放 FastSAM GPU 显存，为 SAM3 full propagation 腾出空间。
4. **Full propagation**：对所有帧调用 `propagate_all(fidx)`，SAM3 执行 `propagate_in_video` 并将结果缓存到 `_propagation_cache`。后续读取从缓存返回。

#### Phase 2b: Discovery

逐帧运行 FastSAM，与 SAM3 缓存的 mask 做 IoU 比较，发现新物体：

```python
for fidx in range(1, len(frames)):
    fastsam_dets = fastsam.detect(frames[fidx])
    cached_masks = sam3.propagate_all(fidx)  # 读缓存
    for det in fastsam_dets:
        if not any_mask_iou_above(det.mask, cached_masks, discovery_iou_thresh):
            # 新物体！取 mask 质心作为 point prompt
            cy, cx = mask_centroid(det.mask)
            sam3.add_object_point(fidx, (cx, cy))
```

**关键**：`add_object_point()` 内部调用 SAM3 的 `add_tracker_new_points()`，**不会** `reset_state()`——已有的 tracker 状态（memory bank、所有旧物体的 mask）保持完好。这是 v2 架构相比 v1 re-anchor 的核心改进。

Discovery 完成后调用 `fastsam.unload()` 释放显存。

#### Phase 2c: Partial Propagation

如果 Phase 2b 发现了新物体：

```python
sam3.propagate_new_objects()
```

这触发 SAM3 的 `propagation_partial` 模式：
- 只对新添加的物体执行 propagation
- 自动与缓存的旧物体 mask 合并
- 更新 `_propagation_cache` 中所有帧的结果

#### Phase 2d: Assembly → CPU Pipeline

从 SAM3 缓存读取每帧的合并 mask，构建 `FastFrameInput`，送入 CPU worker 队列：

```python
for sam3_idx, image in enumerate(frames):
    frame_masks = sam3.propagate_all(sam3_idx)  # 读更新后的缓存
    fi = build_frame_input(image, da3_results[sam3_idx], frame_masks, ...)
    cpu_queue.put(fi)
```

### Step 4. Mask + Depth → 3D Object State (每帧每对象)
- 输入：`mask_{i,t}`（来自 SAM3 cache）、`depth_t, K_t, T_cw_t = inv(T_wc_t)`（来自 Phase 1）
- 回投影公式（像素 → 世界坐标）：
  ```
  对 mask 内每个像素 (u, v)：
  1. 读取深度：d = depth_t[v, u]
  2. 像素 → 相机坐标：p_cam = d * K_t^{-1} @ [u, v, 1]^T
  3. 相机 → 世界坐标：p_world = T_cw_t @ [p_cam; 1]    （注意：用 T_cw，不是 T_wc）
  ```
  得到对象 3D 点集 `P_{i,t}`。
- 深度置信度过滤：回投影前，丢弃 `depth_conf_t[v, u] < conf_thresh`（默认 0.5）的像素。
- 计算（供 Step 6 使用）：
  - `centroid_xyz = mean(P_{i,t})`
  - 每轴统计 `(μ, σ, min, max) × 3` — 直接构成 shape token
  - `num_points = |P_{i,t}|`
- 过滤：`num_points < min_points` 或单轴 extent > `max_extent` 的对象直接丢弃。

### Step 5. Global ID Management

目标：将 SAM3 run 内的本地 `obj_id_local` 映射为全局轨迹 ID `k`。

**v2 中的简化**：Two-pass 架构下通常只有一个 run（Phase 2a 创建），新物体通过 `add_object_point` 加入同一 run，不需要 re-anchor。Global ID 管理主要处理：

1. **单 run 内 ID 映射**：`(run_id, obj_id_local)` → 全局 `k`，SAM3 memory bank 保证同 run 内 obj_id 跨帧一致。
2. **跨 run mask 去重**（边界情况）：保留 IoU + centroid + temporal gap 三重 gate 的合并逻辑，用于鲁棒性。

全局 ID 生命周期：
- `active`：当前帧有 mask 输出。
- `lost`：连续 `lost_patience`（默认 5）帧无 mask → 标记 lost。
- `archived`：lost 状态持续 `archive_patience`（默认 30）帧 → archived，不再参与匹配。
- **不做 re-identification**：archived 对象不会被重新激活（见 `docs/bugs/TRACK_NO_REIDENTIFICATION.md`）。

**缓解**：Phase 2b 的 FastSAM discovery 大幅降低了物体被静默丢失的概率——即使 SAM3 tracker 暂时丢失某个物体，FastSAM 在后续帧中检测到它后会通过 point prompt 重新添加（见 `docs/bugs/SAM3_SILENT_OBJECT_LOSS.md`）。

### Step 6. Build STEP Tokens (每帧每对象)
对每个全局对象 `k` 在帧 `t` 的观测，构建 STEP token（严格遵循 SNOW 论文 Section 3.2, Eq. 4）：

```
S_t^k = {τ_{k,1}^t, ..., τ_{k,m}^t,  c_t^k,  s_t^k,  θ_t^k}
```

1. **patch_tokens (τ) — 网格索引元数据**：
   ```
   a) 使用原始 frame 与 mask 计算 16×16 网格上每个 cell 与 mask 的 IoU。
   b) 共 256 个 cell，保留 IoU > 0.5 的 cell。
   c) 每个保留 cell 输出 `(row, col, iou)`，形成 `tau`。
   d) 不生成图像 crop 或 embedding，仅保留结构化坐标 token。
   ```
2. **centroid_token (c)**：直接取 Step 4 计算的 `centroid_xyz = (x̄, ȳ, z̄)`，作为文本 token。
3. **shape_token (s)**：直接取 Step 4 计算的每轴统计 `(μ, σ, min, max) × 3`，共 12 维，作为文本 token。
4. **temporal_token (θ)**：初始 `(t, t)`；Step 7 中按 track 更新为 `(t_start, t_end)`，作为文本 token。

**关键**：节点不附加 class label 或任何外部语义标签。语义靠 VLM 在 `visual_anchor` 与文本 token 结合下推理（STEP open-world 设计）。

### Step 7. Build 4DSG

**Temporal tracks** `F_k`（论文 Eq. 7 — 跨帧 STEP token 堆叠）：
- 基于 Step 5 的全局 ID，将同一对象跨帧的 STEP token 串联：`F_k = {S_{t-T}^k, ..., S_t^k}`。
- **F_k 是 4DSG 节点的核心表示**——一个物体节点不是单帧快照，而是 T 帧内的完整 STEP token 序列。
- **窗口内全帧保留**：不做观测截断，track 在窗口 `[t-T, t]` 内每个可见帧的完整 `S_t^k` 都保留在 `F_k` 中。
- 更新所有 `S^k` 的 temporal token `θ` 为 track 级：`(t_start, t_end) = (min frame, max frame)`。

**4DSG** `M_t`（论文 Eq. 8）：
- `M_t = ({F_k}, ego_poses)`
- 附加 ego pose 序列（Phase 1 的 `T_cw_t = inv(T_wc_t)` 逐帧提取）。
- **不预计算边（relations）**：VLM 直接从 ego pose 序列 + object centroid 序列推导空间/时序关系。

### Step 8. Serialize + VLM Inference（论文 Eq. 9）

将 4DSG 构造为 **文本主干 + 关键帧图像** 的 VLM prompt：`tau_grid` 与 `c/s/θ` 为文本 token，`visual_anchor` 提供关键帧原图上下文。

#### 8.1 STEP Token 的文本序列化

**单个 `S_t^k` 的 VLM 输入构造**：

```text
Object k, frame t=12:
  tau: [(3,4,0.71), (3,5,0.66), (4,4,0.82)]
  c: [5.12, -2.30, 1.05]
  s: {x: {mu:5.12, sigma:1.12, min:2.90, max:7.40},
      y: {mu:-2.30, sigma:0.45, min:-3.10, max:-1.60},
      z: {mu:1.05, sigma:0.36, min:0.20, max:1.70}}
  theta: [3, 18]
```

#### 8.2 F_k 跨帧序列化（论文 Eq. 7）

每个物体的 `F_k` 按时间顺序堆叠所有帧的 STEP token：

```
=== Object 0 (F_k, 7 frames) ===
  t=3: tau=[(3,4,0.71),(3,5,0.66)] c=[5.12,-2.30,1.05] s={...} theta=[3,8]
  t=4: tau=[(3,4,0.70),(4,4,0.61)] c=[5.15,-2.28,1.04] s={...} theta=[3,8]
  ...
  t=8: tau=[(3,4,0.68),(3,5,0.72)] c=[5.20,-2.25,1.06] s={...} theta=[3,8]

=== Object 1 (F_k, 5 frames) ===
...
```

#### 8.3 完整 VLM Prompt 结构

```
[SYSTEM] You are a spatial reasoning agent. Analyze the 4DSG below.
[METADATA] grid: 16x16, frames: 10, tracks: 5, coord_system: ...
[EGO POSES] t=0: xyz=[...], t=1: xyz=[...], ...

[OBJECT 0 — F_k across T frames]
  Frame 3: {"tau":[(3,4,0.71),(3,5,0.66)], "c":[...], "s":{...}, "theta":[3,8]}
  ...
[OBJECT 1 — F_k across T frames]
  ...

[Keyframe images: frame 0, frame 1, ..., frame N]

[QUERY] q
```

**推理**（论文 Eq. 9）：`ŷ = VLM(q | M_t)`

VLM 支持两个 provider：
- `"openai"`: OpenAI API (GPT-5.2 等)
- `"google"`: Google genai API (Gemini 等)

#### 8.4 持久化存储格式

**4dsg.json Schema**：
```json
{
  "metadata": {
    "grid": "16x16",
    "num_frames": 10,
    "num_tracks": 5,
    "coordinate_system": "World frame = first frame camera. X=right, Y=down, Z=forward.",
    "visual_anchor": [
      {"frame_idx": 0, "path": "/tmp/fast_snow_frames_xxx/000000.jpg"},
      {"frame_idx": 3, "path": "/tmp/fast_snow_frames_xxx/000001.jpg"}
    ]
  },
  "ego": [
    {"t": 0, "xyz": [1.23, 4.56, 0.78]},
    {"t": 1, "xyz": [1.25, 4.60, 0.78]}
  ],
  "tracks": [
    {
      "object_id": 0,
      "F_k": [
        {
          "t": 3,
          "tau": [{"row":3, "col":4, "iou":0.71}, {"row":3, "col":5, "iou":0.66}],
          "c": [5.12, -2.30, 1.05],
          "s": {"x": {"mu":5.12, "sigma":1.12, "min":2.90, "max":7.40}, ...},
          "theta": [3, 8]
        }
      ]
    }
  ]
}
```

#### 8.5 序列化参数

**不做截断 / 不做量化**（严格 STEP 合规）：
- **F_k 不截断**：窗口内每帧 `S_t^k` 全保留，不做子采样。
- **float 不量化**：所有浮点数保持 float32 原始精度序列化。
- 不在 SG 中序列化图像 crop，`tau` 仅保留 `(row, col, iou)` 文本。

#### 8.6 Token 预算估算

每个 `S_t^k` 的 VLM token 构成：

| 组件 | 类型 | 估算 |
|------|------|------|
| `tau` 坐标 token（~8 cells） | text | ~30 text tokens |
| `c/s/theta` + JSON 结构 | text | ~154 text tokens |
| **单个 S_t^k 小计** | text | **~184 tokens** |

全 4DSG token 预算（N_tracks=30, 窗口 10 帧, ~210 observations）：

| 组件 | 估算 |
|------|------|
| metadata + ego | ~800 text tokens |
| tracks `S_t^k`（210 obs × ~184） | ~38,640 text tokens |
| **合计** | **~39,440 text tokens** |

---

## 4. SG Schema (v1)

### 4.1 Node = F_k = 跨帧 STEP Token 序列（严格遵循论文 Eq. 4-5-7）

**4DSG 节点不是单帧观测，而是同一物体在 T 帧内的 STEP token 序列** `F_k`（论文 Eq. 7）。

单帧 STEP token set（论文 Eq. 4）：
```
S_t^k = {τ_{k,1}^t, ..., τ_{k,m}^t,  c_t^k,  s_t^k,  θ_t^k}
         ├─ tau_tokens (row,col,iou) ─┘     centroid  shape   temporal
```

物体节点（论文 Eq. 7）：
```
F_k = {S_{t-T}^k, ..., S_t^k}    ← T 帧的 STEP token 堆叠
```

| STEP 组件 | 模态 | 内部表示 | JSON/VLM 序列化 | 说明 |
|-----------|------|----------|-----------------|------|
| **τ** (patch) | text | `PatchToken(row: int, col: int, iou: float)` | `{"row": r, "col": c, "iou": 0.xx}` | 16×16 grid 的稀疏 token |
| **c** (centroid) | text | `CentroidToken(x, y, z)` | `c: [x, y, z]` | 3D 质心 |
| **s** (shape) | text | `ShapeToken(μ, σ, min, max × 3)` | `s: {x: {mu,sigma,min,max}, ...}` | 每轴 Gaussian 统计，共 12 维 |
| **θ** (temporal) | text | `TemporalToken(t_start, t_end)` | `theta: [t_start, t_end]` | track 级时间跨度 |

**设计原则**：
- τ 为网格坐标语义 token，视觉信息由 `visual_anchor`（关键帧图像）承载。
- 节点 = F_k = 跨帧序列。VLM 看到同一物体的多帧变化 + 3D 轨迹演化。
- JSON 字段名直接使用论文符号（`tau`, `c`, `s`, `theta`），不做重命名。
- 不存 semantic_tag——语义由 VLM 在推理时联合 visual_anchor 与 tau/c/s/θ 推理。

### 4.2 Edge（已移除）

Fast-SNOW 不预计算边。VLM 在推理时直接使用 `ego[].xyz` 和 `tracks[].F_k[].c` 推导空间/时序关系。

### 4.3 视频帧
- 不作为 SG 节点。
- 作为 `visual_anchor` 元数据（关键帧图像路径 + frame_idx），供多模态 VLM 参考。

---

## 5. Storage Rules (必须遵守)

### 5.1 代码仓库放置规范

1. 所有 Fast-SNOW 自身代码统一放在 `fast_snow/` 包下。
2. 需要从 GitHub clone 的视觉模型源码统一放在 `fast_snow/vision/` 下的对应子目录。

```text
fast_snow/
  vision/
    sam3/              # SAM3 源码
    da3/               # Video-Depth-Anything v3 源码
    Video-Depth-Anything/  # DA3 依赖
    recognize-anything/    # RAM++ (legacy, 不在主路线)
  perception/
    da3_wrapper.py         # DA3 推理封装
    fastsam_wrapper.py     # FastSAM 推理封装
    sam3_wrapper.py        # SAM3 session 管理（支持 bbox + point prompt）
    sam3_shared_session_wrapper.py  # SAM3 shared session（two-pass 核心）
```

### 5.2 模型权重与缓存放置规范

```text
fast_snow/
  models/
    sam3/              # SAM3 checkpoint
    fastsam/           # FastSAM-s.pt
    da3-small/         # DA3 small variant
    hf_cache/          # Hugging Face cache
```

`fast_snow/models/` 下的二进制权重默认不纳入 Git 版本管理。

### 5.3 路径规范

代码、配置、脚本一律使用项目根目录相对路径。

### 5.4 文档归档规范

- `docs/roadmap/`：路线与架构规划文档
- `docs/bugs/`：已知问题与 bug 追踪
- `docs/reference/`：参考材料与论文笔记

### 5.5 STEP/4DSG 存储一致性

- 不存每帧完整点云。
- F_k 不截断。
- 不做序列化量化。
- `tau` 只保留 `(row, col, iou)` 文本字段。

---

## 6. Hyperparameter Config（集中管理）

### 6.0 采样与调度

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sampling.target_fps` | 10.0 (Hz) | 时间采样频率 |
| `sampling.max_frames` | `None` | 可选帧数上限 |

### 6.1 FastSAM 检测

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fastsam.model_path` | `fast_snow/models/fastsam/FastSAM-s.pt` | FastSAM 模型路径 |
| `fastsam.conf_threshold` | 0.55 | 检测置信度阈值 |
| `fastsam.iou_threshold` | 0.9 | NMS IoU 阈值 |
| `fastsam.imgsz` | 640 | 推理输入图像尺寸 |
| `fastsam.max_det` | 200 | 单帧最大检测数 |
| `fastsam.discovery_iou_thresh` | 0.3 | Phase 2b: FastSAM mask vs SAM3 cache，低于此值 = 新物体 |

### 6.2 SAM3 追踪

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sam3.model_path` | `fast_snow/models/sam3` | SAM3 模型路径 |
| `sam3.score_threshold_detection` | 0.3 | SAM3 detector 输出的最低 score |
| `sam3.offload_state_to_cpu` | True | 将 tracker 状态卸载到 CPU 减少 GPU 占用 |
| `sam3.offload_video_to_cpu` | True | 将视频 features 卸载到 CPU |

### 6.3 DA3 深度估计

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `da3.model_path` | `fast_snow/models/da3-small` | DA3 模型路径/变体 |
| `da3.process_res` | 504 | 推理分辨率 |
| `da3.chunk_size` | 0 | 批量推理分块大小，0=不分块 |
| `da3.chunk_overlap` | 5 | 分块间重叠帧数（SIM3 对齐用） |

### 6.4 深度 & 3D 过滤

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `depth_filter.conf_thresh` | 0.5 | 深度置信度过滤阈值 |
| `depth_filter.min_points` | 50 | 最少 3D 点数 |
| `depth_filter.max_extent` | 30.0 (m) | 单轴 extent 上限 |

### 6.5 Mask 去重 & ID 融合

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fusion.cross_run_iou_thresh` | 0.5 | 跨 run mask IoU 阈值 |
| `fusion.merge_centroid_dist_m` | 2.0 | centroid 距离上限（米） |
| `fusion.merge_temporal_gap` | 2 | 最近观测帧差上限 |
| `fusion.lost_patience` | 5 (帧) | active → lost |
| `fusion.archive_patience` | 30 (帧) | lost → archived |

### 6.6 STEP Token

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `step.grid_size` | 16 | patch token 网格尺寸 |
| `step.iou_threshold` | 0.5 | 网格 cell 覆盖率阈值 |
| `step.max_tau_per_step` | 0 | top-k patches per STEP token，0=不限 |
| `step.temporal_window` | 10 | F_k 滑动窗口大小 |

### 6.7 VLM

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vlm.provider` | `"openai"` | `"openai"` 或 `"google"` |
| `vlm.model` | `"gpt-5.2"` | 模型名称 |
| `vlm.max_output_tokens` | 1024 | 最大输出 token |
| `vlm.temperature` | 1.0 | 温度 |

---

## 7. Known Issues

| 问题 | 状态 | 文档 |
|------|------|------|
| SAM3 V100-32GB OOM | Open | [SAM3_V100_OOM.md](../bugs/SAM3_V100_OOM.md) |
| 归档 Track 无法重新识别 | Open | [TRACK_NO_REIDENTIFICATION.md](../bugs/TRACK_NO_REIDENTIFICATION.md) |
| SAM3 连续空帧静默丢失 | Mitigated | [SAM3_SILENT_OBJECT_LOSS.md](../bugs/SAM3_SILENT_OBJECT_LOSS.md) |
| CPU Worker 异常延迟检测 | Open | [CPU_WORKER_DELAYED_ERROR.md](../bugs/CPU_WORKER_DELAYED_ERROR.md) |
| DA3 Batch OOM | Fixed | [DA3_BATCH_OOM.md](../bugs/DA3_BATCH_OOM.md) |

---

## 8. Acceptance Criteria

评测协议（固定）：
- 数据：使用固定清单 `configs/eval/fast_snow_eval_manifest.json`。
- 硬件：单卡 H200（同机型）记录端到端延迟。
- 基线：在同一清单上运行 SNOW 基线并保存结果。

验收条件：
1. 全局 ID 唯一性：同一窗口内重复全局 ID 率为 0。
2. 追踪稳定性：报告 `IDF1`、`ID switches / 1k frames`、`fragmentation / 1k frames`。
3. VLM 精度：同一问答集上，Fast-SNOW 相对 SNOW 的 absolute accuracy 下降不超过 3%。
4. 速度：报告 `fps` 与 `p50/p95 latency`，显著优于 SNOW。
5. 可复现性：固定随机种子、模型版本和配置文件，重复 3 次方差可控。

---

## API Breaking Changes

| 日期 | 变更 | 影响 |
|------|------|------|
| 2026-02-22 | FastSAM 替代 YOLO 作为主检测器，config `yolo` → `fastsam` | `FastSNOWConfig.yolo` 已移除，改用 `FastSNOWConfig.fastsam`。`YOLOConfig` 保留供 asset 脚本使用。 |
| 2026-02-22 | Two-pass 架构替代 per-frame YOLO+SAM3 | `_run_pipeline()` 内部完全重构。Phase 1 DA3 batch + Phase 2 two-pass + Phase 3 CPU worker。 |
| 2026-02-22 | SAM3SharedSessionManager 新增 `add_object_point()` 和 `propagate_new_objects()` | 支持 mid-video point prompt 和 partial propagation。 |
| 2026-02-22 | DA3Wrapper / FastSAMWrapper 新增 `unload()` | 用于 GPU 显存分时管理。 |
| 2026-02-19 | `SamplingConfig` 从 `stride` 改为 `target_fps` | 旧配置 `sampling.stride` 不再生效。 |
| 2026-02-19 | `build_4dsg_from_video()` 返回 `FastSNOW4DSGResult` 对象 | 需调用 `result.cleanup()` 释放临时目录。 |
