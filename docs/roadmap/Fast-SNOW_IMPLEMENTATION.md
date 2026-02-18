# Fast-SNOW Implementation Spec (Fast Path v1)

## 1. Objective

目标：比 SNOW 更快，同时保持可接受准确率。

核心思路：用 RAM++ 语义发现 + SAM3 文本引导分割/追踪 替代 SNOW 的 HDBSCAN 聚类 + SAM2 点提示分割 + 匈牙利匹配追踪，跳过昂贵的 3D 点云聚类步骤。

约束：
- 以跟踪稳定性和速度优先，不追求完备语义。
- 节点存紧凑观测，不存整帧点云。
- RAM++ tag 仅用于 object discovery（驱动 SAM3 找到物体），**不进入 SG 节点**——节点严格遵循 SNOW 论文的 STEP token set 定义 `S_t^k = {τ, c, s, θ}`，JSON 序列化与内部表示 **1:1 对应**，不做压缩、截断或量化（见 §4.1）。F_k 窗口内全帧保留，float 保持原始精度。保留 open-world 设计：让 VLM 自己从 patch tokens 推理语义。

与 SNOW 的对比：

| | SNOW | Fast-SNOW |
|---|------|------|
| object discovery | HDBSCAN 3D 聚类 → 采样点 | RAM++ image tagging |
| 分割 prompt | SAM2 point prompt | SAM3 text prompt |
| 跨帧追踪 | 匈牙利匹配（STEP geometric + semantic cost） | SAM3 内置 memory bank + detector 自动发现新实例 |
| 3D 来源 | LiDAR 点云 / MapAnything 重建 | DA3 单目深度估计 |
| 迭代细化 | N_iter 轮 HDBSCAN + H_hop 几何校验 | 不做（SAM3 mask 质量更稳定，见 §3 说明） |
| SG 节点 | STEP token set `S_t^k = {τ, c, s, θ}` | 完全相同，JSON 1:1 对应（见 §4.1） |

---

## 2. Final Route

```
Frame I_t 到达（单 GPU 流式）
    │
    ├─ GPU ─→ Step 2. DA3    → depth_t, K_t, T_wc_t (→ T_cw_t)  [每帧]
    ├─ GPU ─→ Step 3. SAM3   → masks per object                   [每帧, propagate]
    ├─ GPU ─→ Step 1. RAM++  → tags_t → 触发新 SAM3 run           [每帧]
    │
    └─ CPU ─→ Step 4. Mask + Depth → 3D (centroid, shape, filter)
              Step 5. Global ID Fusion (跨 run ID 合并)
              Step 6. Build STEP Tokens
              Step 7. Build 4DSG (spatial graph + temporal tracks)
    │
    ▼ (按需)
    Step 8. Serialize + VLM Inference
```

---

## 3. Runtime Pipeline

### Step 0. Sampling & Scheduling
- 输入：视频帧 `I_t`，`t=0..T-1`。
- 默认每帧处理（`stride=1`），必要时降采样。

**帧调度模型**（单 GPU 流式处理）：
```
Frame t 到达
 ├─ GPU: DA3(I_t) → depth_t, K_t, T_wc_t       [每帧必须, ~30-80ms, 单次调用]
 ├─ GPU: SAM3 for all discovered runs            [每帧, 逐 run 串行 propagate]
 │       for run in all_runs:                     [每 run ~50-100ms, 含 detector]
 │           run.propagate(frame_t) → masks       [总计随 run 数自然增长]
 ├─ GPU: RAM++(I_t) → tags_t                     [每帧, ~10-30ms, 单次调用]
 └─ CPU: backproject + STEP + SG update          [与下一帧 GPU 重叠, ~5-10ms]
```
- 三个模型权重常驻同一 GPU，按时分复用顺序执行（DA3 → SAM3 runs → RAM++）。
- RAM++ 每帧执行——帧间隔可达 1s+，每帧场景变化显著，不能跳帧。
- CPU 工作（Step 4-7）与下一帧的 GPU 推理异步重叠。
- VLM 推理不在实时循环中，按查询按需触发。

### Step 1. RAM++ (每帧)
- 输入：`I_t`
- 输出：`tags_t: list[str]`
- 作用：**仅做 object discovery**，不输出框/掩码，tag 不进入最终 SG。

Tag 管理：
- `global_tag_set`：历史已触发 tag 集合。
- `new_tags_t = tags_t - global_tag_set`：当前帧首次出现的 tag，加入 `global_tag_set`，触发 SAM3 新 run。
- 已在 `global_tag_set` 中的 tag 不重复触发，由已有 run 的传播负责持续跟踪。
- run 采用自然增长策略：新 tag 只增不减，不设置 run 数硬上限，不做基于预算的主动淘汰。

### Step 2. DA3 (每帧)
- 输入：`I_t`
- 输出：
  - `depth_t`：`(H, W)` 米制深度图
  - `K_t`：`(3, 3)` 相机内参
  - `T_wc_t`：`(4, 4)` world-to-camera 外参（将世界坐标系中的点变换到相机坐标系）
  - `depth_conf_t`：`(H, W)` 逐像素置信度
- 与 Step 1 按帧并行执行。

坐标变换约定：
- `T_wc_t`（DA3 直接输出）：world → camera，即 `p_cam = T_wc_t @ p_world`。
- `T_cw_t = inv(T_wc_t)`：camera → world，即 `p_world = T_cw_t @ p_cam`。
- **Step 4 回投影使用 `T_cw_t`**（先像素→相机坐标，再 camera→world）。
- **Ego pose** = `T_cw_t`（相机在世界坐标系中的位姿）。

### Step 3. SAM3 (事件触发 + 持续传播)

SAM3 运行方式为**多 run 并行**：

- **新 tag 触发新 run**：当 `new_tags_t` 非空时，对每个新 tag 调用 `add_prompt(text_str=tag, frame_idx=t)` 启动一个独立 run。
- **已有 run 持续传播**：每个 run 内部，SAM3 的 memory bank 自动对已 prompt 的对象逐帧 propagation，产出帧级一致的 `obj_id_local`。
- **run 之间互相独立**：当前仓库中 `add_prompt` 会重置 inference state，因此每个 tag 必须在独立 run 中执行。同一 run 内的多帧追踪是连续的，但不同 run 之间没有共享状态。

多实例处理（SAM3 原生能力）：

SAM3 原生支持单 text prompt 检测多实例，无需额外覆盖率检测：

1. **初始检测**：`add_prompt(text_str="car", frame_idx=t)` 内部调用 SAM3Image detector，输出当前帧中**所有** score > `score_threshold_detection` 的实例，每个实例自动分配独立 `obj_id`。即一次 prompt 即可发现场景中所有可见的同类物体。
2. **传播期间自动发现**：`propagate_in_video()` 在每帧调用 detector（`allow_new_detections=True`），未匹配已有 track 的新检测自动获得新 `obj_id`。因此，后续帧中新进入场景的同类实例无需重新 prompt。
3. **非重叠约束**：当多个实例 mask 重叠时，SAM3 自动应用 `_apply_object_wise_non_overlapping_constraints`，抑制低置信度 mask，保证输出 mask 不互相覆盖。

因此，每个 tag 只需触发一次 `add_prompt`，后续的多实例发现和追踪完全由 SAM3 内部 tracker + detector 循环处理。

Run 生命周期（自然增长）：
- `created`：tag 首次出现，创建 run。
- `active`：run 每帧都执行 `propagate`。
- `ended`：仅在视频结束或窗口结束时统一释放。
- 不引入 `dormant/closed` 状态，不做 run 预算淘汰，避免由重建 run 引入的额外 ID switch。

关于迭代细化：
- SNOW 使用 N_iter 轮 HDBSCAN 重聚类 + H_hop 几何校验来处理分割错误。
- Fast-SNOW 不做迭代细化。原因：SAM3 的 text-guided segmentation 直接针对语义目标分割，不依赖 3D 聚类质量；几何异常（如 50m 车顶）在 Step 4 通过 `num_points` 和 extent 过滤处理。

### Step 4. Mask + Depth → 3D Object State (每帧每对象)
- 输入：`mask_{i,t}`（来自 SAM3）、`depth_t, K_t, T_cw_t = inv(T_wc_t)`（来自 DA3）
- 回投影公式（像素 → 世界坐标）：
  ```
  对 mask 内每个像素 (u, v)：
  1. 读取深度：d = depth_t[v, u]
  2. 像素 → 相机坐标：p_cam = d * K_t^{-1} @ [u, v, 1]^T
  3. 相机 → 世界坐标：p_world = T_cw_t @ [p_cam; 1]    （注意：用 T_cw，不是 T_wc）
  ```
  得到对象 3D 点集 `P_{i,t}`。
- 深度置信度过滤：回投影前，丢弃 `depth_conf_t[v, u] < conf_thresh`（默认 0.5）的像素，避免低质量深度估计污染 3D 点集。
- 计算（供 Step 6 使用）：
  - `centroid_xyz = mean(P_{i,t})`
  - 每轴统计 `(μ, σ, min, max) × 3` — 直接构成 shape token
  - `num_points = |P_{i,t}|`
- 过滤：`num_points < min_points` 或单轴 extent > `max_extent` 的对象直接丢弃（等效于 SNOW 的 H_hop 几何校验）。

### Step 5. Global ID Fusion
目标：把 SAM3 各独立 run 的本地 `obj_id_local` 合并为全局轨迹 ID `k`。

**背景**：SAM3 **没有**原生跨 run 融合能力。每个 `add_prompt` 创建独立 session（独立 memory bank、独立 obj_id 命名空间），`_apply_object_wise_non_overlapping_constraints` 仅在单 run 内部生效。因此跨 run 去重必须由 pipeline 自己处理。

**设计原则：mask 主导 + 最小几何/时间门控，不做 tag 级去重。** Tag embedding 相似度无法可靠区分"同义词"和"近义但不同类别"（如 "truck" vs "car"），因此合并依据以 mask 为主，并加轻量 3D 与时间一致性 gate 降低误并。

**Mask 去重算法**（帧级后处理）：

```
每帧处理完所有 run 的 mask 后：
1. 收集所有 run 的 (run_id, obj_id_local, mask, score) → candidates
2. 按 score 降序排列
3. 对所有 candidate pairs (i, j) where i.run_id ≠ j.run_id：
     if IoU(mask_i, mask_j) > cross_run_iou_thresh
        and ||centroid_i - centroid_j|| < merge_centroid_dist_m
        and |last_seen_t_i - last_seen_t_j| <= merge_temporal_gap:
       - 保留 score 更高的一方
       - 将另一方的 obj_id_local 映射到保留方的全局 ID k
       - 被合并方的 run 对该 obj_id 标记 suppressed（不影响 run 内其他 obj_id 继续追踪）
4. 未被合并的 candidate 各自获得独立全局 k
```

其中：`centroid_i` 来自 Step 4 的当前帧 3D 质心，`last_seen_t_i` 为该候选对应全局轨迹最近一次被观测到的帧号。

- **只比较不同 run 之间的对象**——同一 run 内 SAM3 已保证不重叠。
- **merge gate**：先看 2D mask IoU，再用最小 3D centroid gate 和时间邻近 gate 做二次确认。
- **不使用 Hungarian 匹配**——贪心（按 score 降序）足够，因为跨 run 重叠是稀疏的（同一物体最多被 2-3 个同义 tag 的 run 追踪到）。
- **suppressed 不等于删除 run**——被合并的只是某个具体 obj_id，该 run 内的其他物体不受影响。
- 复杂度：O(N_runs² × N_objects)，其中 N_runs 为累计触发的 tag-run 数。

全局 ID 生命周期：
- `active`：当前帧有 mask 输出。
- `lost`：连续 `lost_patience`（默认 5）帧无 mask → 标记 lost。
- `archived`：lost 状态持续 `archive_patience`（默认 30）帧 → archived，不再参与匹配。
- **不做 re-identification**：archived 对象不会被重新激活。如果同一物体重新出现，SAM3 propagation 会自动发现并分配新 obj_id，获得新全局 k。这简化了实现，代价是同一物体离开又重新进入场景时会有 ID switch，该权衡适用于以实时性优先的通用视频场景。

### Step 6. Build STEP Tokens (每帧每对象)
对每个全局对象 `k` 在帧 `t` 的观测，构建 STEP token（沿用 SNOW Eq. 4）：

```
S_t^k = {τ_{k,1}^t, ..., τ_{k,m}^t,  c_t^k,  s_t^k,  θ_t^k}
```

1. **patch_tokens (τ)**：在原图上将 mask 内像素着色 → 整张图按 16×16 grid 划分 → 每个 cell 算与 mask 的 IoU → 保留 IoU > 0.5 的 cell → 输出 `list[(row, col, iou)]`。
2. **centroid_token (c)**：直接取 Step 4 计算的 `centroid_xyz = (x̄, ȳ, z̄)`。
3. **shape_token (s)**：直接取 Step 4 计算的每轴统计 `(μ, σ, min, max) × 3`，共 12 维。
4. **temporal_token (θ)**：初始 `(t, t)`；Step 7 中按 track 更新为 `(t_start, t_end)`。

**关键**：节点不附加 RAM++ tag 或任何语义标签。

### Step 7. Build 4DSG

**a) Per-frame spatial graph** `G_t = (V_t, E_ego_t, E_obj_t)`（论文 Eq. 5 扩展）：
- 节点 `V_t`：该帧所有可见对象的 STEP token `{S_t^k}`。
- Ego-Object 边 `E_ego_t`：对每个可见物体，计算 ego 视角方位角（bearing）、垂直关系、距离、运动趋势。不做稀疏化。
- Object-Object 边 `E_obj_t`：对近邻物体对，计算 ego frame 中方向、垂直关系、距离。稀疏化：kNN(k=3)，总数 ≤ `max_obj_relations`。
- 所有方向均在 **ego frame** 中表达（以 ego heading 为 front），保持参考系一致。

**b) Temporal tracks** `F_k`（论文 Eq. 7）：
- 基于 Step 5 的全局 ID，将同一对象跨帧的 STEP token 串联：`F_k = {S_{t-T}^k, ..., S_t^k}`。
- **窗口内全帧保留**：不做观测截断，track 在窗口 `[t-T, t]` 内每个可见帧的完整 `S_t^k` 都保留在 `F_k` 中。
- 更新所有 `S^k` 的 temporal token `θ` 为 track 级：`(t_start, t_end) = (min frame, max frame)`。

**c) 4DSG** `M_t`（论文 Eq. 8）：
- `M_t = (G_{t-T:t}, {F_k})`
- 附加 ego pose 序列（Step 2 的 `T_cw_t = inv(T_wc_t)` 逐帧提取）。

### Step 8. Serialize + VLM Inference（论文 Eq. 9）

将 4DSG 序列化为 JSON，与关键帧图像一起喂给 VLM。

**设计原则**：JSON 严格 1:1 对应 STEP token set，不做任何压缩、截断或量化。每个 `S_t^k` 是一个逐帧逐对象的完整观测，`F_k` 保留窗口内全部帧。

**单个 `S_t^k` 的 JSON 结构**（float 保持原始精度，不做四舍五入）：
```json
{
  "t": 12,
  "tau": [{"row":3,"col":4,"iou":0.71}, {"row":3,"col":5,"iou":0.66}, {"row":4,"col":4,"iou":0.82}],
  "c": [5.12, -2.30, 1.05],
  "s": {
    "x": {"mu":5.12, "sigma":1.12, "min":2.90, "max":7.40},
    "y": {"mu":-2.30, "sigma":0.45, "min":-3.10, "max":-1.60},
    "z": {"mu":1.05, "sigma":0.36, "min":0.20, "max":1.70}
  },
  "theta": [3, 18]
}
```

**完整 4DSG JSON Schema**：
```json
{
  "metadata": {
    "grid": "16x16",
    "num_frames": 10,
    "num_tracks": 5
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
          "tau": [{"row":3,"col":4,"iou":0.71}, {"row":3,"col":5,"iou":0.66}],
          "c": [5.12, -2.30, 1.05],
          "s": {
            "x": {"mu":5.12, "sigma":1.12, "min":2.90, "max":7.40},
            "y": {"mu":-2.30, "sigma":0.45, "min":-3.10, "max":-1.60},
            "z": {"mu":1.05, "sigma":0.36, "min":0.20, "max":1.70}
          },
          "theta": [3, 8]
        },
        {
          "t": 8,
          "tau": [{"row":3,"col":4,"iou":0.68}, {"row":3,"col":5,"iou":0.72}],
          "c": [5.20, -2.25, 1.06],
          "s": {
            "x": {"mu":5.20, "sigma":1.10, "min":3.00, "max":7.30},
            "y": {"mu":-2.25, "sigma":0.44, "min":-3.05, "max":-1.55},
            "z": {"mu":1.06, "sigma":0.35, "min":0.22, "max":1.68}
          },
          "theta": [3, 8]
        }
      ]
    }
  ],
  "ego_relations": [
    {"object_id": 0, "bearing": "front-left", "elev": "level", "dist_m": 5.20, "motion": "approaching"},
    {"object_id": 1, "bearing": "right",      "elev": "level", "dist_m": 12.30, "motion": "lateral"}
  ],
  "object_relations": [
    {"src": 0, "dst": 2, "dir": "front-right", "elev": "level", "dist_m": 4.80},
    {"src": 1, "dst": 2, "dir": "left",        "elev": "level", "dist_m": 9.50}
  ]
}
```

**结构说明**：
- `tracks[i].F_k` = 论文 Eq. 7 的时序轨迹 `F_k = {S_{t-T}^k, ..., S_t^k}`，每个元素是完整的 `S_t^k`。
- 每个 `S_t^k` 都携带完整的 `tau`（含 iou）、`c`、`s`（含 μ/σ/min/max）、`theta`。
- `theta` 在同一 track 内所有观测中值相同（track 级），严格按论文定义随 `S_t^k` 一起序列化。
- `ego_relations` / `object_relations` 对应当前查询帧的边（Step 7 计算）。

**序列化参数**（`Fast-SNOWSerializationConfig`）：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_obj_relations` | 20 | object-object 边最大数 |
| `motion_window` | 3 | 计算 motion 所需的最近帧数 |

**不做截断 / 不做量化**（严格 STEP 合规）：
- **F_k 不截断**：窗口内每帧 `S_t^k` 全保留，不做 "最近 N 帧" 子采样。`F_k` 长度 = 该 track 在窗口内的实际可见帧数。
- **float 不量化**：所有浮点数保持 float32 原始精度序列化（如 `5.123456`），不做 `round(x, 2)` 或 float16 转换。
- 论文没有要求这两类压缩，为 "完全一致" 而移除。token 增加可通过缩短时间窗口或按对象分块多轮查询控制。

**Visual anchor**：关键帧图像作为独立 content block 传入 VLM（非 JSON 内嵌）。VLM 通过 `metadata.grid = "16x16"` + 每个 `S_t^k` 的 `tau` 坐标，将图像中的网格区域与 3D 数据关联。

**不包含语义字段**：无 `label/motion_state/velocity`——语义由 VLM 从 tau patch 位置 + 图像 + shape 分布自主推理（STEP open-world 设计）。

**Token 预算示例**（N_tracks=30, 窗口 10 帧, 平均 track 可见 ~7 帧 → ~210 observations）：

每个 `S_t^k` 的 token 构成（float32 全精度，数字更长）：
| 组件 | 估算 |
|------|------|
| `tau`（~8 patches × `{"row":X,"col":Y,"iou":0.XXXXXX}` ≈ 10 tokens/patch） | ~80 tokens |
| `c`（`[x.xxxxxx, y.yyyyyy, z.zzzzzz]`） | ~8 tokens |
| `s`（3 axes × `{"mu":,"sigma":,"min":,"max":}` ≈ 16 tokens/axis） | ~48 tokens |
| `theta`（`[t_start, t_end]`） | ~3 tokens |
| JSON 结构开销（key names, braces, `"t":` 等） | ~15 tokens |
| **单个 S_t^k 小计** | **~154 tokens** |

全 4DSG token 预算：
| 组件 | 估算 |
|------|------|
| metadata + ego | ~300 tokens |
| tracks（210 obs × ~154 tokens） | ~32,340 tokens |
| ego_relations（30 obj × ~12 tokens） | ~360 tokens |
| object_relations（20 edges × ~10 tokens） | ~200 tokens |
| JSON 结构开销（track headers 等） | ~500 tokens |
| **合计** | **~33,700 tokens** |

与截断+量化版（~19.6K）相比增加 ~72%，但仍在主流 VLM 上下文窗口内（128K+）。如果实际部署时 token 过多，通过缩短时间窗口或按对象分块多轮查询控制，而不是截断 F_k 或量化 float——保持 STEP 定义的完整性优先。

**VLM 推理**：`ŷ = VLM(q | M_t, I_keyframes)`。不在实时循环中，按查询按需触发。

---

## 4. SG Schema (v1)

### 4.1 Node = STEP Token Set（严格遵循论文 Eq. 4-5）
节点是一个 **STEP token set**（4 组 token 组成一个逐帧逐对象节点），不是单个 token：

```
S_t^k = {τ_{k,1}^t, ..., τ_{k,m}^t,  c_t^k,  s_t^k,  θ_t^k}
         ├─ patch tokens ─┘          centroid  shape   temporal
```

**内部表示 = JSON 序列化 = SNOW 原始定义**，三者 1:1 对应，不做任何压缩：

| STEP 组件 | SNOW 定义 | JSON 字段 | 说明 |
|-----------|----------|----------|------|
| **τ** (patch) | `list[(row, col, iou)]` | `tau: [{row, col, iou}, ...]` | 16×16 grid 中 IoU > 0.5 的 cell，**保留 iou 值**，**每帧每对象都有** |
| **c** (centroid) | `(x̄, ȳ, z̄)` | `c: [x, y, z]` | 3D 质心（= shape μ） |
| **s** (shape) | `(μ, σ, min, max) × 3`，共 12 维 | `s: {x: {mu,sigma,min,max}, y: {...}, z: {...}}` | 每轴 Gaussian 统计 + 极值，**完整保留** |
| **θ** (temporal) | `(t_start, t_end)` | `theta: [t_start, t_end]` | track 级时间跨度 |

**设计原则**：
- 每个 `S_t^k` 是逐帧逐对象的完整观测，不是对象级压缩汇总。
- JSON 字段名直接使用论文符号（`tau`, `c`, `s`, `theta`），不做重命名。
- 4DSG 时序轨迹 `F_k = {S_{t-T}^k, ..., S_t^k}` 中每个 `S_t^k` 都携带完整 4 组 token。
- 关键帧图像是 VLM 的额外视觉输入，不改变 STEP 定义——每帧 patch 是 STEP 自身的组件，不依赖 VLM 是否"看到"该帧。

`k` 是图结构的索引（全局 ID），不存在节点内部；序列化时作为 JSON 引用键 `object_id`。
不存 semantic_tag——语义由 VLM 在推理时从 patch tokens + 关键帧图像自主推理。

### 4.2 Edge (帧级空间关系)

设计原则：**预计算 VLM 做不好的数学（坐标旋转、距离差分），让 VLM 只做语义推理。**

边分两类：

#### 4.2.1 Ego-Object 边（每帧每个可见物体）

VLM 最常回答 ego 视角问题（"前方有什么"、"左边的车离我多远"），必须预计算 ego-to-object 的空间关系。

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | 物体 graph index |
| `bearing` | str | ego 视角方位角，8 方向（见下表） |
| `elev` | str | 垂直关系：above / level / below（\|dz\| > 0.5m） |
| `dist_m` | float | ego 到物体 centroid 的欧氏距离 |
| `motion` | str | 运动趋势：approaching / receding / lateral / static |

**Bearing 8 方向**（以 ego heading 为 front，每 45° 一个 sector）：
```
              front
        front-left  front-right
    left                    right
        back-left   back-right
              back
```

**Ego heading 提取**（从 `T_cw_t` 得到 `R_ego`）：
```
R_cw = T_cw_t[:3, :3]                          # camera→world 旋转矩阵
forward_world = R_cw @ [0, 0, 1]               # 相机 z 轴在世界坐标中的方向 = ego 前进方向
yaw = atan2(forward_world[1], forward_world[0]) # 只取 yaw 分量（地面平面假设）
R_ego = [[cos(yaw), -sin(yaw)],                # 2D 旋转矩阵：world → ego frame
         [sin(yaw),  cos(yaw)]]
ego_pos_xy = T_cw_t[:2, 3]                     # ego 在世界坐标中的 XY 位置
```
- **地面假设**：忽略 pitch/roll，只取 yaw。相机近似水平的场景下，pitch/roll 对方位角影响通常较小。
- `R_ego^T`（= `R_ego` 的转置）将世界坐标系差向量旋转到 ego frame。

**Bearing 计算**：
```
d = obj_centroid_xy - ego_pos_xy               # 世界坐标 XY 差向量
d_ego = R_ego^T @ d                            # 旋转到 ego frame
angle = atan2(d_ego[1], d_ego[0])              # ego frame 中的方位角
bearing = quantize_to_8_sectors(angle)          # 按 45° 量化
```

**Motion 计算**（需最近 N 帧数据，N ≥ 2）：
```
dist_history = [||obj_pos_t - ego_pos_t|| for t in recent_N_frames]
rate = (dist_history[-1] - dist_history[0]) / N
if rate < -threshold:  "approaching"       # 距离在缩小
elif rate > threshold:  "receding"          # 距离在增大
elif lateral_change > threshold: "lateral"  # 距离不变但横向位移大
else: "static"
```
- **冷启动**：当历史帧不足 `motion_window` 时，`motion` 字段输出 `"unknown"`，不猜测。

不做稀疏化——所有可见物体都需要 ego 边。

#### 4.2.2 Object-Object 边（帧级，稀疏）

物体间空间关系，用于回答 "行人在公交车站旁边吗" 等非 ego 视角问题。

| 字段 | 类型 | 说明 |
|------|------|------|
| `src`, `dst` | int | graph indices |
| `dir` | str | src → dst 方向，**ego frame 中表达**（与 ego 边同一参考系） |
| `elev` | str | above / level / below |
| `dist_m` | float | centroid 间欧氏距离 |

**Direction 计算**（与 bearing 相同，只是起点从 ego 换成 src object）：
```
d = dst_centroid_xy - src_centroid_xy
d_ego = R_ego^T @ d                            # 仍然旋转到 ego frame，保持参考系一致
dir = quantize_to_8_sectors(atan2(d_ego[1], d_ego[0]))
```

**稀疏化**：kNN（k=3）per object，总数不超过 `max_obj_relations`（默认 20）。不做全连接。

**不再使用**：
- 合并字符串 `right_level_near` → 拆为 `dir` + `elev` + `dist_m` 独立字段
- 距离分箱 `near/medium/far` → 直接给精确距离 `dist_m`，VLM 能读数字

### 4.3 视频帧
- 不作为 SG 节点。
- 作为 `visual_anchor` 元数据（关键帧图像路径 + frame_idx），供多模态 VLM 参考。

---

## 5. Storage Rules (必须遵守)

### 5.1 代码仓库放置规范

1. 所有 Fast-SNOW 自身代码统一放在 `fast_snow/` 包下（按功能子模块组织）。
2. 需要从 GitHub clone 的视觉模型源码（如 SAM3、RAM++、DA3）统一放在 `fast_snow/vision/` 下的对应子目录。
3. 禁止把第三方模型源码散落到仓库根目录、`docs/`、`assets/` 或其他非 `fast_snow/vision/` 目录。

示例：
```text
fast_snow/
  vision/
    sam3/
    ram_plus/
    depth_anything_v3/
```

### 5.2 模型权重与缓存放置规范

1. 所有模型权重（ckpt/pt/bin/safetensors）统一存放在 `fast_snow/models/`。
2. 各模型独立子目录，禁止将权重混放到源码目录（例如禁止放在 `fast_snow/vision/sam3/checkpoints`）。
3. Hugging Face 缓存目录也统一落在 `fast_snow/models/hf_cache/`。
4. `fast_snow/models/` 下的二进制权重默认不纳入 Git 版本管理。

示例：
```text
fast_snow/
  models/
    sam3/
    ram_plus/
    da3/
    hf_cache/
```

### 5.3 路径规范（统一相对路径）

1. 代码、配置、脚本一律使用项目根目录相对路径，不使用机器相关绝对路径（如 `/Users/...`）。
2. 统一通过 `project_root` 解析路径（CLI 可接受 `--project-root`，默认 `.`）。
3. 配置文件中路径字段均写成相对路径，例如：
   - `fast_snow/models/...`
   - `fast_snow/vision/...`
   - `docs/phases/...`

### 5.4 文档归档规范

1. 路线与架构规划文档放在 `docs/roadmap/`。
2. 开发阶段文档（phase-by-phase 计划、实验记录、迭代说明）放在 `docs/phases/`。
3. 参考材料与论文笔记放在 `docs/reference/`。
4. 用户向总览说明保留在根 `README.md`，文档索引放在 `docs/README.md`。

### 5.5 STEP/4DSG 存储一致性（与论文对齐）

1. 不存每帧完整点云（点集仅在 Step 4 计算过程中使用，不持久化）。
2. **F_k 不截断**：窗口内每帧 `S_t^k` 全保留，不做观测子采样。
3. **不做序列化量化**：所有浮点数以 float32 原始精度存储和序列化。
4. 上下文长度通过时间窗口大小与查询分块（chunking）控制，而非截断 F_k 或量化 float。

---

## 6. Hyperparameter Config（集中管理）

所有可调参数集中在此，pipeline 各 Step 引用此表的变量名。

### 6.1 SAM3 检测

| 参数 | 默认值 | 来源 Step | 说明 |
|------|--------|-----------|------|
| `score_threshold_detection` | 0.3 | Step 3 | SAM3 detector 输出的最低 score，低于此值不产生 mask |

### 6.2 深度 & 3D 过滤

| 参数 | 默认值 | 来源 Step | 说明 |
|------|--------|-----------|------|
| `conf_thresh` | 0.5 | Step 4 | DA3 深度置信度过滤阈值，低于此值的像素不参与回投影 |
| `min_points` | 50 | Step 4 | 回投影后最少 3D 点数，低于此值丢弃对象 |
| `max_extent` | 30.0 (m) | Step 4 | 单轴 extent（max−min）上限，超过则视为深度噪声丢弃 |

### 6.3 Mask 去重 & ID 融合

| 参数 | 默认值 | 来源 Step | 说明 |
|------|--------|-----------|------|
| `cross_run_iou_thresh` | 0.5 | Step 5 | 跨 run mask IoU 阈值，高于此值进入合并候选 |
| `merge_centroid_dist_m` | 2.0 | Step 5 | 最小 3D gate：候选对的 centroid 距离上限（米） |
| `merge_temporal_gap` | 2 | Step 5 | 时间一致性 gate：候选对最近观测帧差上限 |
| `lost_patience` | 5 (帧) | Step 5 | active → lost 所需连续无 mask 帧数 |
| `archive_patience` | 30 (帧) | Step 5 | lost → archived 所需额外帧数 |

### 6.4 STEP Token

| 参数 | 默认值 | 来源 Step | 说明 |
|------|--------|-----------|------|
| `grid_size` | 16 | Step 6 | patch token 网格尺寸（16×16 = 256 cells） |
| `iou_threshold` | 0.5 | Step 6 | 网格 cell 与 mask 的 IoU 阈值，低于此值不保留 |

### 6.5 SG 边计算

| 参数 | 默认值 | 来源 Step | 说明 |
|------|--------|-----------|------|
| `elev_thresh` | 0.5 (m) | Step 7 | \|dz\| > 此值判定 above/below，否则 level |
| `motion_thresh` | 0.3 (m/帧) | Step 7 | 距离变化率阈值，区分 approaching/receding/static |
| `lateral_thresh` | 0.3 (m/帧) | Step 7 | 横向位移阈值，区分 lateral/static |
| `knn_k` | 3 | Step 7 | object-object 边 kNN 参数 |

### 6.6 序列化 & VLM

| 参数 | 默认值 | 来源 Step | 说明 |
|------|--------|-----------|------|
| `max_obj_relations` | 20 | Step 8 | object-object 边最大数 |
| `motion_window` | 3 (帧) | Step 8 | 计算 motion 所需的最近帧数 |

不设置 `max_tracks` 硬限制：track 全量保留，必要时通过缩短时间窗口或多轮分块查询控制上下文长度。

**已移除**（严格 STEP 合规）：
- ~~`precision`~~：不做 float 四舍五入，保持 float32 原始精度。
- ~~`max_obs_per_track`~~：不截断 F_k，窗口内所有帧全保留。
- ~~`max_tracks`~~：不对 track 数做裁剪。

---

## 7. Acceptance Criteria

评测协议（固定）：
- 数据：使用固定清单 `configs/eval/fast_snow_eval_manifest.json`（同一批视频、同一问题集）。
- 分辨率：统一到 `1920x1080`（或在清单中固定）。
- 硬件：单卡 H200（同机型）记录端到端延迟。
- 基线：在同一清单上运行 SNOW 基线并保存结果。

验收条件：
1. 全局 ID 唯一性：同一窗口内重复全局 ID 率为 0。
2. 追踪稳定性：报告并对比 `IDF1`、`ID switches / 1k frames`、`fragmentation / 1k frames`。
3. VLM 精度：同一问答集上，Fast-SNOW 相对 SNOW 的 absolute accuracy 下降不超过 3%。
4. 速度：报告 `fps` 与 `p50/p95 latency`，并显著优于 SNOW 的 ~1.1 FPS。
5. 可复现性：固定随机种子、模型版本和配置文件，重复 3 次方差可控。
