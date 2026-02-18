# SNOW (Sohn et al., 2025) 工程复现说明书

本文档基于论文 "SNOW: Spatio-Temporal Scene Understanding with World Knowledge for Open-World Embodied Reasoning" (arXiv:2512.16461) 整理，目标是**逐组件完整复刻 SNOW**，提供**输入/输出、张量维度、参数、数学公式**和**伪代码级实现细节**。

---

## 目录
1. [记号与输入定义](#1-记号与输入定义)
2. [组件级复现细节](#2-组件级复现细节)
   - 2.1 [点云聚类与候选点采样](#21-点云聚类与候选点采样-hdbscan)
   - 2.2 [投影与 SAM2 分割](#22-投影与-sam2-分割)
   - 2.3 [多视角一致性匹配](#23-多视角一致性匹配匈牙利算法)
   - 2.4 [3D 点与 Mask 关联](#24-3d-点与-mask-关联)
   - 2.5 [STEP 编码](#25-step-编码spatio-temporal-tokenized-patch-encoding)
   - 2.6 [迭代细化与 H-hop 几何校验](#26-迭代细化与-h-hop-几何一致性校验)
   - 2.7 [4D Scene Graph 构建](#27-4d-scene-graph-4dsg-构建)
   - 2.8 [VLM 推理](#28-vlm-推理)
3. [Algorithm 1 完整伪代码](#3-algorithm-1-完整伪代码)
4. [实验配置与超参数](#4-实验配置与超参数)
5. [关键张量/对象 I/O 汇总](#5-关键张量对象-io-汇总)
6. [代码模块设计建议](#6-代码模块设计建议)
7. [评测协议与数据集](#7-评测协议与数据集)
8. [论文未明确需工程决策的细节](#8-论文未明确需工程决策的细节)
9. [复现 Checklist](#9-复现-checklist)
10. [版本记录](#10-版本记录)

---

## 1. 记号与输入定义

### 1.1 传感器输入

| 符号 | 含义 | 形状/类型 |
|------|------|-----------|
| `I_t^c` | 第 `t` 帧第 `c` 个相机的 RGB 图像 | `(H, W, 3)`, `uint8`, RGB |
| `P_t` | 第 `t` 帧的 3D 点云 | `(N_t, 3)`, `float32`, 世界坐标系 |
| `K^c` | 相机 `c` 的内参矩阵 | `(3, 3)` |
| `[R^c | t^c]` | 相机 `c` 的外参（世界系到相机系） | `(3, 4)` |
| `T_ego^t` | 第 `t` 帧的 ego pose（车辆/机器人位姿） | `(4, 4)` SE(3) |

**传感器要求**：
- 所有传感器需要**时间同步**（论文：约 1Hz，T=10 帧 ≈ 10 秒）
- **几何标定**：已知相机内外参，可实现点云到图像的投影

### 1.2 仅视频输入的处理方式

论文明确提到（Section 3, Supplementary 6）：**当没有 LiDAR 点云时，使用 MapAnything [22] 从图像序列重建 3D**。

**A. 论文推荐路径**
- **MapAnything**：image-only 重建，输出：
  - 点云 `P_t`（局部窗口内的融合点云）
  - 相机位姿 `T_{w←c}`（全局坐标系下的相机轨迹）

**B. 替代方案**
- 单目深度估计 + SLAM/VO（如 MiDaS + ORB-SLAM）
- RGB-D 传感器直接投影

### 1.3 投影函数（论文 Eq. 2）

将 3D 点 `p_i^t ∈ R^3` 投影到相机 `c` 的图像平面：

```
π(p_i^t, I_t^c) → (x_img, y_img)
```

**数学公式**：
```
p_cam = R^c · p_i^t + t^c           # 世界系 → 相机系, (3,)
p_img = K^c · p_cam                  # 相机系 → 图像齐次坐标, (3,)
x_img = p_img[0] / p_img[2]          # 归一化
y_img = p_img[1] / p_img[2]          # 归一化
```

**有效性检查**：
- `p_cam[2] > 0`（点在相机前方）
- `0 ≤ x_img < W` 且 `0 ≤ y_img < H`（在图像范围内）

### 1.4 关键超参数（论文明确给出）

| 参数 | 值 | 说明 |
|------|-----|------|
| `m` | 4 | 每个簇采样的 prompt 点数 |
| `G` | 16 | STEP 网格划分尺寸（16×16=256 patches） |
| `IoU_thresh` | 0.5 | STEP patch 保留阈值 |
| `T` | 10 | 时间窗口帧数（约 10 秒） |
| `N_iter` | 1 | 迭代细化次数 |
| `H_hop` | 1 | 推理 hop 数（几何异常检测轮数） |

### 1.5 模型组件配置

| 组件 | 配置 | 说明 |
|------|------|------|
| SAM2 | `SAM2 Hiera Large` (video predictor) | 分割模型 |
| VLM | `Gemma3-4B-IT` | 推理骨干 |
| SLAM (LiDAR) | `KISS-SLAM` | 提供全局坐标对齐 |
| SLAM (image-only) | `MapAnything` | 视觉重建 + 位姿估计 |

---

## 2. 组件级复现细节

### 2.1 点云聚类与候选点采样 (HDBSCAN)

**论文引用**: Section 3.1, Eq. 1

#### 输入
| 名称 | 形状 | 说明 |
|------|------|------|
| `U_t` | `(N, 3)` | 未映射点集（初始为 `P_t`） |

#### 处理流程

**Step 1: HDBSCAN 聚类**

HDBSCAN（Hierarchical Density-Based Spatial Clustering of Applications with Noise）在欧氏空间聚类：

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=30,      # 最小簇大小（论文未明确，建议值）
    min_samples=5,            # 核心点邻域最小样本数（建议值）
    cluster_selection_epsilon=0.0,  # 聚类选择 epsilon（建议值）
    metric='euclidean',       # 欧氏距离
    cluster_selection_method='eom'  # Excess of Mass（默认）
)
labels = clusterer.fit_predict(U_t)  # labels: (N,), -1 表示噪声
```

**输出**：簇集合 `R_t = {R_t^1, ..., R_t^K}`，其中噪声点 (`label == -1`) 被排除。

**数学表示**（论文 Eq. 1）：
```
R_t = HDBSCAN(U_t) = {R_t^1, ..., R_t^K}
```

**Step 2: 均匀采样 m=4 个 prompt 点**

对每个簇 `R_t^k`，均匀采样 `m=4` 个代表点作为 SAM2 的 point prompts：

```python
def uniform_sample(cluster_points, m=4):
    """
    从簇中均匀采样 m 个点

    Args:
        cluster_points: (N_k, 3) 簇内点云
        m: 采样数量

    Returns:
        sampled_points: (m, 3)
    """
    N_k = cluster_points.shape[0]
    if N_k <= m:
        # 点数不足，重复采样或 padding
        indices = np.random.choice(N_k, m, replace=True)
    else:
        # 均匀间隔采样
        indices = np.linspace(0, N_k - 1, m, dtype=int)
        # 或使用 FPS（最远点采样）获得更好的空间分布
    return cluster_points[indices]
```

**建议实现**：使用最远点采样（FPS）获得更均匀的空间分布：
```python
def farthest_point_sampling(points, m):
    """最远点采样"""
    N = points.shape[0]
    selected = [np.random.randint(N)]
    distances = np.full(N, np.inf)

    for _ in range(m - 1):
        last = points[selected[-1]]
        dist = np.linalg.norm(points - last, axis=1)
        distances = np.minimum(distances, dist)
        selected.append(np.argmax(distances))

    return points[selected]
```

#### 输出
| 名称 | 形状 | 说明 |
|------|------|------|
| `R_t` | `List[ndarray]` | K 个簇，`R_t^k` 形状为 `(N_k, 3)` |
| `V_t^k` | `(m, 3)` = `(4, 3)` | 簇 k 的采样 prompt 点 |

#### HDBSCAN 参数选择建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `min_cluster_size` | 30-50 | 驾驶场景中小物体（行人）约 30 个点 |
| `min_samples` | 5-10 | 核心点邻域密度 |
| `cluster_selection_epsilon` | 0.0-0.5 | 控制簇合并的距离阈值 |
| `metric` | `'euclidean'` | 3D 欧氏距离 |

---

### 2.2 投影与 SAM2 分割

**论文引用**: Section 3.2, Eq. 2-3

#### 输入
| 名称 | 形状 | 说明 |
|------|------|------|
| `P_t` | `(N, 3)` | 完整点云 |
| `V_t^k` | `(4, 3)` | 簇 k 的 prompt 点 |
| `I_t^c` | `(H, W, 3)` | 相机 c 的 RGB 图像 |
| `K^c, [R^c|t^c]` | 相机内外参 | 投影所需 |

#### 处理流程

**Step 1: 批量投影点云到图像**

```python
def project_points_to_image(points_3d, K, R, t, img_shape):
    """
    批量投影 3D 点到图像平面

    Args:
        points_3d: (N, 3) 世界坐标系下的点
        K: (3, 3) 相机内参
        R: (3, 3) 旋转矩阵
        t: (3,) 平移向量
        img_shape: (H, W)

    Returns:
        img_coords: (N, 2) 图像坐标 (x, y)
        valid_mask: (N,) bool，标记有效投影
    """
    H, W = img_shape

    # 世界系 → 相机系
    points_cam = (R @ points_3d.T + t.reshape(3, 1)).T  # (N, 3)

    # 深度检查（在相机前方）
    valid_depth = points_cam[:, 2] > 0

    # 相机系 → 图像齐次坐标
    points_img = (K @ points_cam.T).T  # (N, 3)

    # 归一化
    z = points_img[:, 2:3]
    z[z == 0] = 1e-8  # 避免除零
    img_coords = points_img[:, :2] / z  # (N, 2)

    # 边界检查
    valid_bounds = (
        (img_coords[:, 0] >= 0) & (img_coords[:, 0] < W) &
        (img_coords[:, 1] >= 0) & (img_coords[:, 1] < H)
    )

    valid_mask = valid_depth & valid_bounds

    return img_coords, valid_mask
```

**Step 2: SAM2 Video Predictor 调用**

```python
from sam2.build_sam import build_sam2_video_predictor

# 初始化 SAM2 Hiera Large
predictor = build_sam2_video_predictor(
    config_file="sam2_hiera_l.yaml",
    ckpt_path="sam2_hiera_large.pt"
)

def generate_masks_sam2(predictor, video_frames, prompt_points_list, prompt_labels):
    """
    使用 SAM2 生成分割 mask

    Args:
        predictor: SAM2 video predictor
        video_frames: List[(H, W, 3)] 视频帧序列
        prompt_points_list: List[(m, 2)] 每个对象的 prompt 点坐标
        prompt_labels: List[(m,)] 每个点的标签（1=前景, 0=背景）

    Returns:
        masks: List[(H, W)] bool 每个对象的 mask
    """
    with torch.inference_mode():
        # 初始化状态
        state = predictor.init_state(video_path=video_frames)

        masks = []
        for obj_idx, (points, labels) in enumerate(zip(prompt_points_list, prompt_labels)):
            # 添加 point prompts
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,  # 当前帧
                obj_id=obj_idx,
                points=points,  # (m, 2) in (x, y) format
                labels=labels   # (m,) all 1s for foreground
            )

            # 传播到视频序列
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                mask = (mask_logits[obj_idx] > 0).cpu().numpy()  # (H, W)
                masks.append(mask)

        predictor.reset_state(state)

    return masks
```

**SAM2 Point Prompt 格式**：
- `points`: `(m, 2)` 数组，格式为 `(x, y)` 像素坐标
- `labels`: `(m,)` 数组，`1` 表示前景，`0` 表示背景
- 对于 SNOW，所有采样点都标记为前景 (`labels = np.ones(m)`)

#### 输出
| 名称 | 形状 | 说明 |
|------|------|------|
| `m_t^{k,c}` | `(H, W)` bool | 相机 c 下对象 k 的分割 mask |

**数学表示**（论文 Eq. 3）：
```
m_t^{k,c} = SAM2({V_t^k}_img, I_t^c) ⊂ I_t^c
```

---

### 2.3 多视角一致性匹配（匈牙利算法）

**论文引用**: Section 3.2, "Consistency between masks... via Hungarian matching [23]"

#### 目的
确保同一物理对象在不同相机视角下的 mask 被正确关联。

#### 输入
| 名称 | 形状 | 说明 |
|------|------|------|
| `masks_c1` | `List[(H, W)]` | 相机 1 的所有 masks |
| `masks_c2` | `List[(H, W)]` | 相机 2 的所有 masks |
| `P_t` | `(N, 3)` | 点云（用于计算 3D 一致性） |

#### 处理流程

**代价矩阵定义**（论文未明确，以下为合理推断）：

```python
from scipy.optimize import linear_sum_assignment

def compute_matching_cost(mask1, mask2, points_3d_1, points_3d_2):
    """
    计算两个 mask 之间的匹配代价

    可选代价函数：
    1. IoU-based: 1 - IoU（需要重投影到同一视角）
    2. 3D centroid distance: ||c1 - c2||_2
    3. 特征相似度: cosine distance of features
    """
    # 方法 1: 3D 质心距离（推荐）
    c1 = points_3d_1.mean(axis=0) if len(points_3d_1) > 0 else np.zeros(3)
    c2 = points_3d_2.mean(axis=0) if len(points_3d_2) > 0 else np.zeros(3)
    cost = np.linalg.norm(c1 - c2)

    return cost

def hungarian_matching(masks_cam1, masks_cam2, points_per_mask_cam1, points_per_mask_cam2):
    """
    使用匈牙利算法匹配两个相机的 masks

    Args:
        masks_cam1: List of masks from camera 1
        masks_cam2: List of masks from camera 2
        points_per_mask_cam*: 每个 mask 对应的 3D 点集

    Returns:
        matches: List of (idx1, idx2) pairs
    """
    n1, n2 = len(masks_cam1), len(masks_cam2)
    cost_matrix = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            cost_matrix[i, j] = compute_matching_cost(
                masks_cam1[i], masks_cam2[j],
                points_per_mask_cam1[i], points_per_mask_cam2[j]
            )

    # 匈牙利算法求解最优匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 过滤高代价匹配（阈值需要调参）
    matches = []
    COST_THRESHOLD = 5.0  # 米，需要根据场景调整
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < COST_THRESHOLD:
            matches.append((i, j))

    return matches
```

#### 输出
| 名称 | 说明 |
|------|------|
| `matches` | 跨视角 mask 匹配对 `(idx_c1, idx_c2)` |
| `m_t^k` | 融合后的统一对象 mask 表示 |

---

### 2.4 3D 点与 Mask 关联

**论文引用**: Section 3.2, "Each 3D point... is assigned to mask m_t^{k,c} if its projection lies within the support"

#### 输入
| 名称 | 形状 | 说明 |
|------|------|------|
| `img_coords` | `(N, 2)` | 所有点的图像投影坐标 |
| `valid_mask` | `(N,)` | 有效投影标记 |
| `m_t^{k,c}` | `(H, W)` bool | 对象 k 在相机 c 的 mask |

#### 处理流程

```python
def assign_points_to_masks(img_coords, valid_mask, masks, points_3d):
    """
    将 3D 点分配给对应的 masks

    Args:
        img_coords: (N, 2) 投影坐标
        valid_mask: (N,) 有效投影标记
        masks: List[(H, W)] 所有 masks
        points_3d: (N, 3) 原始点云

    Returns:
        assigned_points: List[ndarray] 每个 mask 对应的 3D 点集
        point_labels: (N,) 每个点的 mask 归属（-1=未分配）
    """
    N = points_3d.shape[0]
    point_labels = np.full(N, -1, dtype=int)
    assigned_points = [[] for _ in masks]

    for i in range(N):
        if not valid_mask[i]:
            continue

        x, y = int(img_coords[i, 0]), int(img_coords[i, 1])

        # 检查点落在哪个 mask 内
        for k, mask in enumerate(masks):
            if mask[y, x]:  # 注意 mask 索引是 [y, x]
                point_labels[i] = k
                assigned_points[k].append(points_3d[i])
                break  # 假设 masks 不重叠

    # 转换为 numpy 数组
    assigned_points = [np.array(pts) if pts else np.empty((0, 3))
                       for pts in assigned_points]

    return assigned_points, point_labels
```

#### 多视角融合

当同一对象在多个相机可见时，合并来自不同视角的点：

```python
def merge_multiview_points(assigned_points_per_cam, matches):
    """
    合并多视角的点云归属

    Args:
        assigned_points_per_cam: List[List[ndarray]] 每个相机每个对象的点
        matches: 跨视角匹配结果

    Returns:
        merged_points: List[ndarray] 融合后每个对象的点集
    """
    # 使用 Union-Find 或简单合并
    merged = {}
    for cam_idx, assigned in enumerate(assigned_points_per_cam):
        for obj_idx, points in enumerate(assigned):
            key = (cam_idx, obj_idx)
            # 根据 matches 找到统一 ID
            unified_id = find_unified_id(key, matches)
            if unified_id not in merged:
                merged[unified_id] = []
            if len(points) > 0:
                merged[unified_id].append(points)

    # 连接所有点
    merged_points = [np.vstack(pts) if pts else np.empty((0, 3))
                     for pts in merged.values()]
    return merged_points
```

#### 输出
| 名称 | 形状 | 说明 |
|------|------|------|
| `R̂_t^k` | `(N_k, 3)` | 对象 k 关联的 3D 点集 |

---

### 2.5 STEP 编���（Spatio-Temporal Tokenized Patch Encoding）

**论文引用**: Section 3.2, Figure 3, Eq. 4

STEP 是 SNOW 的核心创新，将语义、几何、时间信息编码为统一的 token 表示。

#### 输入
| 名称 | 形状 | 说明 |
|------|------|------|
| `m_t^{k,c}` | `(H, W)` bool | 对象 k 的分割 mask |
| `I_t^c` | `(H, W, 3)` | 原始 RGB 图像 |
| `R̂_t^k` | `(N_k, 3)` | 对象 k 的 3D 点集 |
| `t` | int | 当前帧索引 |
| `t_start, t_end` | int | 对象首次/末次出现帧 |

#### 处理流程

**Step 1: Mask 可视化预处理**

```python
def prepare_masked_image(image, mask, color=(255, 0, 0)):
    """
    论文: "The object mask is isolated by coloring all in-mask pixels"

    Args:
        image: (H, W, 3) RGB 图像
        mask: (H, W) bool mask
        color: 着色颜色

    Returns:
        masked_image: (H, W, 3) 着色后的图像
    """
    masked_image = image.copy()
    masked_image[mask] = color
    return masked_image
```

**Step 2: 16×16 网格划分与 IoU 筛选**

```python
def compute_patch_tokens(mask, grid_size=16, iou_threshold=0.5):
    """
    论文: "partitioned into a fixed 16×16 grid... Cells with IoU > 0.5 are retained"

    Args:
        mask: (H, W) bool mask
        grid_size: G = 16
        iou_threshold: 0.5

    Returns:
        patch_tokens: List[dict] 保留的 patch 信息
            - 'row': int, patch 行索引 (0-15)
            - 'col': int, patch 列索引 (0-15)
            - 'iou': float, mask 在该 patch 的 IoU
    """
    H, W = mask.shape
    patch_h = H // grid_size
    patch_w = W // grid_size

    patch_tokens = []

    for row in range(grid_size):
        for col in range(grid_size):
            # 提取 patch 区域
            y_start = row * patch_h
            y_end = (row + 1) * patch_h if row < grid_size - 1 else H
            x_start = col * patch_w
            x_end = (col + 1) * patch_w if col < grid_size - 1 else W

            patch_mask = mask[y_start:y_end, x_start:x_end]

            # 计算 IoU（intersection / patch_area）
            intersection = patch_mask.sum()
            patch_area = (y_end - y_start) * (x_end - x_start)
            iou = intersection / patch_area

            if iou > iou_threshold:
                patch_tokens.append({
                    'row': row,
                    'col': col,
                    'iou': float(iou),
                    'bbox': (x_start, y_start, x_end, y_end)  # 用于提取特征
                })

    return patch_tokens
```

**Step 3: Centroid Token**

```python
def compute_centroid_token(points_3d):
    """
    论文: "centroid token c_t^k = (x̄, ȳ, z̄) encoding the 3D center"

    Args:
        points_3d: (N_k, 3) 对象的 3D 点集

    Returns:
        centroid: (3,) 质心坐标
    """
    if len(points_3d) == 0:
        return np.zeros(3)
    return points_3d.mean(axis=0)
```

**数学公式**：
```
c_t^k = (x̄, ȳ, z̄) = (1/N_k) Σ_{i=1}^{N_k} p_i
```

**Step 4: Shape Token（高斯分布 + 空间范围）**

```python
def compute_shape_token(points_3d):
    """
    论文: "shape token s_t^k = {(μ_a, σ_a, a_min, a_max)}_{a∈{x,y,z}}"

    "derived from Gaussian distributions and spatial extents along each axis,
    where μ_a and σ_a denote the mean and standard deviation, and a_min, a_max
    capture the axis-aligned boundaries"

    Args:
        points_3d: (N_k, 3)

    Returns:
        shape_token: (12,) = [μ_x, σ_x, x_min, x_max, μ_y, σ_y, y_min, y_max, μ_z, σ_z, z_min, z_max]
    """
    if len(points_3d) == 0:
        return np.zeros(12)

    shape_token = []
    for axis in range(3):  # x, y, z
        values = points_3d[:, axis]
        mu = values.mean()
        sigma = values.std()
        a_min = values.min()
        a_max = values.max()
        shape_token.extend([mu, sigma, a_min, a_max])

    return np.array(shape_token)
```

**数学公式**：
```
s_t^k = {(μ_a, σ_a, a_min, a_max)}_{a∈{x,y,z}}

其中：
  μ_a = (1/N_k) Σ p_i[a]           # 均值
  σ_a = √[(1/N_k) Σ (p_i[a] - μ_a)²]  # 标准差
  a_min = min(p_i[a])              # 最小值
  a_max = max(p_i[a])              # 最大值
```

**Step 5: Temporal Token**

```python
def compute_temporal_token(t_start, t_end):
    """
    论文: "temporal tokens θ_t^k = (t_start, t_end) encoding the time of
    first appearance and disappearance"

    Args:
        t_start: int, 首次出现帧
        t_end: int, 最后出现帧（当前帧或消失帧）

    Returns:
        temporal_token: (2,)
    """
    return np.array([t_start, t_end])
```

**Step 6: 组装完整 STEP Token**

```python
@dataclass
class STEPToken:
    """STEP Token 数据结构"""
    object_id: int                      # 对象 ID
    frame_idx: int                      # 帧索引

    # Image patch tokens
    patch_tokens: List[dict]            # [{row, col, iou, bbox}, ...]

    # Geometric tokens
    centroid: np.ndarray                # (3,)
    shape: np.ndarray                   # (12,)

    # Temporal tokens
    temporal: np.ndarray                # (2,) = (t_start, t_end)

    # Optional: 原始数据引用
    points_3d: np.ndarray = None        # (N_k, 3)
    mask: np.ndarray = None             # (H, W)

def create_step_token(object_id, frame_idx, mask, points_3d, t_start, t_end,
                      grid_size=16, iou_threshold=0.5):
    """
    创建完整的 STEP Token

    论文 Eq. 4:
    S_t^k = {τ_{k,1}^t, ..., τ_{k,m}^t, c_t^k, s_t^k, θ_t^k}
    """
    return STEPToken(
        object_id=object_id,
        frame_idx=frame_idx,
        patch_tokens=compute_patch_tokens(mask, grid_size, iou_threshold),
        centroid=compute_centroid_token(points_3d),
        shape=compute_shape_token(points_3d),
        temporal=compute_temporal_token(t_start, t_end),
        points_3d=points_3d,
        mask=mask
    )
```

#### 输出

| 名称 | 形状 | 说明 |
|------|------|------|
| `S_t^k` | STEPToken | 完整的 STEP token |
| └ `patch_tokens` | `List[M_k]` | 图像 patch tokens，数量 M_k ≤ 256 |
| └ `centroid` | `(3,)` | 3D 质心 |
| └ `shape` | `(12,)` | 形状描述（3轴×4统计量） |
| └ `temporal` | `(2,)` | 时间戳 (t_start, t_end) |

#### STEP Token 维度汇总

| Token 类型 | 维度 | 说明 |
|-----------|------|------|
| Patch tokens | 变长，最多 256 | 每个包含 (row, col, iou) |
| Centroid | 3 | (x̄, ȳ, z̄) |
| Shape | 12 | 3轴 × (μ, σ, min, max) |
| Temporal | 2 | (t_start, t_end) |
| **总计（不含 patches）** | **17** | 固定维度几何+时间信息 |

---

### 2.6 迭代细化与 H-hop 几何一致性校验

**论文引用**: Section 3.2, Table 1

#### 输入
| 名称 | 说明 |
|------|------|
| `U_t` | 当前未映射点集 |
| `STEP_tokens` | 当前轮生成的所有 STEP tokens |
| `N_iter` | 迭代次数（默认 1） |
| `H_hop` | 几何校验 hop 数（默认 1） |

#### 迭代细化流程

```python
def iterative_refinement(P_t, images, cameras, N_iter=1, H_hop=1):
    """
    Algorithm 1 的迭代细化部分
    """
    U_t = P_t.copy()  # 初始化未映射点集
    all_step_tokens = []
    assigned_points_all = set()

    for n in range(N_iter):
        if len(U_t) == 0:
            break

        # Step 1: 聚类
        clusters = hdbscan_cluster(U_t)

        # Step 2: 采样 prompts
        prompts = [uniform_sample(c, m=4) for c in clusters]

        # Step 3: 投影 + SAM2 分割
        masks = sam2_segment(images, prompts, cameras)

        # Step 4: 多视角匹配
        matched_masks = hungarian_match(masks)

        # Step 5: 点云关联
        assigned_points = assign_points(P_t, matched_masks, cameras)

        # Step 6: STEP 编码
        step_tokens = [create_step_token(...) for obj in assigned_points]

        # Step 7: H-hop 几何校验
        for h in range(H_hop):
            step_tokens, rejected_points = geometry_validation(step_tokens)
            U_t = np.vstack([U_t, rejected_points]) if len(rejected_points) > 0 else U_t

        # 更新未映射点集
        assigned_indices = get_assigned_indices(assigned_points)
        U_t = P_t[~np.isin(np.arange(len(P_t)), assigned_indices)]

        all_step_tokens.extend(step_tokens)

    return all_step_tokens
```

#### H-hop 几何异常检测

**论文 Table 1 示例**：检测不合理的几何形状，如 "50m 车顶"（elongated Gaussians）

```python
def geometry_validation(step_tokens, thresholds=None):
    """
    H-hop 几何一致性检验

    论文: "detecting implausible geometries (e.g., elongated Gaussians such as
    a 50 m car roof) and reassigning them to U_t"

    检测规则（推断）：
    1. 异常长宽比：某轴范围远大于其他轴
    2. 异常位移：两帧间位移不合理（如行人 32m/2s）
    3. 尺寸突变：同一对象尺寸剧烈变化
    """
    if thresholds is None:
        thresholds = {
            'max_extent': 50.0,        # 单轴最大范围（米）
            'max_aspect_ratio': 20.0,  # 最大长宽比
            'max_velocity': 30.0,      # 最大速度（m/s）
            'max_size_change': 5.0,    # 最大尺寸变化率
        }

    valid_tokens = []
    rejected_points = []

    for token in step_tokens:
        is_valid = True

        # 检查 1: 异常范围
        shape = token.shape  # (12,)
        extents = [shape[3] - shape[2],   # x_max - x_min
                   shape[7] - shape[6],   # y_max - y_min
                   shape[11] - shape[10]] # z_max - z_min

        if max(extents) > thresholds['max_extent']:
            is_valid = False

        # 检查 2: 异常长宽比
        extents_sorted = sorted(extents, reverse=True)
        if extents_sorted[-1] > 0:
            aspect_ratio = extents_sorted[0] / extents_sorted[-1]
            if aspect_ratio > thresholds['max_aspect_ratio']:
                is_valid = False

        # 检查 3: 与前一帧对比（如果有历史）
        # ... 需要跨帧追踪信息

        if is_valid:
            valid_tokens.append(token)
        else:
            if token.points_3d is not None:
                rejected_points.append(token.points_3d)

    rejected_points = np.vstack(rejected_points) if rejected_points else np.empty((0, 3))

    return valid_tokens, rejected_points
```

#### 输出
| 名称 | 说明 |
|------|------|
| `valid_tokens` | 通过校验的 STEP tokens |
| `rejected_points` | 被拒绝的点，重新加入 U_t |

---

### 2.7 4D Scene Graph (4DSG) 构建

**论文引用**: Section 3.3, Eq. 5-8

#### 2.7.1 单帧场景图 (Spatial Scene Graph)

**论文 Eq. 5**:
```
G_t = (V_t, E_t)
```

```python
@dataclass
class SceneGraphNode:
    """场景图节点"""
    node_id: int
    step_token: STEPToken

@dataclass
class SceneGraphEdge:
    """场景图边"""
    source_id: int
    target_id: int
    relation: str           # 空间关系类型
    distance: float         # 3D 距离
    direction: np.ndarray   # 相对方向向量 (3,)

def build_spatial_scene_graph(step_tokens, distance_threshold=10.0):
    """
    构建单帧场景图

    论文: "edges E_t encode spatial relations derived from
    geometric proximity and relative orientation"

    Args:
        step_tokens: List[STEPToken] 当前帧的所有 STEP tokens
        distance_threshold: 建立边的距离阈值（米）

    Returns:
        G_t: (nodes, edges)
    """
    # 创建节点
    nodes = [SceneGraphNode(i, token) for i, token in enumerate(step_tokens)]

    # 创建边（基于空间邻近性）
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            c_i = nodes[i].step_token.centroid
            c_j = nodes[j].step_token.centroid

            distance = np.linalg.norm(c_i - c_j)

            if distance < distance_threshold:
                direction = (c_j - c_i) / (distance + 1e-8)
                relation = compute_spatial_relation(c_i, c_j)

                edges.append(SceneGraphEdge(
                    source_id=i,
                    target_id=j,
                    relation=relation,
                    distance=distance,
                    direction=direction
                ))

    return nodes, edges

def compute_spatial_relation(c_i, c_j):
    """
    计算空间关系（论文未明确具体定义）

    可选实现：
    - 方向量化：left/right/front/back/above/below
    - 距离分级：near/medium/far
    """
    diff = c_j - c_i

    # 水平方向
    if abs(diff[0]) > abs(diff[1]):
        horizontal = "right" if diff[0] > 0 else "left"
    else:
        horizontal = "front" if diff[1] > 0 else "back"

    # 垂直方向
    vertical = "above" if diff[2] > 0.5 else ("below" if diff[2] < -0.5 else "level")

    return f"{horizontal}_{vertical}"
```

#### 2.7.2 时间窗口聚合

**论文 Eq. 6**:
```
G_{t-T:t} = {G_{t-T}, ..., G_t}
```

```python
class TemporalSceneGraphBuffer:
    """时间窗口内的场景图缓冲"""

    def __init__(self, window_size=10):
        self.window_size = window_size  # T = 10
        self.graphs = deque(maxlen=window_size)

    def add_frame(self, G_t):
        """添加新帧的场景图"""
        self.graphs.append(G_t)

    def get_window(self):
        """获取时间窗口内的所有场景图"""
        return list(self.graphs)
```

#### 2.7.3 跨帧对象关联

**论文 Eq. 7**:
```
F_k = {S_{t-T}^k, ..., S_t^k}
```

跨帧关联使用语义 + 3D 几何相似性：

```python
def associate_objects_across_frames(current_tokens, previous_tokens,
                                     semantic_weight=0.5, geometric_weight=0.5):
    """
    跨帧对象关联

    论文: "Each detected object instance k is associated across frames
    by using semantic and 3D spatial cues"

    相似度度量（论文未明确，推断）：
    - 语义相似度：patch token 重叠度 / VLM 特征余弦相似度
    - 几何相似度：质心距离 + 形状相似度
    """
    if not previous_tokens:
        return {i: [t] for i, t in enumerate(current_tokens)}

    # 构建代价矩阵
    n_curr = len(current_tokens)
    n_prev = len(previous_tokens)
    cost_matrix = np.zeros((n_curr, n_prev))

    for i, curr in enumerate(current_tokens):
        for j, prev in enumerate(previous_tokens):
            # 几何相似度：质心距离
            centroid_dist = np.linalg.norm(curr.centroid - prev.centroid)

            # 形状相似度：shape token 欧氏距离
            shape_dist = np.linalg.norm(curr.shape - prev.shape)

            # 语义相似度：patch IoU（简化版）
            patch_sim = compute_patch_overlap(curr.patch_tokens, prev.patch_tokens)

            # 综合代价（越小越好）
            geometric_cost = centroid_dist + 0.1 * shape_dist
            semantic_cost = 1.0 - patch_sim

            cost_matrix[i, j] = (geometric_weight * geometric_cost +
                                  semantic_weight * semantic_cost)

    # 匈牙利匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 建立跨帧序列 F_k
    object_tracks = {}
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < ASSOCIATION_THRESHOLD:
            track_id = previous_tokens[j].object_id
            object_tracks[track_id] = object_tracks.get(track_id, []) + [current_tokens[i]]

    # 处理新出现的对象
    unmatched = set(range(n_curr)) - set(row_ind)
    for i in unmatched:
        new_id = generate_new_object_id()
        object_tracks[new_id] = [current_tokens[i]]

    return object_tracks

def compute_patch_overlap(patches1, patches2):
    """计算 patch tokens 的重叠度"""
    set1 = {(p['row'], p['col']) for p in patches1}
    set2 = {(p['row'], p['col']) for p in patches2}

    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union
```

#### 2.7.4 4DSG 构建

**论文 Eq. 8**:
```
M_t = (G_{t-T:t}, {S_k^{t-T:t}})
```

```python
@dataclass
class FourDSceneGraph:
    """4D Scene Graph"""

    # 时序场景图序列
    spatial_graphs: List[Tuple[List[SceneGraphNode], List[SceneGraphEdge]]]  # G_{t-T:t}

    # 对象轨迹（每个对象的 STEP token 时序）
    object_tracks: Dict[int, List[STEPToken]]  # {S_k^{t-T:t}}

    # Ego 信息
    ego_poses: List[np.ndarray]  # T_{ego}^{t-T:t}, List of (4, 4) SE(3)

    # 全局坐标系（由 SLAM 提供）
    global_frame: str  # 'world' or 'map'

    def query(self, object_id):
        """查询特定对象的时序信息"""
        return self.object_tracks.get(object_id, [])

    def get_all_objects_at_frame(self, frame_idx):
        """获取特定帧的所有对象"""
        return self.spatial_graphs[frame_idx][0]  # nodes

    def get_spatial_relations_at_frame(self, frame_idx):
        """获取特定帧的空间关系"""
        return self.spatial_graphs[frame_idx][1]  # edges

def build_4dsg(temporal_buffer, object_tracks, ego_poses, slam_backend):
    """
    构建完整的 4DSG

    Args:
        temporal_buffer: TemporalSceneGraphBuffer
        object_tracks: 跨帧对象关联结果
        ego_poses: ego 位姿序列
        slam_backend: SLAM 后端（KISS-SLAM 或 MapAnything）

    Returns:
        M_t: FourDSceneGraph
    """
    # 获取时间窗口内的场景图
    spatial_graphs = temporal_buffer.get_window()

    # SLAM 对齐：确保所有坐标在全局参考系下
    aligned_poses = slam_backend.get_aligned_poses()

    return FourDSceneGraph(
        spatial_graphs=spatial_graphs,
        object_tracks=object_tracks,
        ego_poses=ego_poses,
        global_frame='world'
    )
```

#### SLAM 后端配置

```python
# LiDAR 输入：使用 KISS-SLAM
from kiss_icp import KISS_ICP

kiss_slam = KISS_ICP(
    point_cloud_topic="lidar_points",
    # 其他参数参考 KISS-SLAM 官方配置
)

# Image-only 输入：使用 MapAnything
# MapAnything 提供：
# 1. 从图像序列重建的点云
# 2. 相机位姿估计
# 3. 全局坐标对齐
```

---

### 2.8 VLM 推理

**论文引用**: Section 3.4, Eq. 9

**论文 Eq. 9**:
```
ŷ = VLM(q | M_t)
```

#### 输入
| 名称 | 说明 |
|------|------|
| `q` | 查询/问题（文本） |
| `M_t` | 4DSG |

#### STEP Token 序列化为 VLM 输入

论文未明确序列化格式，以下为工程实现建议：

```python
def serialize_4dsg_for_vlm(M_t: FourDSceneGraph, query: str):
    """
    将 4DSG 序列化为 VLM 可理解的输入

    序列化策略（推断）：
    1. 结构化文本描述
    2. JSON 格式
    3. 特定的 token 格式
    """

    # 方法 1: 结构化文本描述
    prompt_parts = []

    # Ego 信息
    prompt_parts.append("=== Ego Agent Information ===")
    for t, pose in enumerate(M_t.ego_poses):
        pos = pose[:3, 3]
        prompt_parts.append(f"Frame {t}: position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    # 对象信息
    prompt_parts.append("\n=== Objects in Scene ===")
    for obj_id, track in M_t.object_tracks.items():
        prompt_parts.append(f"\nObject ID {obj_id}:")
        for token in track:
            c = token.centroid
            s = token.shape
            t_info = token.temporal

            prompt_parts.append(f"  Frame {token.frame_idx}:")
            prompt_parts.append(f"    Position: ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})")
            prompt_parts.append(f"    Size: x=[{s[2]:.2f}, {s[3]:.2f}], y=[{s[6]:.2f}, {s[7]:.2f}], z=[{s[10]:.2f}, {s[11]:.2f}]")
            prompt_parts.append(f"    Visible: frames {int(t_info[0])} to {int(t_info[1])}")

    # 空间关系
    prompt_parts.append("\n=== Spatial Relations ===")
    for t, (nodes, edges) in enumerate(M_t.spatial_graphs):
        prompt_parts.append(f"\nFrame {t}:")
        for edge in edges:
            prompt_parts.append(f"  Object {edge.source_id} is {edge.relation} of Object {edge.target_id} (distance: {edge.distance:.2f}m)")

    # 查询
    prompt_parts.append(f"\n=== Query ===\n{query}")

    return "\n".join(prompt_parts)

def serialize_4dsg_json(M_t: FourDSceneGraph):
    """JSON 格式序列化"""
    return {
        "ego_poses": [pose.tolist() for pose in M_t.ego_poses],
        "objects": {
            obj_id: [{
                "frame": t.frame_idx,
                "centroid": t.centroid.tolist(),
                "shape": t.shape.tolist(),
                "temporal": t.temporal.tolist(),
                "patches": len(t.patch_tokens)
            } for t in track]
            for obj_id, track in M_t.object_tracks.items()
        },
        "spatial_relations": [
            [{
                "source": e.source_id,
                "target": e.target_id,
                "relation": e.relation,
                "distance": e.distance
            } for e in edges]
            for nodes, edges in M_t.spatial_graphs
        ]
    }
```

#### VLM 调用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

def vlm_inference(query, M_t, model_name="google/gemma-3-4b-it"):
    """
    使用 VLM 进行推理

    Args:
        query: 用户查询
        M_t: 4DSG
        model_name: VLM 模型名称

    Returns:
        answer: 模型回答
    """
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 序列化 4DSG
    context = serialize_4dsg_for_vlm(M_t, query)

    # 构建 prompt
    prompt = f"""You are a spatial reasoning assistant analyzing a 4D scene.

{context}

Based on the scene information above, please answer the following question:
{query}

Answer:"""

    # 生成回答
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=256)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer
```

#### H-hop 推理修正（Table 1 示例）

```python
def hhop_reasoning_refinement(M_t, initial_answer, query, H_hop=1):
    """
    H-hop 推理修正：检测并过滤不合理的推理结果

    论文 Table 1 示例：
    Q: What object moved the most in the last 2 seconds?
    初始答案可能包含异常（如行人移动 32m），需要过滤
    """
    for hop in range(H_hop):
        # 检测异常
        anomalies = detect_motion_anomalies(M_t, initial_answer)

        if anomalies:
            # 更新 prompt，排除异常对象
            refined_query = f"""
            {query}

            Note: Exclude the following objects due to potential tracking errors:
            {anomalies}
            """
            initial_answer = vlm_inference(refined_query, M_t)

    return initial_answer

def detect_motion_anomalies(M_t, answer):
    """检测运动异常"""
    anomalies = []

    for obj_id, track in M_t.object_tracks.items():
        if len(track) < 2:
            continue

        # 计算帧间位移
        for i in range(1, len(track)):
            prev_c = track[i-1].centroid
            curr_c = track[i].centroid
            displacement = np.linalg.norm(curr_c - prev_c)

            # 假设帧率 1 Hz，检测不合理速度
            if displacement > 30:  # > 30 m/s
                anomalies.append({
                    'object_id': obj_id,
                    'displacement': displacement,
                    'reason': 'implausible velocity'
                })

    return anomalies
```

---

## 3. Algorithm 1 完整伪代码

```
Algorithm 1: 4D Spatio-Temporal Scene Understanding with SNOW and STEP Encoding

Require:
    Point clouds {P_t}_{0:T}
    Image sequence {I_t^c}_{0:T}
    Temporal window T = 10
    Iterations N_iter = 1
    Reasoning hops H_hop = 1
    VLM backbone (Gemma3-4B-IT)

1:  Initialize persistent 4DSG: M_0 ← ∅
2:  Initialize object tracker: Tracker ← ∅

3:  for each time step t do
4:      # === Phase 1: Point Cloud Processing ===
5:      Initialize unmapped points: U_t ← P_t
6:
7:      for n = 1 to N_iter do
8:          # Clustering
9:          R_t = HDBSCAN(U_t)  # Eq. 1: {R_t^1, ..., R_t^K}
10:
11:         # Sample prompts
12:         for each cluster R_t^k do
13:             V_t^k = UniformSample(R_t^k, m=4)
14:         end for
15:
16:         # === Phase 2: Multi-view Segmentation ===
17:         for each camera c do
18:             # Project points to image
19:             {P_t}_img^c = π(P_t, I_t^c)  # Eq. 2
20:
21:             # SAM2 segmentation with point prompts
22:             for each cluster k do
23:                 m_t^{k,c} = SAM2({V_t^k}_img, I_t^c)  # Eq. 3
24:             end for
25:         end for
26:
27:         # Multi-view consistency matching
28:         {m_t^k} = HungarianMatch({m_t^{k,c}}_c)
29:
30:         # === Phase 3: 3D-2D Association ===
31:         for each object k do
32:             R̂_t^k = AssignPoints(P_t, m_t^k)
33:         end for
34:
35:         # === Phase 4: STEP Encoding ===
36:         for each object k do
37:             # Compute STEP tokens (Eq. 4)
38:             patch_tokens = ComputePatchTokens(m_t^k, G=16, IoU>0.5)
39:             c_t^k = Centroid(R̂_t^k)           # (x̄, ȳ, z̄)
40:             s_t^k = ShapeToken(R̂_t^k)         # {(μ_a, σ_a, a_min, a_max)}
41:             θ_t^k = TemporalToken(t_start, t_end)
42:
43:             S_t^k = {patch_tokens, c_t^k, s_t^k, θ_t^k}
44:         end for
45:
46:         # === Phase 5: H-hop Geometric Validation ===
47:         for h = 1 to H_hop do
48:             for each S_t^k do
49:                 if IsImplausibleGeometry(S_t^k) then
50:                     U_t = U_t ∪ R̂_t^k  # Reassign to unmapped
51:                     Remove S_t^k
52:                 end if
53:             end for
54:         end for
55:
56:         # Update unmapped points
57:         U_t = P_t \ ∪_k R̂_t^k
58:
59:         if U_t = ∅ then break
60:     end for
61:
62:     # === Phase 6: Scene Graph Construction ===
63:     # Build spatial scene graph (Eq. 5)
64:     V_t = {v_t^k | v_t^k corresponds to S_t^k}
65:     E_t = ComputeSpatialEdges(V_t)  # proximity + orientation
66:     G_t = (V_t, E_t)
67:
68:     # Temporal aggregation (Eq. 6)
69:     G_{t-T:t} = TemporalBuffer.Add(G_t)
70:
71:     # Cross-frame association (Eq. 7)
72:     F_k = Tracker.Associate({S_t^k}, semantic_cues, geometric_cues)
73:
74:     # Build 4DSG (Eq. 8)
75:     M_t = (G_{t-T:t}, {F_k})
76:     M_t = SLAMAlign(M_t, ego_pose_t)  # Global reference alignment
77:
78:     # === Phase 7: VLM Inference ===
79:     if query q is received then
80:         ŷ = VLM(q | M_t)  # Eq. 9
81:         return ŷ
82:     end if
83:
84: end for
```

---

## 4. 实验配置与超参数

### 4.1 通用配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `T` | 10 | 时间窗口（帧），约 10 秒 |
| `N_iter` | 1 | 迭代细化次数 |
| `H_hop` | 1 | 几何校验 hop 数 |
| `m` | 4 | 每簇采样点数 |
| `G` | 16 | STEP 网格尺寸 |
| `IoU_thresh` | 0.5 | Patch 保留阈值 |

### 4.2 模型配置

| 组件 | 配置 | 备注 |
|------|------|------|
| SAM2 | Hiera Large video predictor | `sam2_hiera_l.yaml` |
| VLM | Gemma3-4B-IT | HuggingFace: `google/gemma-3-4b-it` |
| SLAM (LiDAR) | KISS-SLAM | 开源：github.com/PRBonn/kiss-icp |
| SLAM (image) | MapAnything | 论文 [22] |

### 4.3 硬件配置

| 资源 | 配置 |
|------|------|
| GPU | NVIDIA H100 |
| 处理速度 | ~1.1 FPS（随对象数增加而降低，见 Figure 8） |
| 内存 | 建议 ≥40GB VRAM |

### 4.4 推荐的 HDBSCAN 参数（论文未明确）

| 场景 | `min_cluster_size` | `min_samples` | `epsilon` |
|------|-------------------|---------------|-----------|
| 驾驶场景（NuScenes） | 30-50 | 5-10 | 0.0 |
| 室内场景（RoboSpatial） | 20-30 | 3-5 | 0.0 |

---

## 5. 关键张量/对象 I/O 汇总

| 组件 | 输入 | 输出 | 形状/维度 |
|------|------|------|-----------|
| **点云输入** | Raw LiDAR/重建 | `P_t` | `(N, 3)` |
| **HDBSCAN** | `U_t` | `R_t = {R_t^k}` | K 个簇，每个 `(N_k, 3)` |
| **采样** | `R_t^k` | `V_t^k` | `(4, 3)` |
| **投影** | `P_t`, 相机参数 | `(x_img, y_img)` | `(N, 2)` |
| **SAM2** | `I_t^c`, `{V_t^k}_img` | `m_t^{k,c}` | `(H, W)` bool |
| **匈牙利匹配** | 多视角 masks | 匹配对 | `List[(idx1, idx2)]` |
| **点云关联** | mask + 投影 | `R̂_t^k` | `(N_k, 3)` |
| **Patch Tokens** | mask, G=16 | `{τ_{k,j}^t}` | 最多 256 个 |
| **Centroid** | `R̂_t^k` | `c_t^k` | `(3,)` |
| **Shape** | `R̂_t^k` | `s_t^k` | `(12,)` |
| **Temporal** | 帧信息 | `θ_t^k` | `(2,)` |
| **STEP Token** | 以上组合 | `S_t^k` | patches + 17 |
| **Scene Graph** | `{S_t^k}` | `G_t = (V_t, E_t)` | 节点 + 边 |
| **4DSG** | `{G_t}_{t-T:t}` | `M_t` | 4D 场景图 |
| **VLM** | `q`, `M_t` | `ŷ` | 文本答案 |

---

## 6. 代码模块设计建议

### 6.1 推荐目录结构

```
snow/
├── fast_snow/engine/config/
│   ├── default.yaml           # 默认配置
│   └── nuscenes.yaml          # NuScenes 特定配置
├── fast_snow/data/
│   ├── __init__.py
│   ├── point_cloud.py         # 点云数据加载
│   ├── image.py               # 图像数据加载
│   └── calibration.py         # 标定数据处理
├── models/
│   ├── __init__.py
│   ├── hdbscan_cluster.py     # HDBSCAN 聚类
│   ├── sam2_segmenter.py      # SAM2 分割封装
│   ├── step_encoder.py        # STEP 编码
│   ├── scene_graph.py         # 场景图构建
│   └── vlm_inference.py       # VLM 推理
├── utils/
│   ├── __init__.py
│   ├── projection.py          # 投影工具
│   ├── matching.py            # 匈牙利匹配
│   ├── geometry.py            # 几何计算
│   └── visualization.py       # 可视化
├── fast_snow/vision/slam/
│   ├── __init__.py
│   ├── kiss_slam.py           # KISS-SLAM 接口
│   └── mapanything.py         # MapAnything 接口
├── fast_snow/engine/pipeline/
│   ├── __init__.py
│   └── snow_pipeline.py       # 完整 pipeline
├── evaluation/
│   ├── __init__.py
│   ├── nuscenes_qa.py         # NuScenes-QA 评测
│   ├── robospatial.py         # RoboSpatial 评测
│   ├── vlm4d.py               # VLM4D 评测
│   └── lidar_seg.py           # LiDAR 分割评测
└── scripts/
    ├── run_inference.py       # 推理脚本
    └── run_evaluation.py      # 评测脚本
```

### 6.2 核心类接口

```python
# snow/models/step_encoder.py
class STEPEncoder:
    def __init__(self, grid_size=16, iou_threshold=0.5):
        self.grid_size = grid_size
        self.iou_threshold = iou_threshold

    def encode(self, mask, points_3d, t_start, t_end) -> STEPToken:
        """编码单个对象为 STEP token"""
        pass

# snow/models/scene_graph.py
class SceneGraphBuilder:
    def __init__(self, distance_threshold=10.0):
        self.distance_threshold = distance_threshold

    def build_spatial_graph(self, step_tokens) -> Tuple[List, List]:
        """构建空间场景图"""
        pass

    def build_4dsg(self, temporal_graphs, object_tracks, ego_poses) -> FourDSceneGraph:
        """构建 4DSG"""
        pass

# snow/fast_snow/engine/pipeline/snow_pipeline.py
class SNOWPipeline:
    def __init__(self, config):
        self.clusterer = HDBSCANClusterer(config.hdbscan)
        self.segmenter = SAM2Segmenter(config.sam2)
        self.step_encoder = STEPEncoder(config.step)
        self.graph_builder = SceneGraphBuilder(config.graph)
        self.vlm = VLMInference(config.vlm)
        self.slam = SLAMBackend(config.slam)

    def process_frame(self, point_cloud, images, cameras):
        """处理单帧数据"""
        pass

    def query(self, question) -> str:
        """回答查询"""
        pass
```

---

## 7. 评测协议与数据集

### 7.1 NuScenes-QA

| 指标 | 说明 |
|------|------|
| Existence (Ext) | 物体存在性判断 |
| Count (Cnt) | 物体计数 |
| Object (Obj) | 物体识别 |
| Status (Sts) | 物体状态（运动/静止等） |
| Comparison (Cmp) | 比较关系 |
| Accuracy (Acc) | 总体准确率 |

**SNOW 结果**: 60.1% 总体准确率，Status 类提升 +23.5%

### 7.2 RoboSpatial-Home

| 指标 | 说明 |
|------|------|
| Spatial Configuration | 物体间相对空间关系 |
| Spatial Context | 空间定位（点预测） |
| Spatial Compatibility | 空间可行性判断 |

**SNOW 结果**: 72.29% 平均准确率，Context 类提升 +23.82%

### 7.3 VLM4D

| 指标 | 说明 |
|------|------|
| Ego-centric | 自中心空间推理 |
| Exo-centric | 他中心空间推理 |
| Directional | 方向推理 |
| False Positive | 假阳性检测 |

**SNOW 结果**: 73.75% 总体准确率，Directional 提升 +16.36%

### 7.4 NuScenes LiDAR Segmentation

| 指标 | 说明 |
|------|------|
| mIoU | Mean Intersection over Union |

**SNOW 结果**: 38.1 mIoU（training-free，排名第二）

---

## 8. 论文未明确需工程决策的细节

以下内容论文未给出明确数值或实现细节，复现时需要做工程决策：

| 编号 | 细节 | 建议实现 |
|------|------|----------|
| 1 | HDBSCAN 参数 | `min_cluster_size=30-50`, `min_samples=5-10` |
| 2 | 匈牙利匹配代价函数 | 3D 质心距离 + 形状相似度 |
| 3 | 几何异常判定阈值 | 最大范围 50m，最大长宽比 20，最大速度 30m/s |
| 4 | 4DSG 边定义 | 距离阈值 10m，方向量化 6 向 |
| 5 | Patch token 语义嵌入 | 使用 VLM patch 特征或简化为位置 IoU |
| 6 | VLM 输入序列化 | 结构化文本或 JSON |
| 7 | 跨帧关联相似度 | 0.5 × 几何 + 0.5 × 语义 |
| 8 | 关联阈值 | 根据场景调参 |

---

## 9. 复现 Checklist

### Phase 1: 数据准备
- [ ] 准备 NuScenes 数据集（点云 + 图像 + 标定）
- [ ] 准备 RoboSpatial-Home 数据集
- [ ] 准备 VLM4D 数据集
- [ ] 验证传感器同步和标定

### Phase 2: 核心组件
- [ ] 实现 HDBSCAN 聚类模块
- [ ] 实现点采样模块（FPS 或均匀采样）
- [ ] 实现相机投影模块
- [ ] 集成 SAM2 Hiera Large
- [ ] 实现多视角匈牙利匹配
- [ ] 实现 3D-2D 点关联

### Phase 3: STEP 编码
- [ ] 实现 16×16 网格 patch token 提取
- [ ] 实现 centroid token 计算
- [ ] 实现 shape token 计算（高斯统计）
- [ ] 实现 temporal token 管理
- [ ] 组装完整 STEP token

### Phase 4: 场景图
- [ ] 实现空间场景图构建
- [ ] 实现时间窗口管理
- [ ] 实现跨帧对象关联
- [ ] 实现 4DSG 构建
- [ ] 集成 SLAM 后端（KISS-SLAM / MapAnything）

### Phase 5: VLM 推理
- [ ] 实现 4DSG 序列化
- [ ] 集成 Gemma3-4B-IT
- [ ] 实现 H-hop 推理修正

### Phase 6: 评测
- [ ] NuScenes-QA 评测脚本
- [ ] RoboSpatial-Home 评测脚本
- [ ] VLM4D 评测脚本
- [ ] LiDAR 分割评测脚本

### Phase 7: 验证
- [ ] 验证单帧处理速度（目标：~1.1 FPS on H100）
- [ ] 验证各 benchmark 结果与论文一致

---

## 10. 版本记录

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | - | 基于论文正文 + Supplementary 初始整理 |
| v2.0 | 2025-01 | 完善工程实现细节：添加完整数学公式、代码示例、模块接口设计、评测协议 |
