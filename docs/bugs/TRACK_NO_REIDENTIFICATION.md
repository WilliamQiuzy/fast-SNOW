# 归档 Track 无法重新识别（ID 碎片化）

## 状态：Open

## 问题描述

当一个物体连续 `lost_patience + archive_patience`（默认 5 + 30 = 35）帧未被检测到时，其 track 状态变为 `archived`。一旦 track 被归档，即使该物体后来重新出现在画面中，pipeline 会为它**分配一个全新的 global ID**，而不是将其关联回原来的 track。

这导致**同一个物体在 4DSG 中拥有多个独立的 track**（ID 碎片化），VLM 无法知道 track 5 和 track 12 其实是同一个物体。

## 影响

### 直接后果

1. **同一物体多个 ID**：一个物体在 4DSG 中可能被表示为 `object_id: 3`（帧 0-20）和 `object_id: 8`（帧 60-100），VLM 认为它们是两个不同的物体。
2. **运动轨迹断裂**：物体的时空轨迹在 `theta` 中被截断，无法反映完整的运动历史。
3. **track 数量膨胀**：频繁进出画面的物体（如交通中的车辆）会产生大量 track，增加 4DSG 的 token 消耗。
4. **VLM 回答质量下降**：VLM 需要回答"这个物体做了什么"时，看到的是两段不连续的轨迹，无法给出完整答案。

### 严重程度

**Medium**——对于短视频或物体稳定出现的场景影响不大，但对长视频和动态场景影响显著。

## 根因分析

### Track 状态机

```python
# fast_snow_pipeline.py:95-103
def miss(self, cfg: FastSNOWConfig) -> None:
    if self.status == "archived":
        return
    self.missing_streak += 1
    if self.missing_streak >= cfg.fusion.lost_patience + cfg.fusion.archive_patience:
        self.status = "archived"         # ← 35 帧未见，永久归档
    elif self.missing_streak >= cfg.fusion.lost_patience:
        self.status = "lost"             # ← 5 帧未见，标记丢失
```

状态转换：

```
active ──(5帧未见)──→ lost ──(再30帧未见)──→ archived ──(永久，不可逆)
  ↑                    ↓
  └──(重新检测到)──────┘
```

注意：`lost` 状态可以通过 `observe()` 恢复为 `active`，但 `archived` 是终态。

### 归档 Track 的排除逻辑

在 candidate fusion 阶段（Step 5），归档 track 被显式排除在匹配之外：

```python
# fast_snow_pipeline.py:296-312
# Assign provisional global IDs by local (run_id, obj_id)
# Archived tracks do NOT participate in matching (spec §5):
# if the old gid maps to an archived track, allocate a fresh gid.
gid = self._local_to_global.get(key)
if gid is not None:
    track = self._tracks.get(gid)
    if track is not None and track.status == "archived":
        gid = None  # force new allocation
if gid is None:
    gid = self._allocate_global_id()
    self._local_to_global[key] = gid
```

同时在 cross-run fusion 中，归档 track 也被跳过：

```python
# fast_snow_pipeline.py:338-352
ti = self._tracks.get(gi)
if ti is not None and ti.status == "archived":
    continue        # ← 归档 track 不参与 merge
...
tj = self._tracks.get(gj)
if tj is not None and tj.status == "archived":
    continue        # ← 归档 track 不参与 merge
```

### 为什么不直接让归档 track 可重用？

`_TrackState.observe()` 方法显式阻止归档 track 被重新使用：

```python
# fast_snow_pipeline.py:83-88
def observe(self, frame_idx: int, step: STEPToken, centroid: np.ndarray) -> None:
    if self.status == "archived":
        raise RuntimeError(
            f"Bug: observe() called on archived track {self.track_id}. "
            f"Archived tracks must not be re-identified."
        )
```

这是一个主动的设计选择，但没有配套的 re-identification 机制来解决物体重新出现的问题。

## 核心矛盾

```
物体暂时消失超过 35 帧 ──→ track 归档（不可逆）
                                     │
物体重新出现 ──→ 无法匹配到归档 track ──→ 分配新 ID
                                     │
结果 ──→ 同一物体拥有多个 track ID
```

## 可能的解决方向

### 方向 1：基于外观特征的 re-identification

在归档 track 时保存物体的外观特征（如 STEP patch tokens 的均值、3D shape 统计量），当新 track 出现时与归档 track 做特征匹配：

```python
# 伪代码
for new_track in new_candidates:
    for archived_track in archived_tracks:
        if appearance_similarity(new_track, archived_track) > threshold:
            # 复活归档 track，继承其 ID
            reactivate(archived_track, new_track)
            break
```

**匹配特征**：
- 3D shape (s) 的统计相似度
- 3D centroid (c) 的位置接近度（考虑物体可能移动）
- SAM3 local_obj_id 的一致性

**优点**：能正确关联重新出现的物体。

**风险**：
- 特征匹配可能产生 false positive（不同物体被误认为同一个）。
- 需要为归档 track 维护特征缓存，增加内存开销。

### 方向 2：增大 archive_patience

增加 `archive_patience` 的值（如 100 帧），让 track 在更长时间内保持 lost 状态而不被归档。lost 状态可以通过 SAM3 local_obj_id 重新匹配。

**优点**：简单。

**风险**：
- lost track 仍然消耗内存。
- 只是延迟问题而非解决问题。

### 方向 3：基于 SAM3 obj_id 的连续性

SAM3 为每个物体分配唯一的 `obj_id_local`。即使物体暂时消失，SAM3 可能在物体重新出现时返回相同的 `obj_id_local`。利用这个特性，在 `_local_to_global` 映射中保持归档 track 的关联：

```python
# 修改 _fuse_candidates 中的归档 track 处理
if track is not None and track.status == "archived":
    if same_sam3_obj_id:  # SAM3 给出了相同的 local obj id
        track.status = "active"  # 复活
        gid = track.track_id     # 重用 ID
```

**优点**：利用 SAM3 已有的物体一致性。

**风险**：
- 取决于 SAM3 的 obj_id 是否在物体消失/重现时保持一致（在当前实现中，由于 propagate 时使用 `assign_new_obj_to_run=False`，SAM3 auto-discovered 的物体会被拒绝，所以这个方向可能不可行）。

## 相关文件

- [fast_snow_pipeline.py:74-103](fast_snow/engine/pipeline/fast_snow_pipeline.py#L74-L103) — `_TrackState` 定义和状态转换
- [fast_snow_pipeline.py:83-88](fast_snow/engine/pipeline/fast_snow_pipeline.py#L83-L88) — `observe()` 对归档 track 的 RuntimeError
- [fast_snow_pipeline.py:292-312](fast_snow/engine/pipeline/fast_snow_pipeline.py#L292-L312) — candidate fusion 中归档 track 的排除逻辑
- [fast_snow_pipeline.py:338-352](fast_snow/engine/pipeline/fast_snow_pipeline.py#L338-L352) — cross-run fusion 中归档 track 被跳过
- [fast_snow_config.py:103-109](fast_snow/engine/config/fast_snow_config.py#L103-L109) — `FusionConfig` 参数：`lost_patience=5`, `archive_patience=30`
