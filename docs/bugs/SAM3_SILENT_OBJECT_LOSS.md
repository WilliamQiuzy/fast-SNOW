# SAM3 连续空帧静默丢失物体

## 状态：Mitigated（FastSAM discovery 缓解，但未完全消除）

## 问题描述

当 SAM3 在多个连续帧中返回 0 个 mask 时（`frame_masks = []`），Fast-SNOW 仅记录日志警告，**不触发任何恢复机制**。如果这种状态持续超过 `lost_patience + archive_patience`（默认 35 帧），所有正在跟踪的 track 都会被归档，等同于**静默丢失所有物体**。

此后即使 SAM3 恢复输出 mask，这些物体也会被分配全新的 ID（参见 [TRACK_NO_REIDENTIFICATION.md](TRACK_NO_REIDENTIFICATION.md)）。

**缓解措施**（已实现）：Two-pass 架构的 Phase 2b 使用 FastSAM 逐帧检测，与 SAM3 缓存的 mask 做 IoU 比对，发现 SAM3 丢失的物体后通过 `add_object_point()` 重新添加到 tracker。这大幅降低了静默丢失的概率，但如果 SAM3 propagation 本身 OOM 导致全部 runs 被 ended，则所有物体仍会丢失。

## 影响

### 直接后果

1. **所有物体 track 断裂**：连续 35 帧空 mask 后，所有 track 归档。恢复后全部重新分配 ID。
2. **4DSG 中出现大量短 track**：原本连续的物体轨迹被切割成多个独立的短 track。
3. **静默失败**：没有异常抛出、没有告警升级、没有回退机制——pipeline 继续运行但数据质量已严重退化。

### 严重程度

**Medium-High**——在正常场景中 SAM3 不太可能连续 35 帧返回空 mask，但以下场景可以触发：
- 快速相机运动导致帧间变化过大，SAM3 跟丢所有物体
- SAM3 内部 score 在某些帧序列中系统性低于 `score_threshold_detection`
- SAM3 propagation 异常（如 action_history 进入 fetch-only 模式但 cache 为空）

## 根因分析

### 当前的空帧处理逻辑

```python
# fast_snow_e2e.py:397-409

# Track consecutive empty frames for diagnostics.
if frame_masks:
    self._consecutive_empty_frames = 0
else:
    self._consecutive_empty_frames += 1
    if self._consecutive_empty_frames in (1, 5, 10, 20, 50):
        logger.warning(
            "SAM3 returned 0 masks for %d consecutive frame(s) "
            "(current frame %d). This is normal if objects are "
            "temporarily suppressed by SAM3's internal tracker.",
            self._consecutive_empty_frames,
            sam3_frame_idx,
        )
```

这段代码只在特定的连续空帧数（1, 5, 10, 20, 50）打印 WARNING 日志。不会：
- 尝试恢复 SAM3 状态
- 降低 score_threshold 重新过滤
- 通知 pipeline 层做特殊处理
- 触发 re-prompt 或 session 重建

### 下游影响链路

```
SAM3 连续返回空 mask
    ↓
FastFrameInput.detections = [] （无检测）
    ↓
process_frame() 中 candidates 为空 → winners 为空
    ↓
所有现有 track 调用 miss() → missing_streak++
    ↓
5 帧后: 所有 track → "lost"
35 帧后: 所有 track → "archived"
    ↓
SAM3 恢复 mask 输出 → 新 ID 分配 → track 碎片化
```

### 与 SAM3 内部机制的关系

SAM3 的跟踪器有"物体抑制"机制：当 tracker 不确定某个物体是否仍在画面中时，会暂时抑制其 mask 输出（score 降至阈值以下）。这在短期内（1-3 帧）是正常行为。

但如果抑制持续过长，说明 SAM3 的跟踪状态可能已经严重退化，此时应该考虑恢复措施。

### 日志消息的误导性

当前日志消息说 "This is normal if objects are temporarily suppressed"，但**连续 20+ 帧的空 mask 不是正常的 suppression**——这通常意味着跟踪已经失败。日志消息没有区分这两种情况。

## 可能的解决方向

### 方向 1：连续空帧阈值触发告警升级

设定一个阈值（如 `max_consecutive_empty = 10`），超过后：
- 日志级别从 WARNING 升级到 ERROR
- 可选：抛出异常让调用方决定是否继续

```python
MAX_CONSECUTIVE_EMPTY = 10

if self._consecutive_empty_frames >= MAX_CONSECUTIVE_EMPTY:
    logger.error(
        "SAM3 has returned 0 masks for %d consecutive frames. "
        "Tracking may have failed. Consider re-initializing.",
        self._consecutive_empty_frames,
    )
    # 可选: raise RuntimeError(...)
```

**优点**：让用户/调用方早期感知问题。

**风险**：false alarm——某些场景可能确实有长时间的空帧（如室内到室外的场景切换）。

### 方向 2：降低 score_threshold 重试

在连续空帧超过阈值后，临时降低 `score_threshold_detection` 重新过滤 propagation 结果：

```python
if self._consecutive_empty_frames > 5:
    # 尝试用更低阈值重新获取 mask
    relaxed_masks = self._sam3.propagate_all(
        sam3_frame_idx,
        score_threshold=0.1,  # 放宽阈值
    )
```

**优点**：可能恢复被过度抑制的物体。

**风险**：
- 当前 `propagate_all()` 不支持临时的 score_threshold 参数。
- 降低阈值可能引入大量噪声 mask。

### 方向 3：SAM3 session 重建

在连续空帧超过严重阈值（如 20 帧）后，结束当前 SAM3 session 并用最近的 YOLO 检测结果重新初始化：

```python
if self._consecutive_empty_frames > 20:
    # 重建 SAM3 session
    self._sam3.end_all_runs()
    self._sam3.set_video_dir(frame_dir)
    # 用当前帧的 FastSAM 检测重新初始化
    if fastsam_dets:
        self._sam3.create_run_with_initial_bboxes(...)
        self._sam3_initialized = True
```

**优点**：完全重置跟踪状态，有机会恢复。

**风险**：
- `set_video_dir()` 会重置所有 SAM3 内部状态。
- 重建后的 SAM3 从当前帧开始跟踪，无法与之前的 track 关联。
- 与 SAM3_NO_REANCHOR 问题叠加——重建本质上是全新的 session。

### 方向 4：监控 + pipeline 层面处理

在 pipeline 层（`fast_snow_pipeline.py`）增加对"全帧空检测"模式的特殊处理：当检测到所有 track 同时进入 lost 状态时，冻结 miss counter 而非继续累加。

**优点**：防止短暂的跟踪失败导致所有 track 归档。

**风险**：可能掩盖真正的跟踪失败，导致归档延迟但不解决根本问题。

## 相关文件

- [fast_snow_e2e.py:397-409](fast_snow/engine/pipeline/fast_snow_e2e.py#L397-L409) — 连续空帧计数和日志
- [fast_snow_e2e.py:98-100](fast_snow/engine/pipeline/fast_snow_e2e.py#L98-L100) — `_consecutive_empty_frames` 初始化
- [fast_snow_pipeline.py:95-103](fast_snow/engine/pipeline/fast_snow_pipeline.py#L95-L103) — `miss()` 方法和归档逻辑
- [sam3_shared_session_wrapper.py:274-353](fast_snow/vision/perception/sam3_shared_session_wrapper.py#L274-L353) — `propagate_all()` 和 score filtering
- [TRACK_NO_REIDENTIFICATION.md](TRACK_NO_REIDENTIFICATION.md) — 归档后 ID 碎片化的关联问题
