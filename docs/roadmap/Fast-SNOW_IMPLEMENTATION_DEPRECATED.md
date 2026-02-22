# Fast-SNOW Implementation Spec (Deprecated Approaches)

本文件记录已弃用路线，仅用于历史对照。

---

## Deprecated 1: RAM++ + SAM3 主路线

- 原主路线：`RAM++ + SAM3`
- 弃用原因：在当前实验中，RAM++ 文本语义到 SAM3 的对齐稳定性不足，且跨帧追踪/一致性难以稳定保证。
- 替代方案：`YOLO11n bbox -> SAM3`

---

## Deprecated 2: 图像-crop STEP Token（早期尝试）

- 弃用日期：2026-02-20
- 弃用原因：该方案将 `tau` 扩展为 patch crop 图像路径并以视觉 token 喂给 VLM，但当前主线已收敛到文本化 `tau`（`row,col,iou`）+ `visual_anchor` 的融合方式。

### v1 实现（已弃用）

**PatchToken 定义**：
```python
@dataclass(frozen=True)
class PatchToken:
    row: int      # 16×16 网格的行索引
    col: int      # 16×16 网格的列索引
    iou: float    # mask 在该 cell 内的覆盖率
```

**JSON 序列化**：
```json
{
  "tau": [{"row":3,"col":4,"iou":0.71}, {"row":3,"col":5,"iou":0.66}],
  "c": [5.12, -2.30, 1.05],
  "s": {"x": {"mu":5.12, ...}, ...},
  "theta": [3, 8]
}
```

**说明**：
1. 当时的实现将 `tau` 定义为 `row/col/iou + patch path`，需要额外管理 patch 物理文件，工程复杂度高。
2. 当前主实现已改为仅保留文本 `tau`，并在 VLM 推理时统一依赖 `visual_anchor`（关键帧图像）与几何 text token（`c/s/θ`）。

### 替代方案（当前主文档）

见主文档 `Fast-SNOW_IMPLEMENTATION.md` 的 §4.1 和 Step 6/8 更新：`tau` 为 `(row,col,iou)` 文本 token，`visual_anchor` 作为可视化上下文。

---

当前主路线请以 `Fast-SNOW_IMPLEMENTATION.md` 为准。
