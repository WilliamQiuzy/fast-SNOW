# Fast-SNOW 模型权重下载与完整运行（简版）

本文档默认你在仓库根目录执行命令：
`/Users/williamqiu/Desktop/Harvard/HUAWEI/fast-SNOW`

## 1) 准备目录

```bash
mkdir -p fast_snow/models/{sam3,ram_plus,da3}
```

## 2) 下载模型权重

先安装下载工具：

```bash
pip install -U "huggingface_hub[cli]"
```

### 2.1 SAM3（需要先申请 HF 访问权限）

1. 先在 HF 申请并获得 `facebook/sam3` 访问权限。  
2. 登录：

```bash
hf auth login
```

3. 下载到本地默认目录：

```bash
huggingface-cli download facebook/sam3 sam3.pt config.json \
  --local-dir fast_snow/models/sam3
```

### 2.2 RAM++（image tagging）

```bash
huggingface-cli download xinyu1205/recognize-anything-plus-model ram_plus_swin_large_14m.pth \
  --local-dir fast_snow/models/ram_plus
```

### 2.3 DA3（推荐：DA3NESTED-GIANT-LARGE-1.1）

```bash
huggingface-cli download depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
  --local-dir fast_snow/models/da3
```

说明：Fast-SNOW 当前 Step2 需要 metric depth + 相机参数，建议用上面的 NESTED 模型。

## 3) 检查文件是否到位

```bash
ls -lh fast_snow/models/sam3
ls -lh fast_snow/models/ram_plus
ls -lh fast_snow/models/da3 | head
```

关键文件至少应有：
- `fast_snow/models/sam3/sam3.pt`
- `fast_snow/models/ram_plus/ram_plus_swin_large_14m.pth`
- `fast_snow/models/da3/` 下完整 HF 模型文件（config + 权重）

## 4) 运行完整链路（视频 -> 4DSG -> VLM）

如果需要最终问答（Step8 调 VLM）：

```bash
export PYTHONPATH=.
export GOOGLE_AI_API_KEY=你的key
python - <<'PY'
from fast_snow.engine.pipeline import FastSNOWEndToEnd
e2e = FastSNOWEndToEnd()
result = e2e.process_video("path/to/video.mp4", "What is in front of the car?")
print(result.answer)
print(result.scene_json[:1000])
result.cleanup()
PY
```

如果只先跑到 4DSG（不调 VLM）：

```bash
export PYTHONPATH=.
python - <<'PY'
from fast_snow.engine.pipeline import FastSNOWEndToEnd
e2e = FastSNOWEndToEnd()
result = e2e.build_4dsg_from_video("path/to/video.mp4")
print(result.scene_json[:1000])
result.cleanup()
PY
```

## 5) 当前测试命令（代码层）

```bash
export PYTHONPATH=.
pytest -q scripts/test_fast_snow_step01.py scripts/test_fast_snow_pipeline.py scripts/test_fast_snow_smoke.py
```
