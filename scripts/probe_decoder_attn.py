"""Quickly inspect the decoder transformer's attention configuration."""
from __future__ import annotations
import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool

cfg = ROSEConfig()
cfg.da3.model_path = "rose/models/da3-small"
cfg.sam3.use_fa3 = True; cfg.sam3.use_multiplex = True
cfg.sam3.enable_compile = False; cfg.sam3.compile_memory_encoder = False
pool = WarmModelPool(cfg); pool.load_all()

dec = pool._sam3._predictor.model.tracker.model.sam_mask_decoder
tfm = dec.transformer
print(f"transformer: {type(tfm).__name__}")
print(f"  num layers: {len(tfm.layers)}")
for i, layer in enumerate(tfm.layers):
    print(f"  layer[{i}]: {type(layer).__name__}")
    for name in ["self_attn", "cross_attn_token_to_image", "cross_attn_image_to_token"]:
        attn = getattr(layer, name, None)
        if attn is not None:
            print(f"    .{name}: {type(attn).__name__}  use_fa3={getattr(attn, 'use_fa3', None)}")
fa = getattr(tfm, "final_attn_token_to_image", None)
if fa:
    print(f"  final_attn_token_to_image: {type(fa).__name__}  use_fa3={getattr(fa, 'use_fa3', None)}")
