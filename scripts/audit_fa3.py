"""Walk the SAM3 model tree and print every module that has a `use_fa3`
attribute, with its current value. Also count attention modules.
"""
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
cfg.sam3.use_fa3 = True
cfg.sam3.use_multiplex = True
cfg.sam3.enable_compile = False
cfg.sam3.compile_memory_encoder = False  # see attn config in raw state
cfg.sam3.compile_mask_decoder_transformer = False

pool = WarmModelPool(cfg); pool.load_all()

sam3_root = pool._sam3._predictor.model  # the Sam3MultiplexTrackingWithInteractivity

print("\n=== Walking SAM3 root for use_fa3 attributes ===")
counts = {True: 0, False: 0}
by_path = []
for name, module in sam3_root.named_modules():
    if hasattr(module, "use_fa3"):
        val = getattr(module, "use_fa3")
        counts[bool(val)] += 1
        by_path.append((name, type(module).__name__, val))

print(f"Total modules with use_fa3 attribute: {len(by_path)}")
print(f"  use_fa3=True : {counts[True]}")
print(f"  use_fa3=False: {counts[False]}")

# Group by class name for summary
from collections import defaultdict
by_class = defaultdict(lambda: {"T": 0, "F": 0})
for name, cls, val in by_path:
    by_class[cls]["T" if val else "F"] += 1
print("\nBy class:")
for cls in sorted(by_class):
    c = by_class[cls]
    print(f"  {cls:<40} use_fa3=True: {c['T']:<3} use_fa3=False: {c['F']:<3}")

# Group by top-level path prefix
by_prefix = defaultdict(lambda: {"T": 0, "F": 0})
for name, cls, val in by_path:
    prefix = name.split(".")[0] if "." in name else name
    by_prefix[prefix]["T" if val else "F"] += 1
print("\nBy top-level path prefix:")
for p in sorted(by_prefix):
    c = by_prefix[p]
    print(f"  {p:<40} use_fa3=True: {c['T']:<3} use_fa3=False: {c['F']:<3}")

print(f"\nFirst 20 False entries (need to flip to True):")
falses = [(name, cls) for name, cls, val in by_path if not val][:20]
for name, cls in falses:
    print(f"  {cls:<35} {name}")
