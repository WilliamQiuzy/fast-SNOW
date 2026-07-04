"""Probe the SAM3 multiplex model hierarchy to find the right path to the
tracker / encoder for monkey-patching."""
from __future__ import annotations
import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False

    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda()
    sam3 = pool._sam3
    print("\n=== pool._sam3 attributes ===")
    print(f"  type: {type(sam3).__name__}")
    p = sam3._predictor
    print(f"  sam3._predictor: {type(p).__name__}")
    m = p.model
    print(f"  sam3._predictor.model: {type(m).__name__}")
    print(f"  has .tracker? {hasattr(m, 'tracker')}")
    if hasattr(m, "tracker"):
        tk = m.tracker
        print(f"  sam3._predictor.model.tracker: {type(tk).__name__}")
        print(f"    has _prepare_memory_conditioned_features? {hasattr(tk, '_prepare_memory_conditioned_features')}")
        print(f"    has transformer? {hasattr(tk, 'transformer')}")
        if hasattr(tk, "transformer"):
            tfm = tk.transformer
            print(f"    tk.transformer: {type(tfm).__name__}")
            if hasattr(tfm, "encoder"):
                enc = tfm.encoder
                print(f"      tk.transformer.encoder: {type(enc).__name__}")
            else:
                print(f"      transformer has no .encoder. dir: {[x for x in dir(tfm) if not x.startswith('_')][:30]}")
        else:
            print(f"    tracker dir (first 40): {[x for x in dir(tk) if not x.startswith('_')][:40]}")
            # Maybe the encoder is at another path
            for name in ("memory_attention", "mem_attention", "memory_encoder"):
                if hasattr(tk, name):
                    print(f"    HAS tk.{name}: {type(getattr(tk, name)).__name__}")
    # If no .tracker, check m directly
    print(f"  has m._prepare_memory_conditioned_features? {hasattr(m, '_prepare_memory_conditioned_features')}")
    # MRO of m
    print(f"  m.__class__.__mro__: {[c.__name__ for c in type(m).__mro__]}")
    # Find the class in MRO that defines _prepare_memory_conditioned_features
    for cls in type(m).__mro__:
        if "_prepare_memory_conditioned_features" in vars(cls):
            print(f"    defined on class: {cls.__name__}")
            break
    else:
        print(f"    NOT found on m's MRO")

    if hasattr(m, "tracker"):
        for cls in type(m.tracker).__mro__:
            if "_prepare_memory_conditioned_features" in vars(cls):
                print(f"  tracker MRO: defined on {cls.__name__}")
                break
        print(f"  tracker.__class__.__mro__: {[c.__name__ for c in type(m.tracker).__mro__]}")


if __name__ == "__main__":
    main()
