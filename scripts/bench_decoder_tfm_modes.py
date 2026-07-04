"""Inner-transformer compile sweep — try every mode/dynamic combo to find
one that preserves quality."""
from __future__ import annotations

import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")

import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rose/vision/sam3"))

import torch
from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


def quality(sg, mf):
    n = sorted([len(t["F_k"]) for t in sg["tracks"]], reverse=True) if sg else []
    half = max(1, mf // 2)
    return (len(n), sum(1 for x in n if x >= half), max(n, default=0))


def bench(pool, videos, label, runs=2):
    print(f"\n=== {label} ===")
    try:
        t0 = time.time()
        pool.run_inference(InferenceRequest(video_path=videos[0], question=None))
        print(f"  warmup: {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"  warmup err: {type(e).__name__}: {e}")
        return None, [(0,0,0)] * 2
    avgs, qs = [], []
    for k in range(runs):
        ts, q_all = [], []
        for v in videos:
            t0 = time.time()
            resp = pool.run_inference(InferenceRequest(video_path=v, question=None))
            ts.append(time.time() - t0)
            sg = resp.four_dsg_dict if resp.status == "ok" else None
            q_all.append(quality(sg, pool.config.sampling.max_frames or 32) if sg else (0,0,0))
        avg = sum(ts) / 2
        avgs.append(avg); qs.append(q_all)
        print(f"  run {k+1}: E1={ts[0]:.2f}s {q_all[0]}  E2={ts[1]:.2f}s {q_all[1]}  avg={avg:.2f}s")
    return sum(avgs)/len(avgs), qs[-1]


def main():
    cfg = ROSEConfig()
    cfg.da3.model_path = "rose/models/da3-small"
    cfg.sam3.use_fa3 = True
    cfg.sam3.use_multiplex = True
    cfg.sam3.enable_compile = False
    cfg.sam3.compile_memory_encoder = False

    print("Loading...")
    pool = WarmModelPool(cfg); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"
    videos = [
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy1.mp4",
        "/workspace/fast-SNOW/sample_videos_and_analysis/VLM4D-Easy2.mp4",
    ]

    tfm = pool._sam3._predictor.model.tracker.model.sam_mask_decoder.transformer
    tfm_orig = tfm.forward

    bench(pool, videos, "baseline")

    # mode="default" (no CUDA graphs/autotune) — closest to eager numerics
    for dyn in [True, False]:
        for fg in [True, False]:
            label = f"mode=default dyn={dyn} fullgraph={fg}"
            try:
                tfm.forward = torch.compile(tfm_orig, mode="default", dynamic=dyn, fullgraph=fg)
                bench(pool, videos, label)
            except Exception as e:
                print(f"\n=== {label} ===\n  install/run threw: {type(e).__name__}: {e}")
            finally:
                tfm.forward = tfm_orig

    # Try max-autotune (worst case for numerics) — for sanity check
    try:
        tfm.forward = torch.compile(tfm_orig, mode="max-autotune", dynamic=False)
        bench(pool, videos, "mode=max-autotune dyn=False")
    except Exception as e:
        print(f"\nmax-autotune threw: {type(e).__name__}: {e}")
    finally:
        tfm.forward = tfm_orig

    # Compile each layer of TwoWayTransformer individually — see if there's
    # a specific layer that breaks
    print("\n# Layer-by-layer compile (TwoWayAttentionBlock)")
    for li in range(len(tfm.layers)):
        layer = tfm.layers[li]
        lf_orig = layer.forward
        try:
            layer.forward = torch.compile(lf_orig, mode="default", dynamic=True)
            bench(pool, videos, f"tfm.layers[{li}].compile only", runs=2)
        except Exception as e:
            print(f"  layer {li} threw: {e}")
        finally:
            layer.forward = lf_orig

    # Also try compiling just the final_attn_token_to_image
    final_attn = getattr(tfm, "final_attn_token_to_image", None)
    if final_attn is not None:
        fa_orig = final_attn.forward
        try:
            final_attn.forward = torch.compile(fa_orig, mode="default", dynamic=True)
            bench(pool, videos, "final_attn_token_to_image.compile", runs=2)
        except Exception as e:
            print(f"final_attn threw: {e}")
        finally:
            final_attn.forward = fa_orig


if __name__ == "__main__":
    main()
