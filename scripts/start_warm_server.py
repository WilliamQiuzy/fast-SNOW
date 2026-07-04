#!/usr/bin/env python3
"""Start the ROSE warm model server.

Loads DA3, FastSAM, and SAM3 into GPU memory once. Optionally triggers
``torch.compile`` warmup so all subsequent inference runs benefit from
compiled kernels without paying the ~30 s first-run cost.

Usage::

    # Basic (no torch.compile)
    python scripts/start_warm_server.py

    # With torch.compile warmup (recommended for multi-run workflows)
    python scripts/start_warm_server.py --compile

    # Custom port and config
    python scripts/start_warm_server.py --compile --port 8080 --config configs/my.yaml

Then send requests from another terminal or notebook::

    from rose.engine.server.warm_client import ROSEClient
    client = ROSEClient(port=5050)
    client.wait_ready()
    result = client.build_4dsg("benchmark/VLM4D-video/videos_synthetic/synth_001.mp4")
    print(result["status"], result["inference_time_s"])
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Start the ROSE warm model server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--port", type=int, default=5050,
        help="TCP port to listen on (default: 5050)",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Enable torch.compile + warmup (~30 s startup, 20-40%% faster inference)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: use ROSEConfig defaults)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from rose.engine.config.rose_config import (
        ROSEConfig,
        load_rose_config,
    )
    from rose.engine.server.warm_server import create_app

    if args.config:
        config = load_rose_config(args.config)
    else:
        config = ROSEConfig()

    if args.compile:
        config.sam3.enable_compile = True

    app = create_app(config)

    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,  # Single worker — models are not fork-safe
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
