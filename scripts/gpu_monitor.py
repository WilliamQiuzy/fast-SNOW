"""Real-time GPU monitor using pynvml.

Samples GPU at high frequency (default 50 Hz) while the inference runs.
Logs:
  - utilization (% SM busy)
  - memory used (GB)
  - power draw (W)
  - SM clock + memory clock
  - GPU temp

Outputs a CSV time-series + a quick text summary (utilization histogram,
idle-window detection).

Usage:
  from scripts.gpu_monitor import GPUMonitor
  mon = GPUMonitor(interval_hz=50, csv_path="run.csv")
  mon.start()
  ... run inference ...
  stats = mon.stop()
  print(mon.summary())
"""
from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Sample:
    t: float       # seconds since monitor start
    util: float    # % SM busy
    mem_gb: float  # used HBM
    power_w: float
    sm_clock_mhz: int
    mem_clock_mhz: int
    temp_c: int


class GPUMonitor:
    def __init__(
        self,
        gpu_index: int = 0,
        interval_hz: int = 50,
        csv_path: Optional[str] = None,
    ):
        self.gpu_index = gpu_index
        self.interval_s = 1.0 / interval_hz
        self.csv_path = csv_path
        self.samples: List[Sample] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0: float = 0.0

    def _open_nvml(self):
        import pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        self._pynvml = pynvml

    def _loop(self):
        nv = self._pynvml
        h = self._handle
        next_t = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now < next_t:
                # tight spin so we hit 50Hz without sleep-jitter
                time.sleep(max(0.0, next_t - now - 0.0005))
                continue
            util = nv.nvmlDeviceGetUtilizationRates(h).gpu
            mem = nv.nvmlDeviceGetMemoryInfo(h)
            try:
                power = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                power = 0.0
            try:
                sm_mhz = nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)
            except Exception:
                sm_mhz = 0
            try:
                mem_mhz = nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM)
            except Exception:
                mem_mhz = 0
            try:
                temp = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = 0
            self.samples.append(Sample(
                t=now - self._t0,
                util=float(util),
                mem_gb=mem.used / (1024 ** 3),
                power_w=float(power),
                sm_clock_mhz=int(sm_mhz),
                mem_clock_mhz=int(mem_mhz),
                temp_c=int(temp),
            ))
            next_t += self.interval_s

    def start(self):
        self._open_nvml()
        self.samples = []
        self._t0 = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def mark(self, label: str):
        """Add a labelled time-marker (no sample yet — caller can use it later)."""
        t = time.time() - self._t0
        # Append a special sample with negative util so we can mark events.
        # Stored separately for clarity.
        self._marks = getattr(self, "_marks", [])
        self._marks.append((t, label))

    def stop(self) -> List[Sample]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.csv_path:
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_s", "util_pct", "mem_gb", "power_w",
                           "sm_clock_mhz", "mem_clock_mhz", "temp_c"])
                for s in self.samples:
                    w.writerow([f"{s.t:.4f}", s.util, f"{s.mem_gb:.2f}", s.power_w,
                              s.sm_clock_mhz, s.mem_clock_mhz, s.temp_c])
            marks = getattr(self, "_marks", [])
            if marks:
                with open(Path(self.csv_path).with_suffix(".marks.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["t_s", "label"])
                    for t, label in marks:
                        w.writerow([f"{t:.4f}", label])
        return self.samples

    def summary(self) -> str:
        if not self.samples:
            return "(no samples)"
        utils = [s.util for s in self.samples]
        powers = [s.power_w for s in self.samples]
        mems = [s.mem_gb for s in self.samples]
        # Utilisation histogram (10% bins)
        bins = [0] * 11
        for u in utils:
            bins[min(10, int(u // 10))] += 1
        total = len(utils)

        # Idle gaps: contiguous spans with util < 5%
        gaps = []
        in_gap = False
        gap_start = 0.0
        for s in self.samples:
            if s.util < 5:
                if not in_gap:
                    in_gap = True
                    gap_start = s.t
            else:
                if in_gap:
                    gaps.append((gap_start, s.t, s.t - gap_start))
                    in_gap = False
        if in_gap:
            gaps.append((gap_start, self.samples[-1].t, self.samples[-1].t - gap_start))
        # Keep idle gaps >= 100ms (the actionable ones)
        gaps = [g for g in gaps if g[2] >= 0.1]
        gaps.sort(key=lambda g: -g[2])

        lines = [
            f"=== GPU monitor summary ({total} samples over {self.samples[-1].t:.2f}s) ===",
            f"Utilization: mean={sum(utils)/len(utils):.1f}%  max={max(utils):.0f}%  min={min(utils):.0f}%",
            f"Power:       mean={sum(powers)/len(powers):.0f}W  max={max(powers):.0f}W",
            f"HBM:         mean={sum(mems)/len(mems):.1f}GB  max={max(mems):.1f}GB",
            "",
            "Utilization histogram (10% bins, % of samples):",
        ]
        for i, b in enumerate(bins):
            pct = 100 * b / total
            bar = "#" * int(pct / 2)
            lo = i * 10
            hi = (i + 1) * 10 if i < 10 else 100
            lines.append(f"  {lo:>3}-{hi:<3}% : {bar:<50} {pct:>5.1f}%")
        lines.append("")
        lines.append("Top GPU-idle windows (util < 5% for >= 100ms):")
        if not gaps:
            lines.append("  (none — GPU was busy throughout)")
        else:
            for t0, t1, d in gaps[:10]:
                lines.append(f"  t={t0:>6.2f}s..{t1:<6.2f}s  ({d:.2f}s idle)")
        return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test: monitor for 3 seconds
    mon = GPUMonitor(interval_hz=50, csv_path="/tmp/gpu_smoke.csv")
    mon.start()
    time.sleep(3.0)
    mon.stop()
    print(mon.summary())
