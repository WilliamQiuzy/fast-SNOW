# CPU Worker 线程异常延迟检测

## 状态：Open

## 问题描述

Phase 2 的流水线架构使用 GPU 主线程和 CPU worker 线程并行工作：GPU 线程运行 FastSAM + SAM3 two-pass 后逐帧构建 `FastFrameInput`，CPU worker 线程消费队列中的 `FastFrameInput` 执行 backproject → STEP → pipeline update。

如果 CPU worker 线程在处理某一帧时抛出异常，GPU 主线程**不会立即感知**。GPU 线程会继续将所有剩余帧的数据放入队列，直到所有帧处理完毕、发送 sentinel 并 `worker.join()` 后，才检查 `cpu_error[0]`。

这意味着：**CPU worker 在 frame 5 崩溃，但 GPU 线程会继续跑完 frame 6 到 frame N 的 YOLO + SAM3 推理**，浪费大量 GPU 时间和电力。

## 影响

### 直接后果

1. **GPU 时间浪费**：CPU worker 已经崩溃，但 GPU 仍在做无用的推理。对于长视频（50+ 帧），可能浪费数分钟的 GPU 时间。
2. **用户体验差**：用户等到整个视频 GPU 推理完成后才看到错误信息，而实际上错误在很早的帧就已经发生。
3. **错误信息延迟**：异常堆栈指向 CPU worker 中的处理逻辑，但用户经历了长时间等待后才看到这个错误，可能误以为是最后阶段的问题。

### 严重程度

**Low-Medium**——不会导致数据损坏或错误结果（最终会正确抛出异常），但影响效率和调试体验。

## 根因分析

### 流水线架构

```python
# fast_snow_e2e.py:186-218

cpu_queue: queue.Queue[Optional[FastFrameInput]] = queue.Queue()
cpu_error: List[Optional[BaseException]] = [None]

def _cpu_worker() -> None:
    try:
        while True:
            item = cpu_queue.get()
            if item is _SENTINEL:
                break
            pipeline.process_frame(item)
    except Exception as exc:
        cpu_error[0] = exc          # ← 记录异常但不通知主线程

worker = threading.Thread(target=_cpu_worker, daemon=True)
worker.start()

try:
    for sam3_idx, image in enumerate(frames):
        fi, trace = self._build_frame_input(...)    # ← GPU 推理
        step01_trace.append(trace)
        cpu_queue.put(fi)           # ← 持续入队，不检查 cpu_error
finally:
    cpu_queue.put(_SENTINEL)
    worker.join()

if cpu_error[0] is not None:       # ← 最后才检查
    raise cpu_error[0]
```

### 问题时序

```
时间轴 →

GPU 主线程:  [frame 0] [frame 1] [frame 2] ... [frame N] [sentinel] [join] [check error] → raise!
              put(f0)   put(f1)   put(f2)       put(fN)

CPU worker:  [f0 OK]   [f1 OK]   [f2 CRASH! → 记录 error，线程退出]
                                       ↓
                                  f3, f4, ..., fN 永远不会被消费
                                  队列持续积压直到 sentinel
```

### 为什么队列不会阻塞主线程

`queue.Queue()` 默认无界（`maxsize=0`），所以 `cpu_queue.put()` 永远不会阻塞。即使 CPU worker 已死，GPU 线程仍能将所有帧数据无阻塞地放入队列。

## 可能的解决方向

### 方向 1：GPU 线程每帧检查 cpu_error

在每次 `cpu_queue.put()` 前检查 `cpu_error[0]`，如果 CPU worker 已崩溃则立即停止：

```python
try:
    for sam3_idx, image in enumerate(frames):
        # 快速检查 CPU worker 是否还活着
        if cpu_error[0] is not None:
            logger.error("CPU worker crashed, aborting GPU inference")
            break
        fi, trace = self._build_frame_input(...)
        step01_trace.append(trace)
        cpu_queue.put(fi)
finally:
    cpu_queue.put(_SENTINEL)
    worker.join()
```

**优点**：改动最小，一行检查。

**风险**：检查频率取决于 GPU 推理速度——如果每帧 GPU 推理很快（<100ms），检测延迟可以接受；如果很慢（>1s），可能在 CPU 崩溃后多跑 1 帧才停止。

### 方向 2：使用 Event 信号

用 `threading.Event` 让 CPU worker 在崩溃时主动通知 GPU 线程：

```python
abort_event = threading.Event()

def _cpu_worker():
    try:
        while not abort_event.is_set():
            item = cpu_queue.get(timeout=1.0)
            if item is _SENTINEL:
                break
            pipeline.process_frame(item)
    except Exception as exc:
        cpu_error[0] = exc
        abort_event.set()       # ← 立即通知

# GPU 线程
for ...:
    if abort_event.is_set():
        break
    ...
```

**优点**：GPU 线程可以在下一个循环迭代立即感知。

**风险**：稍微增加代码复杂度。

### 方向 3：有界队列

设置 `queue.Queue(maxsize=2)`，当 CPU worker 死后不再消费队列，`cpu_queue.put()` 在队列满时阻塞 GPU 线程。结合超时可以检测异常。

**优点**：同时控制内存使用（不会无限积压帧数据在队列中）。

**风险**：
- 正常情况下有界队列也可能导致 GPU 等待 CPU 消费，降低吞吐量。
- 需要精心选择 `maxsize` 和超时参数。

## 复现方式

在 `fast_snow_pipeline.py` 的 `process_frame()` 中注入一个异常（例如在第 3 帧时 `raise RuntimeError("test")`），然后运行长视频（50+ 帧）。观察 GPU 推理是否在第 3 帧后立即停止（当前不会）。

## 相关文件

- [fast_snow_e2e.py:186-218](fast_snow/engine/pipeline/fast_snow_e2e.py#L186-L218) — CPU worker 定义和错误检查逻辑
- [fast_snow_e2e.py:202-212](fast_snow/engine/pipeline/fast_snow_e2e.py#L202-L212) — GPU 主线程循环（不检查 cpu_error）
