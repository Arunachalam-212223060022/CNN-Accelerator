# Software Design Report
## FPGA CNN Accelerator — PYNQ-Z2 Software Architecture

---

> **Document Type:** Software Architecture & Design Report  
> **Platform:** PYNQ-Z2 (Zynq-7000 XC7Z020)  
> **Language:** Python 3 (PYNQ Framework)  
> **Design Model:** ARM PS as Controller — FPGA PL as Compute Engine  

---

## Table of Contents

1. [Architecture Philosophy](#1-architecture-philosophy)
2. [Memory Design — The Critical Foundation](#2-memory-design--the-critical-foundation)
3. [ARM–FPGA Communication — What Actually Happens](#3-armfpga-communication--what-actually-happens)
4. [Inference Pipeline — Timing Breakdown](#4-inference-pipeline--timing-breakdown)
5. [Weight Quantization in Practice](#5-weight-quantization-in-practice)
6. [CPU Baseline — Why It Matters](#6-cpu-baseline--why-it-matters)
7. [Design Trade-offs — The Real Implications](#7-design-tradeoffs--the-real-implications)
8. [Limitations — Root Cause Analysis](#8-limitations--root-cause-analysis)
9. [Future Architecture — What Changes and Why](#9-future-architecture--what-changes-and-why)
10. [Summary](#10-summary)

---

## 1. Architecture Philosophy

The software layer is not just a wrapper around hardware — it is the **system orchestrator**. Every byte the FPGA processes was placed there by the ARM. Every result the application sees was fetched by the ARM. Understanding this makes the design decisions below make sense.

### 1.1 Why Python on an FPGA System?

Python is not the fastest language, but on this platform it is the **right choice** for reasons that are easy to underestimate:

| Consideration | Impact |
|---|---|
| PYNQ's Python API directly maps to AXI-Lite register writes | No C driver needed — 1 line of Python = 1 hardware register write |
| NumPy operations on allocated buffers are zero-copy | `buf[:] = data` writes directly to physical DDR — no intermediate copy |
| OpenCV C++ backend under Python | Preprocessing is native speed despite Python syntax |
| Rapid iteration on weight sets and parameters | Changing a model requires editing a dict, not recompiling a driver |

The real cost of Python — interpreter overhead — only applies to **control logic**, not to data movement or inference, both of which are handled by C/C++ backends and hardware respectively.

### 1.2 Responsibility Boundary

The single most important architectural decision is where the PS–PL boundary sits:

```
ARM (PS) owns:                      FPGA (PL) owns:
─────────────────────               ─────────────────────────
  Image loading & resize              Conv1 → ReLU → MaxPool
  Weight file I/O                     Conv2 → ReLU → MaxPool
  DDR buffer allocation               Conv3 → ReLU → MaxPool
  Physical address management         Fully Connected Layer
  AXI register writes                 ArgMax + confidence
  Timing measurement                  Result write to DDR
  Result post-processing
  Visualization
```

Everything on the left is flexible and easily changed. Everything on the right is fixed until re-synthesis. This boundary is **permanent at runtime** — there is no way to move computation between PS and PL without regenerating the bitstream.

---

## 2. Memory Design — The Critical Foundation

The memory system is where most FPGA software bugs occur. Understanding it precisely prevents hours of debugging.

### 2.1 Why `pynq.allocate` Is Not Optional

Standard Python `numpy.array` allocations are **virtually addressed** — the operating system can move them anywhere in physical RAM, and the physical address is not guaranteed to stay fixed. The FPGA's AXI master needs a **stable physical address** to perform DMA reads.

`pynq.allocate` solves this by:

1. Calling the Linux CMA (Contiguous Memory Allocator) kernel driver
2. Reserving a physically contiguous region that will not be paged or moved
3. Exposing `.physical_address` — the actual hardware bus address the FPGA uses

```
Without pynq.allocate:
  Python array at virtual 0xB4A2F000  →  Physical could be 0x1F000, 0x4A000, 0x8C000 (anywhere)
  FPGA reads 0xB4A2F000 → AXI bus fault (virtual address meaningless to hardware)

With pynq.allocate:
  Buffer at virtual 0xB4A2F000  →  Physical fixed at 0x1F200000 (guaranteed stable)
  FPGA reads 0x1F200000 → Correct data every time
```

### 2.2 Cache Coherency — The Hidden Problem

The ARM Cortex-A9 has an L1/L2 cache. When Python writes data to a buffer, it may sit in cache and **never reach DDR** before the FPGA tries to read it.

This is why `.flush()` and `.invalidate()` are not optional cleanup — they are **correctness requirements**:

| Operation | When | Why |
|---|---|---|
| `buf.flush()` | After writing to buffer from ARM | Forces CPU cache → DDR write-back before FPGA reads |
| `buf.invalidate()` | Before reading result from ARM | Discards stale CPU cache so ARM reads fresh DDR data written by FPGA |

Missing either call produces **silent data corruption** — the code runs, values come back, but they are stale or wrong. This is one of the hardest bugs to diagnose in PS–PL systems.

### 2.3 Buffer Sizing — Why These Numbers

| Buffer | Shape | dtype | Size | Reasoning |
|---|---|---|---|---|
| `img_buf` | (4096,) | uint8 | 4 KB | 64×64 = 4096 bytes exactly |
| `res_buf` | (16,) | int32 | 64 B | Class ID, scores, confidence — headroom for future fields |
| `conv_w_buf` | (200000,) | int8 | ~195 KB | Largest possible weight set with room for future expansion |

The weight buffers are deliberately **oversized**. The actual Conv1 weights are much smaller than 200,000 bytes — but allocating large fixed buffers avoids reallocation and the risk of running out of space mid-weight-load without warning.

---

## 3. ARM–FPGA Communication — What Actually Happens

### 3.1 The Two-Phase Communication Model

Every inference involves exactly two types of AXI transaction, and they serve completely different purposes:

```
Phase 1 — Setup (AXI-Lite, ARM → FPGA):
  ARM tells FPGA where to find everything in DDR.
  Happens BEFORE inference starts.
  Cost: microseconds.

Phase 2 — Execution (AXI Master, FPGA → DDR → FPGA):
  FPGA reads image and weights, runs CNN, writes result.
  Happens DURING inference.
  Cost: majority of inference latency.
```

### 3.2 Why Address Registers Must Be Written Before `ap_ctrl = 0x01`

The HLS IP reads its address registers **once at the start of execution**. If `ap_ctrl` is set before the addresses are written, the accelerator begins fetching from whatever garbage values were in those registers — which typically points to address `0x00000000` (the Zynq exception vector table). The result is either wrong data or an AXI bus error.

Correct order, enforced in software:

```python
# 1. Write all addresses first
ip.write(0x10, addr & 0xFFFFFFFF)        # image low
ip.write(0x14, (addr >> 32) & 0xFF)      # image high
ip.write(0x1C, conv1_w.physical_address) # conv1 weights
# ... all other addresses ...

# 2. Only then start
ip.write(0x00, 0x01)                     # ap_ctrl START
```

### 3.3 The Polling Loop — Cost and Alternative

```python
while (ip.read(0x00) & 0x02) == 0:
    time.sleep(0.0005)
```

This loop burns ARM CPU cycles reading an AXI-Lite register every 0.5ms. For a 150ms inference, that is approximately **300 register reads** that serve no compute purpose.

The alternative — interrupt-driven completion — would free the ARM to do other work (preprocessing the next frame, updating UI, logging) during the ~150ms the FPGA is computing. At 6.7 FPS, 150ms of idle ARM time per frame is a significant opportunity cost.

This is the single highest-impact software improvement available without hardware changes.

---

## 4. Inference Pipeline — Timing Breakdown

### 4.1 Where Time Is Actually Spent

Not all stages of the pipeline take equal time. Understanding the breakdown reveals where optimization efforts pay off:

| Stage | Runs On | Estimated Time | Optimizable? |
|---|---|---|---|
| Image load from disk | ARM | ~5–20ms | ✅ Cache in RAM |
| BGR → Grayscale | ARM (OpenCV C++) | < 1ms | Negligible |
| Resize to 64×64 | ARM (OpenCV C++) | < 1ms | Negligible |
| Flatten + dtype cast | ARM (NumPy) | < 0.5ms | Negligible |
| `img_buf.flush()` | ARM → DDR | ~0.5–2ms | ✅ AXI-Stream DMA |
| AXI register writes | ARM → FPGA | < 0.1ms | Negligible |
| FPGA CNN execution | FPGA | ~5–150ms | ✅ HLS optimization |
| Poll loop (waiting) | ARM (idle) | ~5–150ms | ✅ Use interrupt |
| `res_buf.invalidate()` | DDR → ARM cache | < 0.1ms | Negligible |
| Post-processing | ARM | < 1ms | Negligible |

**Key insight:** The ARM preprocessing pipeline is fast enough that it is **never the bottleneck**. The bottleneck is always either FPGA execution time or the memory transfer of weights from DDR.

### 4.2 Weight Loading — The One-Time Cost

Weights are loaded from `.npy` files on the SD card into DDR buffers. This is a **one-time cost per session**, not per inference:

```
SD card read of .npy files:   ~50–200ms (SD I/O bound)
NumPy dtype cast + flatten:   ~1–5ms
DDR buffer write + flush:     ~5–20ms
─────────────────────────────────────
Total weight loading:         ~60–225ms (done ONCE)
Per-inference weight cost:    0ms (already in DDR)
```

This is why the software separates `load_weight_set()` from `fpga_classify()` — weights should never be reloaded between inferences on the same image set.

---

## 5. Weight Quantization in Practice

### 5.1 The INT8 Pipeline from File to Silicon

The journey from a trained float32 model to INT8 values executing in hardware involves several steps, each with potential for error:

```
Training (float32)
  conv1_weights.npy  →  dtype: float32, shape: (8, 1, 3, 3)
        │
  .astype(np.int8)   →  Values clipped to [-128, 127]
        │              ⚠️  Values outside this range silently wrap
        │
  .flatten()         →  Shape: (72,) — 1D for AXI transfer
        │
  buf[:72] = data    →  Written to physically contiguous DDR
        │
  buf.flush()        →  Forced to DDR before FPGA reads
        │
  FPGA reads via AXI Master  →  INT8 values arrive at DSP48E1 inputs
        │
  DSP48E1: A×B + C   →  INT8 × INT8 → INT32 accumulator
```

### 5.2 Quantization Failure Modes

| Failure | Cause | Symptom | Fix |
|---|---|---|---|
| Silent value wrapping | Float weight > 127 or < -128 before cast | Wrong predictions, not errors | Normalize weights before quantization |
| Scale mismatch | Different layers quantized at different scales | Activations blow up in later layers | Calibrated per-layer scale factors |
| Dead filters | All weights for a filter round to 0 | Filter produces no activation | Use symmetric quantization with nonzero range |
| Accumulator overflow | Many large INT8 products summed | Garbage output | Use INT32 accumulator (already done in HLS) |

---

## 6. CPU Baseline — Why It Matters

### 6.1 The Baseline Is a Diagnostic Tool, Not Just a Benchmark

The CPU NumPy inference serves a purpose beyond measuring speed: it is the **ground truth reference** for correctness checking. When FPGA and CPU predictions differ (DIFFER case), the baseline tells you which direction the quantization error pushed the decision.

```
FPGA → Dog (55% conf)   ← INT8 quantized weights
CPU  → Cat (100% conf)  ← Float32 weights

Both using same input image.
DIFFER case reveals: quantization changed the prediction.
Question: which is correct?
Answer: ground truth (the image) — in this case, Dog.
```

The baseline also validates the **architecture itself** independently of the FPGA. If the CPU baseline gives wrong predictions even in float32, the problem is in the model or the training data, not in quantization or hardware synthesis.

### 6.2 CPU Performance Is Not the Relevant Comparison

The relevant comparison is not CPU best-case vs FPGA — it is **CPU under production load vs FPGA**:

| Scenario | CPU | FPGA |
|---|---|---|
| Single image, no load | 14.3ms | 150ms |
| Single image, OS under load | 50–200ms | 150ms |
| Sustained 10 fps stream | Degrades, drops frames | Deterministic 150ms always |
| With concurrent processes | Highly variable | Unaffected |

The FPGA's value is **predictability**, not peak speed.

---

## 7. Design Trade-offs — The Real Implications

### 7.1 No DMA — What This Actually Costs

The system uses AXI Master memory-mapped transactions instead of AXI-DMA. This simplifies the software significantly but has a concrete performance cost:

| Aspect | AXI Master (Current) | AXI-Stream DMA (Future) |
|---|---|---|
| Setup complexity | Write 2 address registers | Configure DMA engine, set up scatter-gather |
| Transfer initiation | Happens automatically when IP starts | Explicit DMA transaction per transfer |
| Transfer speed | ~400–800 MB/s (AXI4 burst) | ~1–2 GB/s (streaming) |
| CPU involvement during transfer | None (hardware handles it) | None (hardware handles it) |
| Latency overhead | ~5–20ms for weight fetch | ~1–3ms for streaming |

For a 64×64 image (4KB), the difference is marginal. For a larger model with MB-scale weights, DMA streaming becomes significant.

### 7.2 Polling vs Interrupt — The Quantified Cost

At 150ms inference time and 0.5ms polling interval:

```
Polling cost:  300 AXI reads × ~10 CPU cycles each = 3000 cycles wasted
               = 3000 / 667MHz = ~4.5 microseconds of compute lost
               = negligible compute loss

BUT:
  ARM is blocked for 150ms doing nothing useful
  At 6.7 FPS, ARM is idle 100% of the time between inferences
  With interrupt: ARM could preprocess next frame during 150ms FPGA runtime
  Potential throughput gain: up to 2× with pipelined prefetch
```

### 7.3 Full Trade-off Matrix

| Decision | Why Chosen | Real Cost | When to Reconsider |
|---|---|---|---|
| No AXI-DMA | Simpler software, fewer failure modes | ~5–20ms extra latency per inference | When model weights exceed ~1MB |
| Polling completion | Zero interrupt setup complexity | ARM blocked during inference | When preprocessing next frame matters |
| INT8 quantization | 4× memory, faster MAC | Can flip class on borderline images | When DIFFER rate exceeds acceptable threshold |
| Python control layer | Fast development, easy debugging | ~1–5ms interpreter overhead per call | When sub-10ms total latency is required |
| Fixed buffer sizes | No runtime reallocation risk | Wastes ~1MB RAM in oversized buffers | When RAM is critically constrained |

---

## 8. Limitations — Root Cause Analysis

Each limitation has a specific architectural root cause, not just a description:

### 8.1 Memory-Mapped Transfer Slower Than DMA

**Root cause:** AXI master bursts are initiated by the HLS IP itself, which must complete one burst request before issuing the next. There is no scatter-gather engine to pipeline multiple transfers. The IP fetches sequentially: image → conv1_w → conv1_b → conv2_w → ... — each fetch waits for the previous to complete.

### 8.2 Polling Wastes CPU Cycles

**Root cause:** The HLS IP exports an interrupt signal (`ap_done`) that is connected to `IRQ_F2P` in the block design, but the software never registers an ISR for it. The interrupt infrastructure exists in hardware — it is simply unused in software.

### 8.3 Fixed Model Architecture

**Root cause:** Layer dimensions (filter counts, kernel sizes, FC width) are HLS template parameters or `#define` constants baked into the synthesized netlist. Changing them requires C++ edit → HLS synthesis (~5 min) → Vivado implementation (~20–40 min) → new bitstream. There is no runtime configurability possible with the current AXI-Lite register map.

### 8.4 No Interrupt-Driven Design

**Root cause:** Registering a Linux userspace interrupt handler for PYNQ requires using the `pynq.interrupt` module and `asyncio`-based awaitable patterns. This is a software-only change — the hardware already supports it. The current polling approach was chosen for simplicity during development and never upgraded.

---

## 9. Future Architecture — What Changes and Why

### 9.1 Priority Improvements with Architectural Impact

| Improvement | Changes Required | Latency Benefit | Complexity Added |
|---|---|---|---|
| **Interrupt-driven completion** | Python only — add `pynq.Interrupt` + `asyncio` | Enables frame pipelining (~2× throughput) | Low |
| **AXI-Stream DMA** | New HLS interface + Vivado block design update | ~10–15ms reduction in weight fetch | Medium |
| **Batch inference** | HLS C++ changes + larger DDR buffers | Linear throughput scaling with batch size | Medium |
| **C-based driver** | Replace Python with C using `/dev/mem` or UIO | ~1–5ms reduction in control overhead | High |
| **Quantization-aware training** | Retraining only — no hardware change | Reduces DIFFER rate, improves confidence calibration | Medium |
| **Camera streaming (HDMI/MIPI)** | New PS/PL interface, framebuffer management | Eliminates SD card I/O latency entirely | High |

### 9.2 The Interrupt Pipeline — What It Enables

The most impactful software change requires zero hardware modification:

```
Current (polling):
  Frame N:   [Preprocess]─[Transfer]─[FPGA runs 150ms, ARM polls]─[Read result]
  Frame N+1: ────────────────────────────────────────────────────►[Preprocess]...

With interrupt + async pipeline:
  Frame N:   [Preprocess]─[Transfer]─[FPGA runs 150ms]──────────►[Read result]
  Frame N+1: ─────────────────────────[Preprocess]─[Transfer]─►[FPGA queued]
                                       ▲ starts here, during N's FPGA time
```

This transforms the system from **sequential** to **pipelined** — matching real camera-based embedded AI architectures.

---

## 10. Summary

The software architecture succeeds at its primary goal: **making a complex FPGA accelerator accessible and usable through a clean Python API**. The PYNQ framework handles the low-level complexity of overlay loading and physical memory, while the application layer focuses purely on inference logic.

### What the design gets right

The strict separation of PS (control) and PL (compute) means the software is **independently testable** — the CPU baseline runs without any FPGA hardware, making model debugging possible without a board. Weight loading, preprocessing, and result parsing are all isolated functions that can be validated in isolation before running on silicon.

### Where the design leaves performance on the table

The two largest unused performance opportunities are both **software-only fixes**:

1. The interrupt signal is wired in hardware but never used — replacing polling with `pynq.Interrupt` would unlock frame pipelining
2. Weights are loaded from SD card on every session start — caching them in a RAM-backed tmpfs would cut startup time by 80%

Neither requires touching the HLS code or regenerating the bitstream.

### The co-design insight

The software and hardware are **not independent** — every software decision has a hardware implication and vice versa. The choice to use polling instead of interrupts means the ARM is idle during FPGA inference. The choice to use AXI master instead of DMA means the HLS IP controls its own memory access schedule. Understanding these connections is what distinguishes a working system from an optimized one.

---

*Software Design Report — FPGA CNN Accelerator — PYNQ-Z2 / Zynq-7020 — Vitis HLS 2023.1*
