# Real-Time Person/Object Detector — PYNQ-Z2 CNN Accelerator

> A 3-layer hardware-accelerated CNN deployed on Zynq-7020, detecting **Person vs Object** in 64×64 grayscale images. Built from scratch using Vitis HLS — no Vitis AI, no FINN.

---

## Table of Contents

- [Overview](#overview)
- [Hardware Platform](#hardware-platform)
- [CNN Architecture](#cnn-architecture)
- [System Architecture](#system-architecture)
- [HLS Accelerator](#hls-accelerator)
- [AXI Register Map](#axi-register-map)
- [Weight Layout](#weight-layout)
- [Repository Structure](#repository-structure)
- [Build & Deploy](#build--deploy)
- [Python Driver](#python-driver)
- [Output Format](#output-format)
- [C Simulation Testbench](#c-simulation-testbench)
- [Performance](#performance)
- [Resource Utilization](#resource-utilization)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Overview

This project implements a complete end-to-end CNN inference pipeline on the **PYNQ-Z2** FPGA board. The CNN is synthesized from C++ using **Vitis HLS**, integrated into a **Vivado block design**, and driven from Python using the **PYNQ framework**.

The ARM Cortex-A9 handles preprocessing and control. The FPGA fabric executes all convolution, pooling, and fully connected operations in hardware with parallelized MAC units.

---

## Hardware Platform

| Component      | Details                           |
|----------------|-----------------------------------|
| Board          | PYNQ-Z2                           |
| SoC            | Xilinx Zynq-7020 (XC7Z020CLG400)  |
| CPU            | Dual-core ARM Cortex-A9 @ 650 MHz |
| BRAM           | 140 blocks                        |
| Toolchain      | Vivado 2023.1 + Vitis HLS 2023.1  |
| Target Clock   | 10 ns (100 MHz)                   |
| Target Part    | `xc7z020clg400-1`                 |

---

## CNN Architecture

```
Input: 64×64 grayscale image
         │
    ┌────▼────┐
    │  Conv1  │  3×3, 8 filters   → 62×62×8
    │  ReLU   │
    │  Pool1  │  2×2 max          → 31×31×8
    ├────▼────┤
    │  Conv2  │  3×3, 16 filters  → 29×29×16
    │  ReLU   │
    │  Pool2  │  2×2 max          → 14×14×16
    ├────▼────┤
    │  Conv3  │  3×3, 32 filters  → 12×12×32
    │  ReLU   │
    │  Pool3  │  2×2 max          →  6×6×32
    ├────▼────┤
    │   FC    │  1152 → 64        → 64
    │  ReLU   │
    ├────▼────┤
    │  Out    │  Argmax           → 2 classes
    └─────────┘
         │
   Person / Object
```

| Layer  | Input Shape  | Output Shape | Parameters              |
|--------|--------------|--------------|-------------------------|
| Conv1  | 64×64×1      | 62×62×8      | 8 filters, 3×3          |
| Pool1  | 62×62×8      | 31×31×8      | 2×2 max pooling         |
| Conv2  | 31×31×8      | 29×29×16     | 16 filters, 3×3, 8 ch  |
| Pool2  | 29×29×16     | 14×14×16     | 2×2 max pooling         |
| Conv3  | 14×14×16     | 12×12×32     | 32 filters, 3×3, 16 ch |
| Pool3  | 12×12×32     | 6×6×32       | 2×2 max pooling         |
| FC     | 1152         | 64           | 64 neurons              |
| Output | 64           | 2            | Argmax (Person/Object)  |

**Precision:** INT8 weights and activations, INT16/INT32 accumulators
**Total weights:** 79,560 · **Total biases:** 120

---

## System Architecture

```
┌──────────────────────────────────────────────┐
│              ZYNQ PS (ARM Cortex-A9)         │
│  • Image loading & CLAHE preprocessing       │
│  • Weight loading from .npy files            │
│  • DDR address writing via AXI-Lite          │
│  • Start signal → poll Done                  │
│  • Read result → display classification      │
└───────────────────┬──────────────────────────┘
                    │ AXI-Lite (control)
                    │ AXI HP    (data)
┌───────────────────▼──────────────────────────┐
│            FPGA PL (CNN Accelerator)         │
│                                              │
│  DDR ──► [Image Buffer] ──► Conv1 ──► Pool1  │
│  DDR ──► [Weights]      ──► Conv2 ──► Pool2  │
│                         ──► Conv3 ──► Pool3  │
│                         ──► FC ──► Argmax    │
│                                    │         │
│                         DDR ◄── [Result]     │
└──────────────────────────────────────────────┘
```

### AXI Interface Summary

| Interface   | Bundle    | Purpose                             |
|-------------|-----------|-------------------------------------|
| `m_axi`     | `gmem0`   | Image read + Result write (DDR)     |
| `m_axi`     | `gmem1`   | All weights and bias read (DDR)     |
| `s_axilite` | `control` | Port addresses, start/done, return  |

---

## HLS Accelerator

The accelerator (`real_detector.cpp`) implements all CNN layers in a single HLS kernel.

### Key HLS Pragmas

```cpp
// Pipelined convolution inner loops — II=1 means one output per clock
#pragma HLS PIPELINE II=1

// BRAM-backed feature map buffers (saves LUT-RAM)
#pragma HLS BIND_STORAGE variable=conv1 type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=pool1 type=RAM_2P impl=BRAM

// Full unroll for small arrays (FC output, class scores)
#pragma HLS ARRAY_PARTITION variable=fc complete
#pragma HLS ARRAY_PARTITION variable=scores complete
```

### Convolution with ReLU (INT8 → INT16)

```cpp
ap_int<32> sum = 0;
for(int ky = 0; ky < 3; ky++)
    for(int kx = 0; kx < 3; kx++)
        sum += pixel * weight;   // INT8 × INT8 accumulated into INT32

sum += bias;
if(sum < 0) sum = 0;             // ReLU
output = (ap_int<16>)(sum >> 4); // Scale down + truncate to INT16
```

### Bounding Box via Activation Hotspot

Pool3 (6×6×32) is scanned to find the spatial coordinate with maximum summed activation. That coordinate maps back to the 64×64 image space:

```cpp
best_x = x * 10 + 5;  // 6×6 → 64×64 coordinate mapping
best_y = y * 10 + 5;
```

---

## AXI Register Map

| Offset | Register               | Description                    |
|--------|------------------------|--------------------------------|
| 0x00   | `CTRL`                 | `AP_START` (bit 0) / `AP_DONE` (bit 1) |
| 0x10   | `image` addr low       | Image DDR physical address     |
| 0x14   | `image` addr high      | Image DDR physical address     |
| 0x1C   | `conv1_w` addr         | Conv1 weights DDR address      |
| 0x24   | `conv1_b` addr         | Conv1 bias DDR address         |
| 0x2C   | `conv2_w` addr         | Conv2 weights DDR address      |
| 0x34   | `conv2_b` addr         | Conv2 bias DDR address         |
| 0x3C   | `conv3_w` addr         | Conv3 weights DDR address      |
| 0x44   | `conv3_b` addr         | Conv3 bias DDR address         |
| 0x4C   | `fc_w` addr            | FC weights DDR address         |
| 0x54   | `fc_b` addr            | FC bias DDR address            |
| 0x5C   | `result` addr          | Result buffer DDR address      |

---

## Weight Layout

Weights are stored in separate INT8 arrays and each passed as an individual AXI Master port. They are allocated independently via `pynq.allocate()`.

| Symbol      | Size (bytes) | Formula          |
|-------------|-------------|------------------|
| `CONV1_W`   | 72          | 8 × 1 × 9       |
| `CONV1_B`   | 8           | 8                |
| `CONV2_W`   | 1,152       | 16 × 8 × 9      |
| `CONV2_B`   | 16          | 16               |
| `CONV3_W`   | 4,608       | 32 × 16 × 9     |
| `CONV3_B`   | 32          | 32               |
| `FC_W`      | 73,728      | 64 × 6 × 6 × 32 |
| `FC_B`      | 64          | 64               |
| **Total W** | **79,560**  |                  |
| **Total B** | **120**     |                  |

---

## Repository Structure

```
CNN-Accelerator/
│
├── hls/
│   ├── real_detector.cpp        # HLS CNN accelerator (all 3 conv layers + FC)
│   ├── real_detector.h          # Interface declarations + weight size macros
│   ├── real_detector_test.cpp   # C simulation testbench (2 test cases)
│   └── run_hls.tcl              # HLS synthesis + C-sim + IP export script
│
├── vivado/
│   └── real_detect.xpr          # Vivado block design project
│
├── pynq/
│   ├── overlay.bit              # FPGA bitstream
│   ├── overlay.hwh              # Hardware handoff (PYNQ register map)
│   └── python_driver.py         # Inference driver
│
├── weights/
│   ├── conv1_w.npy / conv1_b.npy
│   ├── conv2_w.npy / conv2_b.npy
│   ├── conv3_w.npy / conv3_b.npy
│   └── fc_w.npy    / fc_b.npy
│
├── docs/
│   ├── system_architecture.md
│   ├── results.md
│   └── images/
│
└── README.md
```

---

## Build & Deploy

### Step 1 — HLS Synthesis

```bash
source /Xilinx/Vitis_HLS/2023.1/settings64.sh
cd hls/
vitis_hls run_hls.tcl
```

`run_hls.tcl` will automatically:
1. Run C simulation (`csim_design -clean`) — validates both test cases
2. Run synthesis (`csynth_design`) — generates RTL and timing/resource report
3. Export IP catalog — ready for Vivado import

Find the synthesis report at:
```
real_detector_hls/solution1/syn/report/real_detector_csynth.rpt
```

### Step 2 — Vivado Block Design

```bash
vivado vivado/real_detect.xpr
```

1. Import HLS IP from `real_detector_hls/solution1/impl/ip`
2. Add **Zynq PS** block → Run Block Automation
3. Connect `M_AXI_GP0` → AXI Interconnect → `s_axilite` (control)
4. Connect `S_AXI_HP0` → AXI Interconnect → `m_axi_gmem0`, `m_axi_gmem1`
5. Enable DDR and GP/HP ports in Zynq PS configuration
6. Run **Generate Bitstream**

### Step 3 — Deploy to PYNQ-Z2

Copy to the board via `scp` or USB:

```
overlay.bit
overlay.hwh
python_driver.py
weights/          (all .npy files)
```

### Step 4 — Run Inference

```bash
python3 python_driver.py
```

---

## Python Driver

```python
from pynq import Overlay, allocate
import numpy as np

ol = Overlay("overlay.bit")
det = ol.real_detector_0

# Load weights
conv1_w = np.load("weights/conv1_w.npy").astype(np.int8)
conv1_b = np.load("weights/conv1_b.npy").astype(np.int8)
# ... repeat for conv2, conv3, fc

# Allocate DDR buffers
img_buf = allocate(shape=(4096,), dtype=np.uint8)
c1w_buf = allocate(shape=(72,),   dtype=np.int8)
c1b_buf = allocate(shape=(8,),    dtype=np.int8)
# ... repeat for all weight buffers
res_buf = allocate(shape=(9,),    dtype=np.int32)

# Copy weights into DDR
c1w_buf[:] = conv1_w
c1b_buf[:] = conv1_b
# ...

# Write DDR addresses to accelerator registers
det.write(0x10, img_buf.physical_address & 0xFFFFFFFF)
det.write(0x14, img_buf.physical_address >> 32)
det.write(0x1C, c1w_buf.physical_address & 0xFFFFFFFF)
# ... all weight/result addresses

# Start and poll Done
import time
det.write(0x00, 1)
t0 = time.time()
while not (det.read(0x00) & 0x2):
    pass
print(f"Latency: {(time.time()-t0)*1000:.2f} ms")

# Read result
label = "Person" if res_buf[0] == 1 else "Object"
print(f"Class:      {label}")
print(f"BBox:       ({res_buf[1]}, {res_buf[2]}, {res_buf[3]}, {res_buf[4]})")
print(f"Confidence: {res_buf[5]}")
assert res_buf[8] == 0xC0FFEE00, "Magic mismatch — accelerator may not have completed"
```

---

## Output Format

The accelerator writes 9 integers to the result DDR buffer:

| Index    | Value           | Description                        |
|----------|-----------------|------------------------------------|
| `[0]`    | `0` or `1`      | Class: 0 = Object, 1 = Person      |
| `[1]`    | `0–63`          | BBox X center (image coordinates)  |
| `[2]`    | `0–63`          | BBox Y center (image coordinates)  |
| `[3]`    | int             | BBox width                         |
| `[4]`    | int             | BBox height                        |
| `[5]`    | int             | Confidence (max pool3 activation)  |
| `[6]`    | int             | Raw Object class score             |
| `[7]`    | int             | Raw Person class score             |
| `[8]`    | `0xC0FFEE00`    | Magic number (sanity check)        |

---

## C Simulation Testbench

`real_detector_test.cpp` validates two synthetic input patterns with random INT8 weights:

| Test | Input Pattern                             | Pass Criterion          |
|------|-------------------------------------------|-------------------------|
| 1    | Vertical blob (x∈[20,40], y∈[10,55])     | `result[8] == 0xC0FFEE00` |
| 2    | Square blob (x∈[15,50], y∈[20,45])       | `result[8] == 0xC0FFEE00` |

The magic number `0xC0FFEE00` written at `result[8]` confirms that the accelerator completed all layers and wrote output correctly. Run via:

```bash
vitis_hls run_hls.tcl   # csim_design runs automatically
```

---

## Performance

| Metric        | CPU Only | FPGA Accelerated | Speedup |
|---------------|----------|------------------|---------|
| Latency (ms)  | XX       | XX               | XX×     |
| Throughput    | XX FPS   | XX FPS           | —       |
| Power (W)     | XX       | XX               | —       |

> Measure FPGA latency with the timing snippet in the Python driver above.
> Measure CPU latency by porting the forward pass to NumPy and timing it identically.

---

## Resource Utilization

> Open in Vivado: **Reports → Report Utilization** after implementation,
> or read: `vivado/runs/impl_1/real_detect_utilization_placed.rpt`

| Resource   | Used | Available | Utilization |
|------------|------|-----------|-------------|
| LUT        | XX   | 53,200    | XX%         |
| Flip-Flop  | XX   | 106,400   | XX%         |
| BRAM (36K) | XX   | 140       | XX%         |
| DSP48E1    | XX   | 220       | XX%         |

---

## Limitations

- Fixed 64×64 input resolution
- Single image inference — no batching
- 2-class output only (Person / Object)
- Bounding box is estimated from pool3 activation hotspot, not a trained regression head
- BRAM constrained on XC7Z020 for larger feature maps

---

## Future Work

- AXI-Stream pipeline for continuous frame input
- Live camera feed integration (OV7670 / USB)
- Trained bounding box regression head
- BRAM-resident weights to eliminate DDR access latency
- Pruning and sparsity for resource reduction
- Quantization-aware training (QAT)
- Multi-class extension

---

## Authors

**Arunachalam** — CNN HLS Design, Vivado Integration, PYNQ Driver

---

## License

[MIT License](LICENSE)

---

## Acknowledgements

Built with [Vitis HLS](https://www.xilinx.com/products/design-tools/vitis/vitis-hls.html) · [Vivado](https://www.xilinx.com/products/design-tools/vivado.html) · [PYNQ](http://www.pynq.io/) · [NumPy](https://numpy.org/) · [OpenCV](https://opencv.org/)
