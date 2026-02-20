# 🚀 Real-Time CNN Acceleration on PYNQ-Z2 FPGA

> A hardware-accelerated Convolutional Neural Network inference system deployed on the PYNQ-Z2 FPGA board, achieving significant speedup over ARM CPU-only execution.

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Hardware & Software Requirements](#hardware--software-requirements)
- [Project Structure](#project-structure)
- [Setup Guide](#setup-guide)
  - [Step 1 — Install PC Software](#step-1--install-pc-software)
  - [Step 2 — Flash PYNQ Image to SD Card](#step-2--flash-pynq-image-to-sd-card)
  - [Step 3 — Boot the PYNQ-Z2 Board](#step-3--boot-the-pynq-z2-board)
  - [Step 4 — Connect to the Board](#step-4--connect-to-the-board)
  - [Step 5 — Transfer Project Files](#step-5--transfer-project-files)
  - [Step 6 — Load the FPGA Bitstream](#step-6--load-the-fpga-bitstream)
  - [Step 7 — Prepare Input Images](#step-7--prepare-input-images)
- [Running Inference](#running-inference)
  - [Hardware Accelerated (FPGA)](#hardware-accelerated-fpga)
  - [Software Baseline (CPU Only)](#software-baseline-cpu-only)
  - [Benchmark Multiple Images](#benchmark-multiple-images)
- [Measuring Performance](#measuring-performance)
- [Checking FPGA Resource Utilization](#checking-fpga-resource-utilization)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [Results Summary](#results-summary)
- [Shutting Down the Board](#shutting-down-the-board)

---

## Overview

This project implements a CNN-based image classifier accelerated on an FPGA using **Vitis HLS** for hardware synthesis and **PYNQ** for Python-based control. The accelerator offloads convolution and pooling layers to programmable logic (PL), while the ARM processor handles pre/post-processing.

**Key Features:**
- FPGA-accelerated inference with reduced latency vs. CPU-only
- Python-driven control via the PYNQ framework
- Supports 64×64 grayscale image classification
- Side-by-side latency benchmarking (FPGA vs. CPU)

---

## System Architecture

```
┌─────────────────────────────────────────────┐
│                  PYNQ-Z2                     │
│                                             │
│   ┌─────────────┐       ┌───────────────┐   │
│   │  ARM Cortex │◄─────►│  CNN IP Core  │   │
│   │  A9 (PS)    │  AXI  │  (PL/FPGA)    │   │
│   │             │       │               │   │
│   │  run_fpga.py│       │  Conv + Pool  │   │
│   │  pre/post   │       │  layers in HW │   │
│   └─────────────┘       └───────────────┘   │
│          │                                  │
│        DDR RAM (shared memory)              │
└─────────────────────────────────────────────┘
```

---

## Hardware & Software Requirements

### Hardware

| Component | Specification |
|-----------|--------------|
| FPGA Board | PYNQ-Z2 |
| SD Card | 8 GB or larger (Class 10 recommended) |
| Power Supply | 5V DC adapter |
| Network | Ethernet cable |
| PC/Laptop | Ubuntu 20.04+ recommended |

### Software (on PC)

| Tool | Version |
|------|---------|
| Vivado | 2023.1 |
| Vitis HLS | 2023.1 |
| Python | 3.8+ |
| Balena Etcher | Latest |

### Python Packages (on PC)

```bash
pip install numpy opencv-python matplotlib
```

---

## Project Structure

```
project/
├── hardware/
│   ├── real_detect.bit          # FPGA bitstream
│   └── real_detect.hwh          # Hardware handoff file
├── software/
│   ├── run_fpga.py              # Main FPGA inference script
│   ├── run_cpu.py               # CPU-only baseline script
│   └── benchmark.py             # Multi-image benchmark script
├── weights/
│   └── cnn_weights.npy          # Pre-trained weight files
├── images/
│   └── test_images/             # Sample test images
├── reports/
│   └── utilization.rpt          # Vivado resource report
└── README.md
```

---

## Setup Guide

### Step 1 — Install PC Software

Install Vivado and Vitis HLS from the [Xilinx Downloads page](https://www.xilinx.com/support/download.html).

Then install required Python packages:

```bash
pip install numpy opencv-python matplotlib
```

---

### Step 2 — Flash PYNQ Image to SD Card

1. Download the official PYNQ-Z2 image from [pynq.io](https://www.pynq.io/)

2. Insert your SD card into the PC

3. Open **Balena Etcher** ([download here](https://etcher.balena.io/))

4. Click **Flash from file** → select the downloaded `.img` file

5. Click **Select target** → choose your SD card

6. Click **Flash!** and wait for completion

7. Eject the SD card safely

> ⚠️ **Warning:** This will erase all data on the SD card. Back up anything important first.

---

### Step 3 — Boot the PYNQ-Z2 Board

1. Insert the flashed SD card into the PYNQ-Z2 board
2. Connect the Ethernet cable between the board and your PC (or router)
3. Connect the 5V power adapter
4. Flip the power switch **ON**
5. Wait approximately **60 seconds** for the board to fully boot

The **DONE LED** will light up when the boot is complete.

---

### Step 4 — Connect to the Board

Open a terminal on your PC and SSH into the board:

```bash
ssh xilinx@pynq
```

**Password:** `xilinx`

If the hostname `pynq` doesn't resolve, use the default IP address:

```bash
ssh xilinx@192.168.2.99
```

> 💡 **Tip:** If connecting over a router, find the board's IP from the router's DHCP client list or connect directly via USB-UART and run `ifconfig`.

---

### Step 5 — Transfer Project Files

From your **PC terminal**, copy the required files to the board:

```bash
# Transfer bitstream and hardware handoff
scp hardware/real_detect.bit xilinx@pynq:/home/xilinx/
scp hardware/real_detect.hwh xilinx@pynq:/home/xilinx/

# Transfer Python scripts
scp software/run_fpga.py xilinx@pynq:/home/xilinx/
scp software/run_cpu.py xilinx@pynq:/home/xilinx/
scp software/benchmark.py xilinx@pynq:/home/xilinx/

# Transfer weight files
scp weights/cnn_weights.npy xilinx@pynq:/home/xilinx/
```

> ⚠️ **Important:** The `.bit` and `.hwh` files **must** be in the same directory and have the same base filename.

---

### Step 6 — Load the FPGA Bitstream

SSH into the board and verify the bitstream loads correctly:

```bash
ssh xilinx@pynq
cd /home/xilinx
python3
```

Inside the Python shell:

```python
from pynq import Overlay

ol = Overlay("/home/xilinx/real_detect.bit")
ip = ol.real_detector_0
print("FPGA loaded successfully ✓")
```

If you see no errors and the success message prints, the overlay is loaded correctly.

```python
exit()
```

---

### Step 7 — Prepare Input Images

Create a folder for test images on the board:

```bash
mkdir -p /home/xilinx/images
```

Transfer images from your PC:

```bash
scp images/cat.jpg xilinx@pynq:/home/xilinx/images/
scp images/dog.jpg xilinx@pynq:/home/xilinx/images/
```

**Accepted formats:** `.jpg`, `.png`  
**Size:** Any size — the script automatically resizes to 64×64.

---

## Running Inference

### Hardware Accelerated (FPGA)

```bash
python3 run_fpga.py
```

**What the script does internally:**

```
1. Loads the FPGA overlay (bitstream)
2. Allocates shared DDR memory buffers
3. Loads CNN weight files
4. Resizes input image to 64×64 grayscale
5. Writes memory addresses to IP registers
6. Starts the hardware accelerator
7. Waits for the done signal
8. Reads output and applies softmax
9. Returns class label and confidence score
```

**Example output:**

```
Loading overlay...
Running inference on: cat.jpg
-----------------------------
Class      : Cat
Confidence : 78%
Latency    : 150 ms
FPS        : 6.7
-----------------------------
```

---

### Software Baseline (CPU Only)

Run the ARM-only version for comparison:

```bash
python3 run_cpu.py
```

**Example output:**

```
Running CPU inference on: cat.jpg
-----------------------------
Class      : Cat
Confidence : 78%
Latency    : 890 ms
FPS        : 1.1
-----------------------------
```

---

### Benchmark Multiple Images

To test across an entire folder of images and get aggregate stats:

```bash
python3 benchmark.py
```

**Example output:**

```
===== BENCHMARK RESULTS =====
Images tested   : 20
Avg Latency     : 152 ms
FPS             : 6.6
Accuracy        : 75%
==============================
```

---

## Measuring Performance

Latency is printed automatically after each inference run.

For manual timing inside a script:

```python
import time

t0 = time.time()
run_cnn(img)
elapsed_ms = (time.time() - t0) * 1000
print(f"Latency: {elapsed_ms:.2f} ms")
print(f"FPS: {1000/elapsed_ms:.2f}")
```

---

## Checking FPGA Resource Utilization

After Vivado implementation, open the utilization report on your PC:

```
project.runs/impl_1/utilization.rpt
```

Key metrics to look for:

| Resource | Description |
|----------|-------------|
| LUT % | Logic utilization |
| DSP % | DSP48 slice usage (multiply-accumulate) |
| BRAM % | Block RAM usage for weights/buffers |
| FF % | Flip-flop registers |

> ⚠️ **Note:** DSP usage is currently at 100% — no headroom for additional operations.

---

## Troubleshooting

### ❌ Cannot SSH into Board

```bash
ping pynq
```

If no response:
- Check the Ethernet cable is properly connected
- Try the IP address directly: `ssh xilinx@192.168.2.99`
- Check your PC's network adapter is set to a static IP in the `192.168.2.x` subnet
- Verify the board has fully booted (DONE LED should be ON)

---

### ❌ Overlay Fails to Load

```
RuntimeError: Cannot find overlay
```

Check:
- The `.bit` and `.hwh` files exist in the **same folder**
- The filenames have the **same base name** (e.g., `real_detect.bit` and `real_detect.hwh`)
- You are providing the **full absolute path** in the `Overlay()` call

---

### ❌ Timeout / Accelerator Hangs

```
TimeoutError: IP core did not complete
```

Check:
- IP register addresses match the HLS-generated address map
- DDR buffer addresses are written to the correct AXI-Lite registers
- The accelerator clock (`ap_clk`) is properly connected in the block design
- The `ap_start` bit is being set correctly

---

### ❌ Wrong Predictions

- Verify weight files match the trained model architecture
- Check that preprocessing (resize, grayscale conversion, normalization) matches training
- Confirm pixel values are in the expected range (0–1 float or 0–255 uint8)

---

### ❌ Inference is Very Slow

- Confirm `run_fpga.py` is running (not `run_cpu.py`)
- Verify the bitstream is loaded: check `ol` object is not `None`
- Ensure the accelerator is not falling back to software emulation

---

## Known Limitations

| Limitation | Detail |
|------------|--------|
| Input size | Fixed at 64×64 pixels |
| DSP usage | 100% — no room for additional logic |
| Data transfer | No DMA — uses memory-mapped AXI Lite |
| Model | Fixed CNN architecture, not configurable |
| Scalability | Not suitable for larger models (ResNet, VGG, etc.) |

---

## Results Summary

| Metric | FPGA | CPU |
|--------|------|-----|
| Avg Latency | ~150 ms | ~890 ms |
| FPS | ~6.7 | ~1.1 |
| Speedup | **~5.9×** | baseline |
| LUT Usage | ~65% | — |
| DSP Usage | 100% | — |
| BRAM Usage | ~40% | — |

---

## Shutting Down the Board

Always shut down properly to avoid SD card corruption:

```bash
sudo shutdown now
```

Wait until all LEDs turn off, then unplug the power adapter.

---

## License

This project is developed for academic/research purposes.

---

## Acknowledgements

- [PYNQ Project](http://www.pynq.io/) — Xilinx/AMD
- [Vitis HLS Documentation](https://docs.xilinx.com/r/en-US/ug1399-vitis-hls)
- [Balena Etcher](https://etcher.balena.io/)
