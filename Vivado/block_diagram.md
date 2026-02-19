# Real Detector – Vivado Block Design Documentation

**Project:** `real_detect`  
**Tool:** Vivado 2023.1  
**Target Device:** Zynq-7000 SoC (`xc7z020clg400-1`)  
**Design Status:** `write_bitstream` Complete  

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Block Descriptions](#2-block-descriptions)
3. [Signal & Interface Connections](#3-signal--interface-connections)
4. [Clock Domain](#4-clock-domain)
5. [Reset Architecture](#5-reset-architecture)
6. [AXI Interconnect Details](#6-axi-interconnect-details)
7. [Memory Map & Address Assignment](#7-memory-map--address-assignment)
8. [Interrupt Routing](#8-interrupt-routing)
9. [Data Flow Summary](#9-data-flow-summary)
10. [Recreating the Design](#10-recreating-the-design)

---

## 1. System Overview

This block design implements a **hardware-accelerated real-time detector** running on a Zynq-7020 SoC. It combines the ARM Cortex-A9 dual-core Processing System (PS) with a custom HLS-generated accelerator IP (`real_detector_0`) in the Programmable Logic (PL).

The architecture follows the standard Zynq PS↔PL paradigm:

- The **PS controls the accelerator** via an AXI-Lite control interface (`M_AXI_GP0` path).
- The **accelerator accesses DDR memory** directly via two AXI master ports (`m_axi_gmem0`, `m_axi_gmem1`) routed through the PS HP0 slave port.
- An **interrupt line** notifies the PS when the accelerator completes processing.

```
┌──────────────────────────────────────────────────────────────────┐
│                      Zynq-7000 SoC                               │
│  ┌────────────────────┐          ┌────────────────────────────┐  │
│  │  Processing System  │          │   Programmable Logic       │  │
│  │  (ARM Cortex-A9)   │          │                            │  │
│  │                    │          │  ┌──────────────────────┐  │  │
│  │  M_AXI_GP0 ────────┼──────────┼─►│ ps7_0_axi_periph    │  │  │
│  │                    │          │  │ (AXI Interconnect)   │  │  │
│  │                    │          │  └──────────┬───────────┘  │  │
│  │                    │          │             │ s_axi_control │  │
│  │                    │          │  ┌──────────▼───────────┐  │  │
│  │                    │   IRQ ◄──┼──┤   real_detector_0    │  │  │
│  │                    │          │  │  (HLS Accelerator)   │  │  │
│  │                    │          │  └────────┬─────────────┘  │  │
│  │                    │          │  m_axi_gmem0 / m_axi_gmem1 │  │
│  │                    │          │  ┌─────────▼────────────┐  │  │
│  │  S_AXI_HP0 ◄───────┼──────────┼──┤  axi_mem_intercon   │  │  │
│  │       │            │          │  │  (AXI Interconnect)  │  │  │
│  │      DDR           │          │  └──────────────────────┘  │  │
│  └────────────────────┘          └────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Block Descriptions

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/d382101a-e443-4bb7-b81f-0b3bab155a46" />

### 2.1 `processing_system7_0` – ZYNQ7 Processing System

The PS7 block instantiates the hard ARM Cortex-A9 dual-core processor subsystem embedded in the Zynq-7000.

| Port Group | Direction | Purpose |
|---|---|---|
| `DDR` | Bidirectional | External DDR3 SDRAM interface |
| `FIXED_IO` | Bidirectional | MIO bank (UART, SD, Ethernet, etc.) |
| `FCLK_CLK0` | Output | Fabric clock – 50 MHz – drives entire PL domain |
| `FCLK_RESET0_N` | Output | Active-low fabric reset signal |
| `M_AXI_GP0` | Master | General Purpose AXI master – PS writes to PL peripherals |
| `M_AXI_GP0_ACLK` | Input | Clock for GP0 port |
| `S_AXI_HP0` | Slave | High-performance slave – allows PL DMA into DDR |
| `S_AXI_HP0_ACLK` | Input | Clock for HP0 port |
| `S_AXI_HP0_FIFO_CTRL` | Interface | FIFO watermark control for HP0 |
| `IRQ_F2P` | Input | Fabric-to-PS interrupt (from real_detector_0) |

**Key Configuration:**
- FCLK_CLK0 = **50 MHz** (drives all PL logic)
- HP0 data width = **64-bit** (maximizes DDR throughput)
- IRQ_F2P enabled for HLS interrupt

---

### 2.2 `rst_ps7_0_50M` – Processor System Reset

A Xilinx utility IP that generates properly sequenced resets for the 50 MHz clock domain.

| Port | Description |
|---|---|
| `slowest_sync_clk` | Input – 50 MHz clock from PS FCLK_CLK0 |
| `ext_reset_in` | Input – Raw reset from PS FCLK_RESET0_N |
| `aux_reset_in` | Auxiliary reset input (unused, tied high internally) |
| `mb_debug_sys_rst` | MicroBlaze debug reset (unused here) |
| `dcm_locked` | DCM/PLL lock indicator input |
| `mb_reset` | Output – MicroBlaze reset (not used) |
| `bus_struct_reset[0:0]` | Output – Bus structure reset |
| `peripheral_reset[0:0]` | Output – Active-high peripheral reset |
| `interconnect_aresetn[0:0]` | Output – Active-low interconnect reset |
| `peripheral_aresetn[0:0]` | Output – **Active-low peripheral reset** → drives all AXI resets |

The `peripheral_aresetn` output is the primary reset used throughout the design.

---

### 2.3 `real_detector_0` – Real Detector HLS Accelerator

This is a custom IP core generated by **Vitis HLS** (indicated by the "Vitis™ HLS" watermark in the diagram). It implements the core signal/image/data detection algorithm in hardware.

| Port | Direction | Interface Type | Description |
|---|---|---|---|
| `s_axi_control` | Slave | AXI4-Lite | Register control interface – PS reads/writes control registers (start, status, scalar arguments) |
| `m_axi_gmem0` | Master | AXI4 Full | Memory master port 0 – burst DMA to/from DDR (e.g., input data buffer) |
| `m_axi_gmem1` | Master | AXI4 Full | Memory master port 1 – burst DMA to/from DDR (e.g., output result buffer) |
| `ap_clk` | Input | Clock | 50 MHz clock from FCLK_CLK0 |
| `ap_rst_n` | Input | Reset | Active-low reset from rst_ps7_0_50M |
| `interrupt` | Output | IRQ | Pulses high when accelerator finishes; routes to PS IRQ_F2P |

**Operating Modes (via s_axi_control registers):**
- Write `0x01` to `ap_ctrl` register → Start accelerator
- Poll `ap_ctrl` bit[1] for `ap_done`
- Or use interrupt-driven completion

---

### 2.4 `ps7_0_axi_periph` – AXI Peripheral Interconnect

A Xilinx AXI Interconnect IP routing the PS GP0 master to the HLS IP's control slave.

| Parameter | Value |
|---|---|
| Number of Slave interfaces (SI) | 1 (`S00_AXI` ← from PS M_AXI_GP0) |
| Number of Master interfaces (MI) | 1 (`M00_AXI` → to real_detector_0 s_axi_control) |
| Protocol | AXI4-Lite (control path) |
| Data width | 32-bit |

**Port Summary:**

| Port | Connected To |
|---|---|
| `S00_AXI` | `processing_system7_0/M_AXI_GP0` |
| `M00_AXI` | `real_detector_0/s_axi_control` |
| `ACLK`, `S00_ACLK`, `M00_ACLK` | `processing_system7_0/FCLK_CLK0` |
| `ARESETN`, `S00_ARESETN`, `M00_ARESETN` | `rst_ps7_0_50M/peripheral_aresetn` |

---

### 2.5 `axi_mem_intercon` – AXI Memory Interconnect

A Xilinx AXI Interconnect IP consolidating two HLS DMA masters onto the single PS HP0 slave.

| Parameter | Value |
|---|---|
| Number of Slave interfaces (SI) | 2 (`S00_AXI`, `S01_AXI` ← from HLS gmem0/gmem1) |
| Number of Master interfaces (MI) | 1 (`M00_AXI` → to PS S_AXI_HP0) |
| Protocol | AXI4 Full (memory path) |
| Data width | 64-bit (matching HP0) |

**Port Summary:**

| Port | Connected To |
|---|---|
| `S00_AXI` | `real_detector_0/m_axi_gmem0` |
| `S01_AXI` | `real_detector_0/m_axi_gmem1` |
| `M00_AXI` | `processing_system7_0/S_AXI_HP0` |
| `ACLK`, `S00/S01/M00_ACLK` | `processing_system7_0/FCLK_CLK0` |
| `ARESETN`, `S00/S01/M00_ARESETN` | `rst_ps7_0_50M/peripheral_aresetn` |

The interconnect arbitrates between the two HLS masters using a round-robin or priority scheme, serializing access to the HP0 DDR slave.

---

## 3. Signal & Interface Connections

### Complete Net List

| Signal Net | Source | Destination | Type |
|---|---|---|---|
| `FCLK_CLK0` | `processing_system7_0` | `rst_ps7_0_50M/slowest_sync_clk` | Clock |
| `FCLK_CLK0` | `processing_system7_0` | `real_detector_0/ap_clk` | Clock |
| `FCLK_CLK0` | `processing_system7_0` | `ps7_0_axi_periph/ACLK,S00_ACLK,M00_ACLK` | Clock |
| `FCLK_CLK0` | `processing_system7_0` | `axi_mem_intercon/ACLK,S00_ACLK,S01_ACLK,M00_ACLK` | Clock |
| `FCLK_CLK0` | `processing_system7_0` | `processing_system7_0/M_AXI_GP0_ACLK` | Clock (loopback) |
| `FCLK_CLK0` | `processing_system7_0` | `processing_system7_0/S_AXI_HP0_ACLK` | Clock (loopback) |
| `FCLK_RESET0_N` | `processing_system7_0` | `rst_ps7_0_50M/ext_reset_in` | Reset |
| `peripheral_aresetn` | `rst_ps7_0_50M` | `real_detector_0/ap_rst_n` | Reset |
| `peripheral_aresetn` | `rst_ps7_0_50M` | `ps7_0_axi_periph/(all ARESETN)` | Reset |
| `peripheral_aresetn` | `rst_ps7_0_50M` | `axi_mem_intercon/(all ARESETN)` | Reset |
| `M_AXI_GP0` bus | `processing_system7_0` | `ps7_0_axi_periph/S00_AXI` | AXI4-Lite |
| `M00_AXI` bus | `ps7_0_axi_periph` | `real_detector_0/s_axi_control` | AXI4-Lite |
| `m_axi_gmem0` bus | `real_detector_0` | `axi_mem_intercon/S00_AXI` | AXI4 Full |
| `m_axi_gmem1` bus | `real_detector_0` | `axi_mem_intercon/S01_AXI` | AXI4 Full |
| `M00_AXI` bus | `axi_mem_intercon` | `processing_system7_0/S_AXI_HP0` | AXI4 Full |
| `interrupt` | `real_detector_0` | `processing_system7_0/IRQ_F2P` | IRQ |
| `DDR` | `processing_system7_0` | External DDR3 SDRAM | PHY |
| `FIXED_IO` | `processing_system7_0` | External MIO (UART, SD, etc.) | PHY |

---

## 4. Clock Domain

The entire design operates in a **single clock domain** at **50 MHz** driven by `FCLK_CLK0` from the PS7 block.

```
processing_system7_0/FCLK_CLK0 (50 MHz)
        │
        ├──► rst_ps7_0_50M/slowest_sync_clk
        ├──► real_detector_0/ap_clk
        ├──► ps7_0_axi_periph/ACLK, S00_ACLK, M00_ACLK
        ├──► axi_mem_intercon/ACLK, S00_ACLK, S01_ACLK, M00_ACLK
        ├──► processing_system7_0/M_AXI_GP0_ACLK  (feedback)
        └──► processing_system7_0/S_AXI_HP0_ACLK  (feedback)
```

Since everything runs at 50 MHz, there are no CDC (Clock Domain Crossing) issues in this design.

---

## 5. Reset Architecture

Reset is generated and distributed by the `rst_ps7_0_50M` IP. The active-low `peripheral_aresetn` signal is the primary reset used across all PL components.

```
processing_system7_0/FCLK_RESET0_N (active-low) ──► rst_ps7_0_50M/ext_reset_in
                                                          │
                                              peripheral_aresetn (active-low)
                                                          │
                  ┌───────────────────────────────────────┤
                  │                                       │
          real_detector_0/ap_rst_n          ps7_0_axi_periph & axi_mem_intercon
                                            (ARESETN, S00/S01/M00_ARESETN)
```

The reset block ensures proper synchronization and sequencing – AXI interconnects and IP cores only come out of reset after the clock is stable and the PS reset signal is deasserted.

---

## 6. AXI Interconnect Details

### 6.1 Control Path: `ps7_0_axi_periph`

This interconnect carries **lightweight register-space traffic** from the ARM CPU to the HLS IP.

- **Protocol:** AXI4-Lite (fixed 32-bit data, no burst)
- **Purpose:** PS firmware reads/writes HLS control registers:
  - `0x00`: `ap_ctrl` – start, done, idle, ready bits
  - `0x04`: Global Interrupt Enable
  - `0x08`: IP Interrupt Enable
  - `0x0C`: IP Interrupt Status
  - `0x10+`: Scalar arguments (e.g., array pointers, sizes)

### 6.2 Data Path: `axi_mem_intercon`

This interconnect carries **high-throughput DMA traffic** between the HLS engine and DDR.

- **Protocol:** AXI4 Full (burst capable, 64-bit data)
- **Two slave ports** arbitrated → **one master** to HP0
- The HP0 port provides a high-bandwidth, low-latency path directly into the DDR controller, bypassing the CPU L2 cache
- HP0 supports up to 1200 MB/s peak theoretical bandwidth at 64-bit/150 MHz, reduced to ~800 MB/s at 50 MHz

**Why two gmem ports?** The HLS tool partitions memory interfaces to allow concurrent access to separate arrays (e.g., read input from `gmem0` while writing output to `gmem1`), improving pipeline efficiency and memory bandwidth utilization.

---

## 7. Memory Map & Address Assignment

### PS Master → PL Slave (Control via M_AXI_GP0)

| Segment | Base Address | Size | Target |
|---|---|---|---|
| `real_detector_0/s_axi_control` | `0x43C0_0000` | 64 KB | HLS control registers |

The CPU accesses the HLS IP at physical address `0x43C00000`. Typical Linux kernel usage via UIO or mmap:

```c
int fd = open("/dev/uio0", O_RDWR);
void *ctrl = mmap(NULL, 65536, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
// Start accelerator
*((volatile uint32_t*)ctrl + 0x00) = 0x01;
```

### PL Master → PS Slave (DMA via S_AXI_HP0)

| Segment | Base Address | Size | Target |
|---|---|---|---|
| `HP0_DDR_LOWOCM` | `0x0000_0000` | 1 GB | DDR3 SDRAM |

The HLS IP is given the physical base address of input/output buffers (typically allocated via `mmap` or `dma_alloc_coherent`) as scalar arguments through the control interface.

---

## 8. Interrupt Routing

```
real_detector_0/interrupt ──► processing_system7_0/IRQ_F2P[0]
```

- The `interrupt` port pulses high for one clock cycle when `ap_done` is asserted
- `IRQ_F2P` maps to Zynq PL-to-PS interrupt ID **61** (IRQ #61 in Linux, offset from GIC base)
- Linux driver registers an ISR with `request_irq()` to service the interrupt and signal completion to user space (e.g., via poll/select on a UIO device)

**Interrupt flow:**
1. PS writes `0x01` to `ap_ctrl` → HLS accelerator starts
2. HLS completes computation → `ap_done` pulses → `interrupt` pin asserts
3. GIC delivers interrupt to CPU → ISR executes
4. ISR acknowledges interrupt and notifies user-space application
5. Application reads results from DDR output buffer

---

## 9. Data Flow Summary

### Control Flow (ARM → Accelerator)

```
ARM CPU
  └─► M_AXI_GP0 (AXI4-Lite, 32-bit)
        └─► ps7_0_axi_periph (1-to-1 routing)
              └─► real_detector_0/s_axi_control
                    └─► Control registers (start, stop, args)
```

### Data Flow (Accelerator ↔ DDR)

```
DDR3 Memory
  ↕ (S_AXI_HP0, AXI4 Full, 64-bit)
axi_mem_intercon (2-to-1 arbiter)
  ├─ S00_AXI ↔ real_detector_0/m_axi_gmem0  (input data channel)
  └─ S01_AXI ↔ real_detector_0/m_axi_gmem1  (output data channel)
```

### Interrupt Flow (Accelerator → ARM)

```
real_detector_0/interrupt
  └─► IRQ_F2P → GIC → ARM CPU → ISR
```

---

## 10. Recreating the Design

### Prerequisites

- Vivado 2023.1 installed
- Zynq-7020 target board (e.g., Digilent Zybo Z7-20 or ZedBoard)
- `real_detector` HLS IP added to IP repository

### Steps

**Option A – Open existing project:**
```bash
vivado -source recreate_project.tcl
# or open the .xpr directly:
vivado real_detect/real_detect.xpr
```

**Option B – Recreate block design from TCL:**
```tcl
# In Vivado TCL console:
source create_bd.tcl
```

**Option C – Manual GUI steps:**
1. Create new RTL project, target `xc7z020clg400-1`
2. Add real_detector IP to IP catalog (Settings → IP → Repository)
3. Create Block Design: Flow Navigator → IP Integrator → Create Block Design
4. Add IPs: `processing_system7`, `proc_sys_reset`, `axi_interconnect` (×2), `real_detector`
5. Run Block Automation on PS7 (applies board preset)
6. Connect all ports as documented in Section 3
7. Assign addresses as in Section 7
8. Validate design → Generate output products → Create HDL wrapper
9. Run Synthesis → Implementation → Generate Bitstream

### Build Flow (after bitstream is complete)

```bash
# Program device
open_hw_manager
connect_hw_server
open_hw_target
program_hw_devices -bitfile real_detect.runs/impl_1/real_detect_wrapper.bit

# Deploy Linux + driver
# Copy bitstream to SD card /boot/
# Load overlay or use full bitstream boot
```

---

## Appendix: Key IP Versions

| IP | Version | Vendor |
|---|---|---|
| processing_system7 | 5.5 | xilinx.com |
| proc_sys_reset | 5.0 | xilinx.com |
| axi_interconnect | 2.1 | xilinx.com |
| real_detector (HLS) | 1.0 | User/Custom |

---

*Documentation generated from Vivado 2023.1 Block Design screenshot. Design status: `write_bitstream` Complete.*
