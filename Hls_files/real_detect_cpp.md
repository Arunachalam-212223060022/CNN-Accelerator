# `hls/cnn_detect.cpp` — HLS Accelerator

## What This File Does

This is the **heart of the entire project** — the C++ function that gets synthesized by Vitis HLS into actual FPGA hardware logic. When you run `vitis_hls run_hls.tcl`, this file is compiled into RTL (Verilog/VHDL) and then placed-and-routed onto the XC7Z020 FPGA fabric.

Every `for` loop here becomes a **hardware circuit** running at 100 MHz. The `#pragma HLS` directives are instructions to the HLS compiler about how to build that hardware — whether to pipeline it, what memory type to use, and so on.

The function runs the full CNN forward pass:
1. Load image from DDR
2. Conv1 + ReLU + Pool1
3. Conv2 + ReLU + Pool2
4. Conv3 + ReLU + Pool3
5. Fully Connected + ReLU
6. ArgMax (Cat vs Dog)
7. Bounding box (activation hotspot)
8. Write all results back to DDR

---

## Full File

```cpp
#include "cnn_detect.h"

// ============================================================
// CNN Accelerator — Cat vs Dog Classifier
// PYNQ-Z2 (XC7Z020) | Vitis HLS 2023.1
//
// Architecture:
//   Conv1: 64×64×1  → 62×62×8   (3×3, 8 filters)
//   Pool1: 62×62×8  → 31×31×8   (2×2 max)
//   Conv2: 31×31×8  → 29×29×16  (3×3, 16 filters)
//   Pool2: 29×29×16 → 14×14×16  (2×2 max)
//   Conv3: 14×14×16 → 12×12×32  (3×3, 32 filters)
//   Pool3: 12×12×32 →  6×6×32   (2×2 max)
//   FC:     1152    →    64      (fully connected)
//   Out:      64    →     2      (Cat=0, Dog=1)
// ============================================================

void cnn_detect(
    ap_uint<8>   *image,
    ap_int<8>    *conv1_w,  ap_int<8> *conv1_b,
    ap_int<8>    *conv2_w,  ap_int<8> *conv2_b,
    ap_int<8>    *conv3_w,  ap_int<8> *conv3_b,
    ap_int<8>    *fc_w,     ap_int<8> *fc_b,
    volatile int *result
) {
```

---

## Section 1 — AXI Interface Pragmas

```cpp
// AXI Master — gmem0: image + result (shared HP port)
#pragma HLS INTERFACE m_axi port=image  offset=slave bundle=gmem0 depth=4096
#pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem0 depth=16

// AXI Master — gmem1: all weight arrays (shared HP port)
#pragma HLS INTERFACE m_axi port=conv1_w offset=slave bundle=gmem1 depth=72
#pragma HLS INTERFACE m_axi port=conv1_b offset=slave bundle=gmem1 depth=8
#pragma HLS INTERFACE m_axi port=conv2_w offset=slave bundle=gmem1 depth=1152
#pragma HLS INTERFACE m_axi port=conv2_b offset=slave bundle=gmem1 depth=16
#pragma HLS INTERFACE m_axi port=conv3_w offset=slave bundle=gmem1 depth=4608
#pragma HLS INTERFACE m_axi port=conv3_b offset=slave bundle=gmem1 depth=32
#pragma HLS INTERFACE m_axi port=fc_w    offset=slave bundle=gmem1 depth=73728
#pragma HLS INTERFACE m_axi port=fc_b    offset=slave bundle=gmem1 depth=64

// AXI-Lite — control: all port addresses + return
#pragma HLS INTERFACE s_axilite port=image    bundle=control
#pragma HLS INTERFACE s_axilite port=conv1_w  bundle=control
// ... (all other ports)
#pragma HLS INTERFACE s_axilite port=return   bundle=control
```

**What these pragmas do:**

| Pragma | Meaning |
|--------|---------|
| `m_axi` | This port becomes an AXI Master — the FPGA initiates DDR reads/writes |
| `bundle=gmem0` | Group `image` and `result` onto the same AXI bus (one HP port) |
| `bundle=gmem1` | Group all weight arrays onto a second AXI bus (same HP port, separate arbitration) |
| `depth=4096` | Tells HLS the max number of elements this pointer can access (for simulation) |
| `offset=slave` | The base address is provided via AXI-Lite registers (not hardcoded) |
| `s_axilite` | This port is a control register — the ARM writes the DDR address into it |
| `port=return` | The function's return register — ARM polls this to know inference is done |

**Why two bundles?** `gmem0` carries image (4096 bytes) and result (36 bytes). `gmem1` carries weights (79,680 bytes). Separating them allows parallel DDR transactions — the HLS scheduler can pipeline weight reads with computation.

---

## Section 2 — Feature Map Buffers

```cpp
ap_uint<8>  img_buf[64][64];

ap_int<16>  conv1[62][62][8];
#pragma HLS BIND_STORAGE variable=conv1 type=RAM_2P impl=BRAM
ap_int<16>  pool1[31][31][8];
#pragma HLS BIND_STORAGE variable=pool1 type=RAM_2P impl=BRAM

ap_int<16>  conv2[29][29][16];
#pragma HLS BIND_STORAGE variable=conv2 type=RAM_2P impl=BRAM
ap_int<16>  pool2[14][14][16];
#pragma HLS BIND_STORAGE variable=pool2 type=RAM_2P impl=BRAM

ap_int<16>  conv3[12][12][32];
#pragma HLS BIND_STORAGE variable=conv3 type=RAM_2P impl=BRAM
ap_int<16>  pool3[6][6][32];
#pragma HLS BIND_STORAGE variable=pool3 type=RAM_2P impl=BRAM

ap_int<32>  fc_out[64];
#pragma HLS ARRAY_PARTITION variable=fc_out complete

ap_int<32>  scores[2];
#pragma HLS ARRAY_PARTITION variable=scores complete
```

These are **on-chip buffers** — local arrays that live inside the FPGA, not in DDR. Each one holds the output of one layer, which becomes the input to the next.

| Buffer | Shape | Type | Why this type |
|--------|-------|------|---------------|
| `img_buf` | 64×64 | uint8 | Raw pixel values (0–255) |
| `conv1` | 62×62×8 | int16 | INT8×INT8 accumulation needs wider precision |
| `pool1` | 31×31×8 | int16 | Half the spatial size after pooling |
| `conv2` | 29×29×16 | int16 | More filters → more channels |
| `pool2` | 14×14×16 | int16 | |
| `conv3` | 12×12×32 | int16 | |
| `pool3` | 6×6×32 | int16 | Final feature map — source for FC and BBox |
| `fc_out` | 64 | int32 | Accumulated from 1152 inputs — needs int32 |
| `scores` | 2 | int32 | Final Cat/Dog scores |

**`BIND_STORAGE type=RAM_2P impl=BRAM`** — forces large arrays into BRAM (Block RAM) tiles instead of LUT-based RAM. BRAM is a dedicated memory resource on Xilinx FPGAs. Without this pragma, HLS might use thousands of LUTs as flip-flop registers to implement the array, wasting logic resources.

**`ARRAY_PARTITION complete`** — splits the array into individual registers with no shared bus. `fc_out[64]` becomes 64 separate flip-flops, all readable in the same clock cycle. Used for small arrays where parallel access is needed (all 64 FC outputs need to be summed at once).

---

## Section 3 — Load Image

```cpp
LOAD_Y: for(int y = 0; y < 64; y++) {
    LOAD_X: for(int x = 0; x < 64; x++) {
        #pragma HLS PIPELINE II=1
        img_buf[y][x] = image[y * 64 + x];
    }
}
```

**What it does:** Reads the 64×64 image from DDR (via AXI Master gmem0) into the on-chip `img_buf` SRAM. DDR access has variable latency; on-chip SRAM access is 1 clock cycle. Copying to on-chip first ensures all subsequent convolution reads are fast and predictable.

**Loop labels (`LOAD_Y`, `LOAD_X`):** Named labels let Vitis HLS identify loops in its synthesis report. You can see per-loop latency and II in the `.rpt` file.

**`#pragma HLS PIPELINE II=1`:** II = Initiation Interval. II=1 means a new loop iteration starts every 1 clock cycle. Without this pragma, the compiler would implement a sequential loop that takes many cycles per iteration. With PIPELINE II=1, the inner loop issues one DDR read per clock, making the entire 4096-byte image load complete in ~4096 + latency cycles.

---

## Section 4 — Convolution Layer 1

```cpp
CONV1_Y: for(int y = 0; y < 62; y++) {
    CONV1_X: for(int x = 0; x < 62; x++) {
        CONV1_F: for(int f = 0; f < 8; f++) {
            #pragma HLS PIPELINE II=1

            ap_int<32> acc = conv1_b[f];  // initialize with bias

            CONV1_KY: for(int ky = 0; ky < 3; ky++) {
                CONV1_KX: for(int kx = 0; kx < 3; kx++) {
                    ap_uint<8> px = img_buf[y + ky][x + kx];
                    ap_int<8>  w  = conv1_w[f * 9 + ky * 3 + kx];
                    acc += (ap_int<32>)px * (ap_int<32>)w;
                }
            }

            // ReLU
            if(acc < 0) acc = 0;

            // Scale down: divide by 16 (right shift 4 bits)
            conv1[y][x][f] = (ap_int<16>)(acc >> 4);
        }
    }
}
```

**Why the output is 62×62:** No padding is used. A 3×3 kernel on a 64×64 image with stride=1 gives output size = `64 - 3 + 1 = 62`.

**Weight indexing `conv1_w[f * 9 + ky * 3 + kx]`:** The weights are stored flat in DDR. For filter `f`, the 3×3 kernel is stored at positions `f*9` to `f*9+8`. The 2D kernel position `(ky, kx)` maps to linear index `ky*3 + kx`.

**Accumulator type `ap_int<32>`:** An INT8 pixel × INT8 weight = INT16 product. Summing 9 such products (3×3 kernel) gives up to INT16 × 9 ≈ INT20. Across 1 input channel the worst case is safe in INT32. Using INT32 accumulator prevents overflow.

**`acc >> 4` (divide by 16):** After accumulation the values are large. Right-shifting by 4 is equivalent to dividing by 16 — a fast approximation to scale the output back to a range that fits in INT16. This is a standard fixed-point quantization technique.

**ReLU:** Simply clamp negatives to zero: `if(acc < 0) acc = 0`. In hardware this synthesizes to a single comparator and mux.

**`PIPELINE II=1` on the filter loop:** The innermost effective loop that gets pipelined is the filter loop. The 3×3 kernel loops (KY, KX) are **fully unrolled** by HLS because they are small fixed-size loops inside a pipeline — meaning all 9 multiply-accumulate operations happen in parallel in the same pipeline stage.

---

## Section 5 — Max Pooling Layer 1

```cpp
POOL1_Y: for(int y = 0; y < 31; y++) {
    POOL1_X: for(int x = 0; x < 31; x++) {
        POOL1_F: for(int f = 0; f < 8; f++) {
            #pragma HLS PIPELINE II=1

            ap_int<16> max_val = 0;

            POOL1_KY: for(int ky = 0; ky < 2; ky++) {
                POOL1_KX: for(int kx = 0; kx < 2; kx++) {
                    ap_int<16> v = conv1[y*2 + ky][x*2 + kx][f];
                    if(v > max_val) max_val = v;
                }
            }

            pool1[y][x][f] = max_val;
        }
    }
}
```

**What max pooling does:** Takes the maximum value from each 2×2 non-overlapping window. This halves the spatial dimensions: `62×62 → 31×31`. It discards positional detail while keeping the strongest activation in each region — giving the network translation invariance and reducing compute for subsequent layers.

**`y*2` and `x*2`:** The pooling window for output pixel `(y,x)` covers input pixels `(y*2, x*2)` to `(y*2+1, x*2+1)` — stride=2, window=2.

**The 2×2 inner loops are unrolled by HLS** — all four comparisons happen in a single clock cycle, implemented as a 4-input max tree in hardware.

---

## Section 6 — Convolution Layer 2

```cpp
CONV2_Y: for(int y = 0; y < 29; y++) {
    CONV2_X: for(int x = 0; x < 29; x++) {
        CONV2_F: for(int f = 0; f < 16; f++) {
            #pragma HLS PIPELINE II=1

            ap_int<32> acc = conv2_b[f];

            CONV2_C: for(int c = 0; c < 8; c++) {        // 8 input channels
                CONV2_KY: for(int ky = 0; ky < 3; ky++) {
                    CONV2_KX: for(int kx = 0; kx < 3; kx++) {
                        ap_int<16> px = pool1[y + ky][x + kx][c];
                        ap_int<8>  w  = conv2_w[(f * 8 + c) * 9 + ky * 3 + kx];
                        acc += (ap_int<32>)px * (ap_int<32>)w;
                    }
                }
            }

            if(acc < 0) acc = 0;
            conv2[y][x][f] = (ap_int<16>)(acc >> 4);
        }
    }
}
```

**Key difference from Conv1:** Conv2 has 8 **input channels** (from pool1's 8 feature maps). This adds the `CONV2_C` channel loop. Each output feature map `f` is the sum of 3×3 convolutions across all 8 input channels.

**Weight indexing `conv2_w[(f * 8 + c) * 9 + ky * 3 + kx]`:**
- `f * 8 + c` selects the specific filter/channel pair
- `* 9` steps to the start of the 3×3 kernel for that pair
- `+ ky * 3 + kx` selects the kernel position

**Output is 29×29** because: `31 - 3 + 1 = 29` (same valid convolution formula).

---

## Section 7 — Max Pooling Layer 2

```cpp
POOL2_Y: for(int y = 0; y < 14; y++) {
    POOL2_X: for(int x = 0; x < 14; x++) {
        POOL2_F: for(int f = 0; f < 16; f++) {
            #pragma HLS PIPELINE II=1

            ap_int<16> max_val = 0;
            POOL2_KY: for(int ky = 0; ky < 2; ky++) {
                POOL2_KX: for(int kx = 0; kx < 2; kx++) {
                    if(y*2+ky < 29 && x*2+kx < 29) {   // bounds check
                        ap_int<16> v = conv2[y*2+ky][x*2+kx][f];
                        if(v > max_val) max_val = v;
                    }
                }
            }
            pool2[y][x][f] = max_val;
        }
    }
}
```

**Bounds check `if(y*2+ky < 29)`:** Pool2 maps a 29×29 input into a 14×14 output. Since `29 / 2 = 14.5`, the last row/column of the output window would read out-of-bounds. The bounds check skips those accesses safely. (In hardware this becomes a disable signal on the comparator, not a branch.)

---

## Section 8 — Convolution Layer 3

```cpp
CONV3_Y: for(int y = 0; y < 12; y++) {
    CONV3_X: for(int x = 0; x < 12; x++) {
        CONV3_F: for(int f = 0; f < 32; f++) {
            #pragma HLS PIPELINE II=1

            ap_int<32> acc = conv3_b[f];

            CONV3_C: for(int c = 0; c < 16; c++) {        // 16 input channels
                CONV3_KY: for(int ky = 0; ky < 3; ky++) {
                    CONV3_KX: for(int kx = 0; kx < 3; kx++) {
                        ap_int<16> px = pool2[y + ky][x + kx][c];
                        ap_int<8>  w  = conv3_w[(f * 16 + c) * 9 + ky * 3 + kx];
                        acc += (ap_int<32>)px * (ap_int<32>)w;
                    }
                }
            }

            if(acc < 0) acc = 0;
            conv3[y][x][f] = (ap_int<16>)(acc >> 4);
        }
    }
}
```

**The deepest conv layer:** 16 input channels × 32 output filters × 3×3 kernel. The accumulator now sums `16 × 9 = 144` multiply-accumulate operations per output pixel per filter. Still safe in INT32.

**Output is 12×12:** `14 - 3 + 1 = 12`.

---

## Section 9 — Max Pooling Layer 3

```cpp
POOL3_Y: for(int y = 0; y < 6; y++) {
    POOL3_X: for(int x = 0; x < 6; x++) {
        POOL3_F: for(int f = 0; f < 32; f++) {
            #pragma HLS PIPELINE II=1

            ap_int<16> max_val = 0;
            POOL3_KY: for(int ky = 0; ky < 2; ky++) {
                POOL3_KX: for(int kx = 0; kx < 2; kx++) {
                    ap_int<16> v = conv3[y*2+ky][x*2+kx][f];
                    if(v > max_val) max_val = v;
                }
            }
            pool3[y][x][f] = max_val;
        }
    }
}
```

**Output is 6×6×32:** `12 / 2 = 6`. This is the final spatial feature map. It is used for both the FC layer and the bounding box estimation. Total elements: `6 × 6 × 32 = 1152`.

---

## Section 10 — Fully Connected Layer

```cpp
// Initialize accumulators with bias
FC_INIT: for(int n = 0; n < 64; n++) {
    #pragma HLS UNROLL
    fc_out[n] = fc_b[n];
}

// Accumulate: fc_out[n] = sum(pool3[y][x][c] * fc_w[n * 1152 + idx])
FC_N: for(int n = 0; n < 64; n++) {
    FC_Y: for(int y = 0; y < 6; y++) {
        FC_X: for(int x = 0; x < 6; x++) {
            FC_C: for(int c = 0; c < 32; c++) {
                #pragma HLS PIPELINE II=1
                int        idx = (y * 6 + x) * 32 + c;
                ap_int<16> px  = pool3[y][x][c];
                ap_int<8>  w   = fc_w[n * 1152 + idx];
                fc_out[n]     += (ap_int<32>)px * (ap_int<32>)w;
            }
        }
    }
    if(fc_out[n] < 0) fc_out[n] = 0;  // ReLU
}
```

**What the FC layer does:** Every one of the 64 neurons sees every one of the 1152 pool3 values. It's a dense matrix-vector multiplication: `fc_out = fc_w (64×1152) × pool3_flat (1152) + fc_b (64)`.

**`FC_INIT` with `UNROLL`:** All 64 bias initializations happen in 1 clock cycle — each `fc_out[n]` is a separate register (from `ARRAY_PARTITION complete`), so all 64 can be written simultaneously.

**Flattening index `(y * 6 + x) * 32 + c`:** Converts the 3D pool3 coordinates `(y, x, c)` into a 1D index `0..1151`. This is how PyTorch's `view(-1)` or NumPy's `flatten()` works, just done explicitly in hardware.

**`ReLU after FC`:** Applied per-neuron after the full accumulation.

---

## Section 11 — Class Scores and ArgMax

```cpp
scores[0] = 0;  // Cat
scores[1] = 0;  // Dog

// Cat score = sum of first 32 FC neurons
CAT_SCORE: for(int n = 0;  n < 32; n++) {
    #pragma HLS PIPELINE II=1;
    scores[0] += fc_out[n];
}

// Dog score = sum of last 32 FC neurons
DOG_SCORE: for(int n = 32; n < 64; n++) {
    #pragma HLS PIPELINE II=1;
    scores[1] += fc_out[n];
}

// ArgMax: whichever score is higher wins
int predicted_class = (scores[1] > scores[0]) ? 1 : 0;
```

**How this classifies:** The 64 FC neurons are split into two halves. The first 32 neurons represent "Cat energy" and the last 32 represent "Dog energy". Whichever sum is larger is the predicted class. This is a simple linear classifier on top of the learned features.

**Why this works instead of a full softmax output layer:** For a 2-class problem this is equivalent — the class with the larger raw score is the same class that would have the larger softmax probability. No exponential function needed in hardware.

---

## Section 12 — Bounding Box Detection

```cpp
int        bbox_x = 32, bbox_y = 32;  // default: image center
int        bbox_w = 20, bbox_h = 20;
ap_int<32> max_act = 0;

BBOX_Y: for(int y = 0; y < 6; y++) {
    BBOX_X: for(int x = 0; x < 6; x++) {
        #pragma HLS PIPELINE II=1
        ap_int<32> act = 0;

        BBOX_C: for(int c = 0; c < 32; c++) {
            act += pool3[y][x][c];      // sum all 32 channels at this spatial position
        }

        if(act > max_act) {
            max_act = act;
            bbox_x  = x * 10 + 5;      // map 6×6 → 64×64 image space
            bbox_y  = y * 10 + 5;
        }
    }
}
```

**What this does:** After Pool3 we have a 6×6 spatial map of 32-channel activations. The spatial position with the highest total activation across all 32 channels is likely where the cat/dog is in the image. This is a lightweight alternative to a trained detection head.

**Coordinate mapping `x * 10 + 5`:** The pool3 grid covers 6 cells. Each cell corresponds to a `64/6 ≈ 10.67` pixel region. Multiplying by 10 and adding 5 maps the cell center back to approximately the correct 64×64 pixel coordinate.

---

## Section 13 — Write Results to DDR

```cpp
result[0] = predicted_class;   // 0=Cat, 1=Dog
result[1] = bbox_x;            // X center
result[2] = bbox_y;            // Y center
result[3] = bbox_w;            // Box width
result[4] = bbox_h;            // Box height
result[5] = (int)max_act;      // Confidence (max pool3 activation)
result[6] = (int)scores[0];    // Raw Cat score
result[7] = (int)scores[1];    // Raw Dog score
result[8] = MAGIC_NUMBER;      // 0xC0FFEE00 — written last as completion signal
```

**Why `MAGIC_NUMBER` is written last:** The ARM polls `AP_DONE` to know the accelerator is done, but `result[8]` provides an additional layer of verification. If inference completed correctly all the way through, this value is `0xC0FFEE00`. If the accelerator hung or crashed mid-way, this value would still be zero (or garbage from a previous run). The Python driver asserts this value before trusting any other result.
