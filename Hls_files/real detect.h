# `hls/cnn_detect.h` — Header File

## What This File Does

This is the **header file** for the CNN accelerator. It serves three purposes:

1. **Declares the top-level HLS function** — so the testbench and other files can call it
2. **Defines every weight array size as a macro** — so the exact same numbers are used in the C++ source, testbench, and Python driver without copy-paste errors
3. **Documents the result buffer layout** — explains what each `result[i]` index means so the Python driver and hardware are always in sync

This file is `#include`-d by both `cnn_detect.cpp` and the testbench.

---

## Full File

```cpp
#ifndef CNN_DETECT_H
#define CNN_DETECT_H

#include <ap_int.h>

// ============================================================
// CNN Accelerator — Cat vs Dog Classifier
// PYNQ-Z2 (XC7Z020) | Vitis HLS 2023.1
// Input : 64×64 grayscale (INT8)
// Output: 2-class (0=Cat, 1=Dog)
// ============================================================

void cnn_detect(
    ap_uint<8>   *image,     // Input image  [4096 bytes]
    ap_int<8>    *conv1_w,   // Conv1 weights [8×1×9   = 72]
    ap_int<8>    *conv1_b,   // Conv1 bias    [8]
    ap_int<8>    *conv2_w,   // Conv2 weights [16×8×9  = 1152]
    ap_int<8>    *conv2_b,   // Conv2 bias    [16]
    ap_int<8>    *conv3_w,   // Conv3 weights [32×16×9 = 4608]
    ap_int<8>    *conv3_b,   // Conv3 bias    [32]
    ap_int<8>    *fc_w,      // FC weights    [64×1152 = 73728]
    ap_int<8>    *fc_b,      // FC bias       [64]
    volatile int *result     // Output buffer [9 × int32]
);

// ============================================================
// Weight Sizes
// ============================================================

#define CONV1_W_SIZE   (8  * 1  * 3 * 3)   //    72
#define CONV1_B_SIZE   (8)                  //     8

#define CONV2_W_SIZE   (16 * 8  * 3 * 3)   //  1152
#define CONV2_B_SIZE   (16)                 //    16

#define CONV3_W_SIZE   (32 * 16 * 3 * 3)   //  4608
#define CONV3_B_SIZE   (32)                 //    32

#define FC_W_SIZE      (64 * 6 * 6 * 32)   // 73728
#define FC_B_SIZE      (64)                 //    64

#define TOTAL_WEIGHTS  (CONV1_W_SIZE + CONV2_W_SIZE + CONV3_W_SIZE + FC_W_SIZE)  // 79560
#define TOTAL_BIAS     (CONV1_B_SIZE + CONV2_B_SIZE + CONV3_B_SIZE + FC_B_SIZE)  //   120

// ============================================================
// Output Result Indices  (result[0..8])
// ============================================================
// result[0]  — predicted class  : 0 = Cat, 1 = Dog
// result[1]  — bbox x_center    : 0–63 (64×64 image space)
// result[2]  — bbox y_center    : 0–63
// result[3]  — bbox width
// result[4]  — bbox height
// result[5]  — confidence       : max pool3 activation sum
// result[6]  — cat score        : sum of FC neurons [0..31]
// result[7]  — dog score        : sum of FC neurons [32..63]
// result[8]  — magic number     : 0xC0FFEE00 (sanity check)

#define MAGIC_NUMBER   0xC0FFEE00

#endif // CNN_DETECT_H
```

---

## Line-by-Line Explanation

### Include Guard

```cpp
#ifndef CNN_DETECT_H
#define CNN_DETECT_H
```

Prevents the header from being included more than once in the same compilation unit. Standard C/C++ practice.

---

### `ap_int.h` Include

```cpp
#include <ap_int.h>
```

This is a **Vitis HLS specific header**. It gives access to arbitrary-precision integer types like:

| Type | Meaning |
|------|---------|
| `ap_uint<8>` | Unsigned 8-bit integer (0–255) — used for pixel values |
| `ap_int<8>`  | Signed 8-bit integer (−128 to 127) — used for INT8 weights |
| `ap_int<16>` | Signed 16-bit — used for intermediate feature maps |
| `ap_int<32>` | Signed 32-bit — used for accumulators and scores |

These types tell HLS exactly how wide each hardware register/wire should be, giving fine-grained control over FPGA resource usage.

---

### Function Declaration

```cpp
void cnn_detect(
    ap_uint<8>   *image,
    ap_int<8>    *conv1_w,
    ap_int<8>    *conv1_b,
    ...
    volatile int *result
);
```

Each argument becomes an **AXI port** in hardware. The types matter:

- `ap_uint<8> *image` — pixel values are unsigned (0–255), so `uint8`
- `ap_int<8> *conv1_w` — weights are signed INT8 (−128 to 127 after quantization)
- `volatile int *result` — `volatile` tells the compiler not to cache this pointer; it's a memory-mapped output that hardware writes and ARM reads

---

### Weight Size Macros

```cpp
#define CONV1_W_SIZE   (8  * 1  * 3 * 3)   // 72
```

Breaking down `8 * 1 * 3 * 3`:
- `8` = number of Conv1 filters
- `1` = input channels (grayscale = 1 channel)
- `3 * 3` = kernel size (3×3 spatial)

```cpp
#define CONV2_W_SIZE   (16 * 8  * 3 * 3)   // 1152
```

- `16` = Conv2 output filters
- `8`  = Conv1 output channels (Conv2's input channels)
- `3 * 3` = kernel

```cpp
#define FC_W_SIZE      (64 * 6 * 6 * 32)   // 73728
```

- `64` = number of FC neurons
- `6 * 6 * 32` = pool3 output shape flattened (1152 values per sample)

These macros are used in both the testbench (to allocate buffers) and Python (to allocate DDR buffers).

---

### Result Index Documentation

```cpp
// result[0]  — predicted class  : 0 = Cat, 1 = Dog
// result[1]  — bbox x_center
// ...
// result[8]  — magic number     : 0xC0FFEE00
```

The accelerator writes 9 integers to DDR. This comment block is the **contract** between hardware and software — the Python driver reads these in the exact same order.

```cpp
#define MAGIC_NUMBER   0xC0FFEE00
```

`0xC0FFEE00` ("coffee") is written to `result[8]` at the very end of the accelerator. The Python driver checks this value after inference — if it's wrong, the accelerator either crashed or never completed. It's a cheap hardware sanity check.
