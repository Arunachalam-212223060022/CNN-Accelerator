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
    ap_int<8>    *conv1_w,
    ap_int<8>    *conv1_b,
    ap_int<8>    *conv2_w,
    ap_int<8>    *conv2_b,
    ap_int<8>    *conv3_w,
    ap_int<8>    *conv3_b,
    ap_int<8>    *fc_w,
    ap_int<8>    *fc_b,
    volatile int *result
) {

// ============================================================
// AXI INTERFACE PRAGMAS
// ============================================================

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
#pragma HLS INTERFACE s_axilite port=image   bundle=control
#pragma HLS INTERFACE s_axilite port=conv1_w bundle=control
#pragma HLS INTERFACE s_axilite port=conv1_b bundle=control
#pragma HLS INTERFACE s_axilite port=conv2_w bundle=control
#pragma HLS INTERFACE s_axilite port=conv2_b bundle=control
#pragma HLS INTERFACE s_axilite port=conv3_w bundle=control
#pragma HLS INTERFACE s_axilite port=conv3_b bundle=control
#pragma HLS INTERFACE s_axilite port=fc_w    bundle=control
#pragma HLS INTERFACE s_axilite port=fc_b    bundle=control
#pragma HLS INTERFACE s_axilite port=result  bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

// ============================================================
// FEATURE MAP BUFFERS  (BRAM-backed to save LUT-RAM)
// ============================================================

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

// ============================================================
// STEP 1: LOAD IMAGE FROM DDR
// ============================================================

LOAD_Y: for(int y = 0; y < 64; y++) {
    LOAD_X: for(int x = 0; x < 64; x++) {
        #pragma HLS PIPELINE II=1
        img_buf[y][x] = image[y * 64 + x];
    }
}

// ============================================================
// STEP 2: CONV LAYER 1  — 64×64×1 → 62×62×8
//   kernel: 3×3, 8 filters, stride=1, no padding
//   accumulate INT8×INT8 into INT32, ReLU, scale → INT16
// ============================================================

CONV1_Y: for(int y = 0; y < 62; y++) {
    CONV1_X: for(int x = 0; x < 62; x++) {
        CONV1_F: for(int f = 0; f < 8; f++) {
            #pragma HLS PIPELINE II=1

            ap_int<32> acc = conv1_b[f];  // start with bias

            CONV1_KY: for(int ky = 0; ky < 3; ky++) {
                CONV1_KX: for(int kx = 0; kx < 3; kx++) {
                    ap_uint<8> px  = img_buf[y + ky][x + kx];
                    ap_int<8>  w   = conv1_w[f * 9 + ky * 3 + kx];
                    acc += (ap_int<32>)px * (ap_int<32>)w;
                }
            }

            // ReLU + scale down
            if(acc < 0) acc = 0;
            conv1[y][x][f] = (ap_int<16>)(acc >> 4);
        }
    }
}

// ============================================================
// STEP 3: MAX POOL 1  — 62×62×8 → 31×31×8
// ============================================================

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

// ============================================================
// STEP 4: CONV LAYER 2  — 31×31×8 → 29×29×16
// ============================================================

CONV2_Y: for(int y = 0; y < 29; y++) {
    CONV2_X: for(int x = 0; x < 29; x++) {
        CONV2_F: for(int f = 0; f < 16; f++) {
            #pragma HLS PIPELINE II=1

            ap_int<32> acc = conv2_b[f];

            CONV2_C: for(int c = 0; c < 8; c++) {
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

// ============================================================
// STEP 5: MAX POOL 2  — 29×29×16 → 14×14×16
// ============================================================

POOL2_Y: for(int y = 0; y < 14; y++) {
    POOL2_X: for(int x = 0; x < 14; x++) {
        POOL2_F: for(int f = 0; f < 16; f++) {
            #pragma HLS PIPELINE II=1

            ap_int<16> max_val = 0;
            POOL2_KY: for(int ky = 0; ky < 2; ky++) {
                POOL2_KX: for(int kx = 0; kx < 2; kx++) {
                    if(y*2+ky < 29 && x*2+kx < 29) {
                        ap_int<16> v = conv2[y*2+ky][x*2+kx][f];
                        if(v > max_val) max_val = v;
                    }
                }
            }
            pool2[y][x][f] = max_val;
        }
    }
}

// ============================================================
// STEP 6: CONV LAYER 3  — 14×14×16 → 12×12×32
// ============================================================

CONV3_Y: for(int y = 0; y < 12; y++) {
    CONV3_X: for(int x = 0; x < 12; x++) {
        CONV3_F: for(int f = 0; f < 32; f++) {
            #pragma HLS PIPELINE II=1

            ap_int<32> acc = conv3_b[f];

            CONV3_C: for(int c = 0; c < 16; c++) {
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

// ============================================================
// STEP 7: MAX POOL 3  — 12×12×32 → 6×6×32
// ============================================================

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

// ============================================================
// STEP 8: FULLY CONNECTED  — 1152 → 64
//   Flatten pool3 (6×6×32=1152) → 64 neurons + ReLU
// ============================================================

FC_INIT: for(int n = 0; n < 64; n++) {
    #pragma HLS UNROLL
    fc_out[n] = fc_b[n];
}

FC_N: for(int n = 0; n < 64; n++) {
    FC_Y: for(int y = 0; y < 6; y++) {
        FC_X: for(int x = 0; x < 6; x++) {
            FC_C: for(int c = 0; c < 32; c++) {
                #pragma HLS PIPELINE II=1
                int     idx    = (y * 6 + x) * 32 + c;
                ap_int<16> px  = pool3[y][x][c];
                ap_int<8>  w   = fc_w[n * 1152 + idx];
                fc_out[n]     += (ap_int<32>)px * (ap_int<32>)w;
            }
        }
    }
    if(fc_out[n] < 0) fc_out[n] = 0;  // ReLU
}

// ============================================================
// STEP 9: OUTPUT — Class Scores & ArgMax
//   Cat score = sum of FC neurons [0..31]
//   Dog score = sum of FC neurons [32..63]
// ============================================================

scores[0] = 0;  // Cat
scores[1] = 0;  // Dog

CAT_SCORE: for(int n = 0;  n < 32; n++) { #pragma HLS PIPELINE II=1; scores[0] += fc_out[n]; }
DOG_SCORE: for(int n = 32; n < 64; n++) { #pragma HLS PIPELINE II=1; scores[1] += fc_out[n]; }

int predicted_class = (scores[1] > scores[0]) ? 1 : 0;  // 0=Cat, 1=Dog

// ============================================================
// STEP 10: BOUNDING BOX — Activation Hotspot from Pool3
//   Scan 6×6 feature map, find max activation cell,
//   map back to 64×64 image coordinate space.
// ============================================================

int          bbox_x = 32, bbox_y = 32;
int          bbox_w = 20, bbox_h = 20;
ap_int<32>   max_act = 0;

BBOX_Y: for(int y = 0; y < 6; y++) {
    BBOX_X: for(int x = 0; x < 6; x++) {
        #pragma HLS PIPELINE II=1
        ap_int<32> act = 0;
        BBOX_C: for(int c = 0; c < 32; c++) {
            act += pool3[y][x][c];
        }
        if(act > max_act) {
            max_act = act;
            bbox_x  = x * 10 + 5;  // map 6×6 → 64×64
            bbox_y  = y * 10 + 5;
        }
    }
}

// ============================================================
// STEP 11: WRITE RESULTS TO DDR
// ============================================================

result[0] = predicted_class;     // 0=Cat, 1=Dog
result[1] = bbox_x;              // X center
result[2] = bbox_y;              // Y center
result[3] = bbox_w;              // Box width
result[4] = bbox_h;              // Box height
result[5] = (int)max_act;        // Confidence
result[6] = (int)scores[0];      // Cat score
result[7] = (int)scores[1];      // Dog score
result[8] = MAGIC_NUMBER;        // 0xC0FFEE00 — sanity check

}
