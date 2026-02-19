## 📌 Project Overview

This project implements a 2 class image classifier accelerated on FPGA (PYNQ) and compares its performance against a CPU-based NumPy implementation. The pipeline demonstrates:

- End-to-end FPGA inference using a custom CNN hardware accelerator

- Dynamic weight loading for different trained models

- Latency and FPS benchmarking between FPGA and CPU

- Foreground-based tight bounding box visualization

- Convolution feature map visualization for model interpretability

- A combined visualization panel for qualitative and quantitative analysis

The final output is a single composite image showing:

- Detection result

- Confidence scores

- FPGA vs CPU performance

- Intermediate convolution activations

## 🧠 System Architecture

The system consists of three major components:

#### 1. FPGA Inference Engine (PYNQ Overlay)
A custom CNN accelerator synthesized into FPGA logic and loaded using a PYNQ overlay (real_detect.bit). The accelerator performs convolution, pooling, and classification directly in hardware.

#### 2. CPU Reference Implementation (NumPy)
- A software version of the same CNN architecture is implemented using NumPy. This serves as:

- A correctness reference

- A performance baseline for benchmarking

#### 3. Visualization & Benchmarking Pipeline
Post-processing code visualizes:

- Classification results

- Foreground bounding boxes

- Confidence bars

- Latency/FPS comparison

- Convolution feature maps for interpretability

## ⚙️ FPGA Inference Workflow

#### 1. Overlay Loading
The FPGA bitstream (real_detect.bit) is loaded onto the PYNQ board using the Overlay API.

#### 2. DMA Buffer Allocation
Physically contiguous memory buffers are allocated for:

- Input image

- CNN weights and biases

- Output results

#### 3. Weight Loading
Pre-trained CNN weights stored as .npy files are copied into FPGA-accessible buffers. The design supports multiple weight sets; in this version, the best-performing set is used.

#### 4. Image Preprocessing
The input image is:

- Converted to grayscale

- Resized to 64 × 64

- Flattened and transferred to the FPGA buffer

#### 5. Hardware Execution
The CPU triggers FPGA execution through memory-mapped control registers. The accelerator processes the image and writes classification scores back to memory.

#### 6. Result Retrieval
The predicted class, raw scores, and confidence margin are read back from the FPGA.

## 💻 CPU Reference Inference

A full forward pass of the same CNN is implemented in NumPy:

- Convolution → ReLU → MaxPooling (3 layers)

- Fully connected layer

- Output logits

This implementation is used to:

- Validate FPGA inference correctness

- Measure CPU latency and FPS

- Quantify hardware speedup

## 🖼️ Visualization Pipeline
#### 1. Foreground-Based Tight Bounding Box

Foreground segmentation (Otsu thresholding + morphology) is used to estimate a tight bounding box around the animal. This improves visual clarity compared to using a full-frame box.

#### 2. Detection Panel

For each inference, the visualization panel shows:

- Predicted label

- Color-coded bounding box

- Confidence bar 

- FPGA latency and FPS

- CPU latency and FPS

- Relative speedup and prediction match/mismatch

#### 3. Convolution Feature Maps

Intermediate feature maps from Conv1, Conv2, and Conv3 are visualized to illustrate what the CNN learns at different stages (edges → textures → higher-level patterns).

#### 4. Combined Figure

All detection panels and convolution strips are assembled into a single composite figure and saved to disk for reporting and documentation.

## 📊 Performance Benchmarking

The script reports:

- FPGA latency (ms) and FPS

- CPU latency (ms) and FPS

- Relative speedup factor

- Prediction consistency between FPGA and CPU

This provides a quantitative demonstration of the benefits of hardware acceleration for CNN inference.

## 🎯 Key Takeaways

- Demonstrates a complete hardware-accelerated CNN inference pipeline on FPGA

- Shows how to integrate Python, PYNQ, DMA buffers, and custom hardware IP

- Provides both quantitative benchmarking and qualitative visualization

- Highlights the practical speedup achievable with FPGA-based acceleration compared to CPU execution
