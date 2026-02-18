# System Architecture

## Overview
Description of overall ARM–FPGA co-design on PYNQ-Z2.

## Hardware–Software Partitioning
### ARM Processor
- Image input  
- Preprocessing  
- Control logic  
- Post-processing  

### FPGA Fabric
- Convolution  
- Activation  
- Pooling  

## Data Flow
Input → ARM → FPGA → ARM → Output

## Interfaces
- AXI Lite (control)  
- AXI DMA / AXI Stream (data)

## Block Diagram
(Insert architecture diagram)
