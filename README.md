# FPGA-Based Hardware Accelerator for Large Language Models (LLMs) Inference

> A senior design project focused on designing and implementing a hardware accelerator optimized for LLM inference, specifically targeting the Qwen 2.5-0.5B architecture.

**Advisor:** Dr. Hassan Salmani

---

## 👥 Team Members

- **Aayush Jha**
- **Jeff Mofor Allo**
- **Chase Adams**
- **Bishesh Adhikari**

---

## 📖 Project Overview

This project aims to design and implement a hardware accelerator optimized for Large Language Models (LLMs), specifically targeting the **Qwen 2.5-0.5B** architecture. By leveraging the parallel processing capabilities of Field-Programmable Gate Arrays (FPGAs), we aim to improve energy efficiency and reduce latency for key inference operations compared to general-purpose CPUs.

### Development Approach

The project follows a **bottom-up approach**:

1. **✅ Software Prototyping** - Complete re-implementation of the LLM inference pipeline (Tokenizer → Embedding → Attention → Decode) in "barebones" Python to establish a Golden Model
2. **🚧 RTL Design** - Translating these logical blocks into synthesizable Verilog/SystemVerilog
3. **⏳ FPGA Deployment** - Mapping the design to the AUP-ZU3 (Zynq UltraScale+) platform
4. **⏳ System Integration** - Building an end-to-end demo with Keyboard Input and VGA/HDMI Output

---

## 🏗️ System Architecture

The accelerator implements a standard **Transformer Decoder inference pipeline**, split into two distinct execution stages:

### 1. Prefill Stage (Compute Bound)

- **Input:** Full user prompt (sequence of tokens)
- **Operation:** Parallel processing of all input tokens to generate the initial KV Cache and the first new token
- **Key Modules:**
  - Tokenizer (Host)
  - Embedding Lookup
  - Parallel Matrix Multiplication (GEMM)
  - Multi-head Self-Attention with GQA
  - Feed-Forward Network (SwiGLU)
  - Softmax

### 2. Decode Stage (Memory Bound)

- **Input:** Single generated token
- **Operation:** Auto-regressive generation of one token at a time using the cached Key/Value matrices
- **Key Modules:**
  - KV Cache Manager
  - Vector-Matrix Multiplication
  - Attention with cached K/V
  - Sampling (Greedy/Temperature-based)

---

## 📂 Repository Structure

```
├── hardware/               # (Future) Verilog/VHDL RTL source code
│   ├── src/                # Core modules (Attention, FFN, Softmax)
│   ├── sim/                # Testbenches and simulation waveforms
│   └── synthesis/          # Vivado/Quartus project files
│
├── software/               # Python Golden Models & Host Control
│   ├── tokenizer.py        # BPE Tokenizer implementation (encode/decode)
│   ├── embedding.py        # Weight extraction and embedding generation
│   ├── self_attention.py   # Multi-head self-attention with GQA + FFN
│   ├── decode.py           # Complete generation pipeline (Prefill + Decode)
│   └── test_qwen_model.py  # Validation against HuggingFace implementation
│
├── Qwen2.5-0.5B/          # Model files (config.json, model.safetensors, etc.)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## 🛠️ Software Prototype (Golden Model) ✅

The complete software implementation serves as a **Golden Model** for hardware verification. All components are implemented using "barebones" Python that mimics FPGA hardware data flow.

### 1. Tokenizer (`tokenizer.py`)

Implements the **Byte Pair Encoding (BPE)** algorithm from scratch.

- **Input:** Raw text string (e.g., `"Hello world"`)
- **Data Sources:**
  - `merges.txt` - Merge priority rules
  - `vocab.json` - Token ID mapping
- **Output:** Integer list `[101, 204, ...]` ready for the hardware embedding layer

**Features:**
- Full BPE implementation without external dependencies
- Handles Qwen-specific tokenization rules
- Includes encode/decode functionality

### 2. Embedding Lookup (`embedding.py`)

Simulates the hardware memory read operation for the first layer.

- **Input:** Token IDs
- **Operation:** Fetches high-dimensional vectors (d=896) from the `.safetensors` weight file
- **Output:** Matrix `[Seq_Len, Hidden_Dim]` passed to the Attention Mechanism

**Features:**
- Direct weight extraction from safetensors format
- Configurable model directory paths
- Dimension verification against config.json

### 3. Self-Attention (`self_attention.py`)

Implements a complete transformer layer with multi-head self-attention and feed-forward network using barebone FPGA-oriented operations.

- **Input:** Embedding vectors `[Seq_Len, Hidden_Dim]`
- **Architecture:** 
  - 14 Query attention heads
  - 2 Key-Value heads (GQA for efficiency)
  - 64-dimensional head size
  - 24 transformer layers (configurable)
- **Operations:**
  - RMS Normalization (pre-attention and pre-FFN)
  - Q, K, V projections with bias support
  - Rotary Position Embeddings (RoPE)
  - Scaled dot-product attention with causal masking
  - Multi-head output concatenation
  - Feed-Forward Network (SwiGLU activation)
  - Residual connections
  - KV Cache management for decode stage

**FPGA Hardware Mapping:**
- **MAC Units** - Vector dot products
- **Systolic Arrays** - Matrix-vector multiplication
- **Softmax Units** - Attention probability computation
- **Parallel Adders/Multipliers** - Element-wise operations
- **CORDIC/LUT** - Trigonometric functions for RoPE

### 4. Complete Generation Pipeline (`decode.py`)

Implements the full end-to-end text generation pipeline with both Prefill and Decode stages.

**Pipeline Flow:**
```
Text Input → Tokenization → Embedding → [24 Transformer Layers] → Final Norm → LM Head → Sampling → Generated Token
                                                                                                        ↓
                                    Decode Loop: Embed Token → [24 Layers with KV Cache] → Sample → Next Token
```

**Features:**
- **Prefill Stage:** Processes entire prompt sequence, populates KV cache
- **Decode Stage:** Auto-regressive generation using cached K/V values
- **Sampling:** Supports both greedy decoding and temperature-based sampling
- **EOS Detection:** Automatically stops on end-of-sequence token
- **Multi-layer Support:** Configurable number of transformer layers

**Key Classes:**
- `DecodeSelfAttentionLayer`: Extends `SelfAttentionLayer` with `forward_decode()` method for single-token processing
- `generate()`: Main generation function that orchestrates the full pipeline

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **torch** (for tensor operations and model loading)
- **safetensors** (for loading model weights)
- **numpy** (for array operations)
- **transformers** (optional, for validation with `test_qwen_model.py`)

### Installation

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch safetensors numpy transformers
```

### Usage

#### Running Complete Text Generation (Recommended)

The main entry point for text generation is `decode.py`:

```bash
cd software
python decode.py
```

This will prompt you for input text and generate tokens using the complete pipeline.

**Example:**
```bash
$ python decode.py
Enter a prompt: Hello, how are you?
=== Tokenization ===
[9906, 11, 527, 499, 366, 30]
====

=== Embedding ===
[[0.123, -0.456, ...], ...]
====

=== Result ===
Input text:  Hello, how are you?
Generated text:  Hello, how are you? I'm doing well, thank you!
====
```

#### Running Individual Components

You can also test individual components:

**Tokenizer:**
```bash
python tokenizer.py
```

**Embedding:**
```bash
python embedding.py
```

**Self-Attention:**
```bash
python self_attention.py
```

#### Validation Against HuggingFace

To compare outputs with the reference HuggingFace implementation:

```bash
python test_qwen_model.py
```

This script loads the model using HuggingFace transformers and can save intermediate values for comparison.

---

## 🔧 Implementation Details

### Hardware-Oriented Design

All software components are designed to mirror FPGA hardware operations:

- **Vector Operations:** Element-wise operations that map to parallel hardware units
- **Matrix-Vector Multiplication:** Systolic array architecture
- **Softmax:** Tree reduction for max-finding, LUT/CORDIC for exponentials
- **KV Cache:** Memory-mapped storage structure for efficient decode stage
- **RoPE:** Trigonometric computations suitable for hardware implementation

### Key Algorithms

1. **BPE Tokenization:** Greedy merge algorithm with priority-based ranking
2. **Grouped Query Attention (GQA):** Reduces KV cache size by sharing K/V heads across multiple Q heads
3. **Rotary Position Embeddings:** Split-half pairing strategy for efficient hardware implementation
4. **SwiGLU Activation:** SiLU-gated linear unit for feed-forward network
5. **RMS Normalization:** Root mean square normalization with learned scaling

---

## 📅 Project Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 1** | Research & Specs | ✅ Complete |
| **Phase 2** | Software Golden Model | ✅ Complete |
| **Phase 3** | RTL Design | 🚧 Next |
| **Phase 4** | Compute Implementation | ⏳ Planned |
| **Phase 5** | Pipeline Integration | ⏳ Planned |
| **Phase 6** | System I/O | ⏳ Planned |
| **Phase 7** | Verification & Benchmarking | ⏳ Planned |

### Phase 1 & 2 Completion Summary

✅ **Completed Components:**
- BPE Tokenizer with encode/decode
- Embedding layer with safetensors support
- Multi-head self-attention with GQA
- Rotary Position Embeddings (RoPE)
- Feed-Forward Network (SwiGLU)
- RMS Normalization
- KV Cache management
- Complete Prefill stage
- Complete Decode stage
- End-to-end text generation pipeline
- Validation against HuggingFace implementation

---


---

## 📝 License

This project is part of a senior design course. Please refer to the model license in `Qwen2.5-0.5B/LICENSE` for model-specific licensing information.

---

## 🤝 Contributing

This is a senior design project. For questions or collaboration, please contact the team members or advisor.

---

**Status:** ✅ Software Implementation Complete - Ready for RTL Design

**Latest Progress:**
- ✅ Complete tokenization pipeline (BPE)
- ✅ Complete embedding layer
- ✅ Complete transformer layer (Attention + FFN)
- ✅ Complete prefill stage implementation
- ✅ Complete decode stage implementation
- ✅ End-to-end text generation working
- ✅ Validation against HuggingFace complete
- 🚧 Next: Begin RTL design and hardware implementation
