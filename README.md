# FPGA-Based Hardware Accelerator for Large Language Models (LLMs)

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

1. **Software Prototyping** - Re-implementing the LLM inference pipeline (Tokenizer → Embedding → Attention) in "barebones" Python to establish a Golden Model
2. **RTL Design** - Translating these logical blocks into synthesizable Verilog/SystemVerilog
3. **FPGA Deployment** - Mapping the design to the AUP-ZU3 (Zynq UltraScale+) platform
4. **System Integration** - Building an end-to-end demo with Keyboard Input and VGA/HDMI Output

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
  - Softmax

### 2. Decode Stage (Memory Bound)

- **Input:** Single generated token
- **Operation:** Auto-regressive generation of one token at a time using the cached Key/Value matrices
- **Key Modules:**
  - KV Cache Manager
  - Vector-Matrix Multiplication
  - Sampling (Top-K/Top-P)

---

## 📂 Repository Structure

```
├── hardware/               # (Future) Verilog/VHDL RTL source code
│   ├── src/                # Core modules (Attention, FFN, Softmax)
│   ├── sim/                # Testbenches and simulation waveforms
│   └── synthesis/          # Vivado/Quartus project files
│
├── software/               # Python Golden Models & Host Control
│   ├── tokenizer.py        # BPE Tokenizer implementation (Logic verification)
│   ├── embedding.py        # Weight extraction and embedding generation
│   └── inference.py        # Main loop (Prefill + Decode logic)
│
├── docs/                   # Design specifications and timeline
├── Qwen2.5-0.5B/          # Model files (config.json, model.safetensors, etc.)
└── README.md              # This file
```

---

## 🛠️ Software Prototype (Golden Model)

Before moving to hardware, we are validating the logic using "barebones" Python scripts that mimic the hardware data flow.

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
- **Operation:** Fetches high-dimensional vectors (d=896 or 1024) from the `.safetensors` weight file
- **Output:** Matrix E⁰ passed to the Attention Mechanism

**Features:**
- Direct weight extraction from safetensors format
- Configurable model directory paths
- Dimension verification against config.json

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **torch** (for tensor verification)
- **safetensors** (for loading model weights)

### Installation

```bash
pip install torch safetensors
```

### Usage

#### Running the Tokenizer Prototype

```bash
python tokenizer.py
```

This will load the vocabulary and convert sample text into token IDs.

**Example:**
```bash
$ python tokenizer.py
Enter a string: Hello, how are you?
Input String: 'Hello, how are you?'
Output Tokens (IDs): [9906, 11, 527, 499, 366, 30]
Number of tokens: 6
```

#### Running the Embedding Lookup

```bash
python embedding.py
```

This demonstrates how integer IDs are converted into floating-point vectors using simulated weights.

**Example:**
```bash
$ python embedding.py
Enter a string to tokenize and embed: Hello world
Input String: 'Hello world'
Token IDs: [9906, 1917]
Number of tokens: 2
Shape: (2, 896)
(Sequence Length: 2, Hidden Dimension: 896)
```

#### Full Pipeline

The embedding script automatically runs the full pipeline:
1. Text input → Tokenization
2. Token IDs → Embedding vectors
3. Ready for accelerator's Prefill stage

---

## 📅 Project Roadmap

| Phase | Focus | Key Deliverables |
|-------|-------|------------------|
| **Phase 1** | Research & Specs | Architecture definition, Toolchain setup, Python Golden Model |
| **Phase 2** | RTL Design | Verilog stubs for Embedding, QKV Projections, and Attention Core |
| **Phase 3** | Compute Implementation | Systolic Array/MAC units for Matrix Multiplication, Softmax Hardware |
| **Phase 4** | Pipeline Integration | Connecting Prefill & Decode stages, Implementing KV Cache logic |
| **Phase 5** | System I/O | Keyboard Controller (Input) and Display Driver (Output) integration |
| **Phase 6** | Verification | Co-simulation (FPGA vs. Python) and Performance Benchmarking |

---

## 📚 References

- [Qwen 2.5 Technical Report](https://github.com/QwenLM/Qwen2.5)
- [Attention Is All You Need (Vaswani et al.)](https://arxiv.org/abs/1706.03762)
- Understanding LLM Prefill vs Decode

---

## 📝 License

This project is part of a senior design course. Please refer to the model license in `Qwen2.5-0.5B/LICENSE` for model-specific licensing information.

---

## 🤝 Contributing

This is a senior design project. For questions or collaboration, please contact the team members or advisor.

---

**Status:** 🚧 In Active Development - Phase 1 (Software Prototyping)
