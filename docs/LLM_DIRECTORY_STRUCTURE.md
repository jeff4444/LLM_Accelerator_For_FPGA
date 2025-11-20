# LLM Model Directory Structure and File Usage

This document describes the structure of the Qwen2.5-0.5B model directory, the contents of each file, what information is parsed from them, and which stage of the inference pipeline uses each file.

---

## 📁 Directory Overview

The `Qwen2.5-0.5B/` directory contains all necessary files for running inference with the Qwen 2.5-0.5B model. The directory structure is as follows:

```
Qwen2.5-0.5B/
├── config.json              # Model architecture configuration
├── generation_config.json   # Generation hyperparameters
├── merges.txt              # BPE merge rules for tokenization
├── model.safetensors        # Model weights (all layers)
├── tokenizer_config.json   # Tokenizer configuration (HuggingFace format)
├── tokenizer.json          # Full tokenizer state (HuggingFace format)
├── vocab.json              # Token vocabulary mapping
├── LICENSE                 # Model license
└── README.md               # Model documentation
```

---

## 📄 File Descriptions and Usage

### 1. `config.json`

**Purpose:** Contains the complete model architecture configuration and hyperparameters.

**Format:** JSON file

**Key Fields Parsed:**
- `vocab_size`: 151936 - Total number of tokens in vocabulary
- `hidden_size`: 896 - Dimension of hidden states
- `num_hidden_layers`: 24 - Number of transformer layers
- `num_attention_heads`: 14 - Number of query attention heads
- `num_key_value_heads`: 2 - Number of key-value heads (GQA)
- `head_dim`: 64 - Calculated as `hidden_size / num_attention_heads`
- `intermediate_size`: 4864 - FFN hidden dimension
- `rms_norm_eps`: 1e-6 - Epsilon for RMS normalization
- `rope_theta`: 1000000.0 - Base frequency for Rotary Position Embeddings
- `eos_token_id`: 151643 - End-of-sequence token ID
- `bos_token_id`: 151643 - Beginning-of-sequence token ID
- `max_position_embeddings`: 32768 - Maximum sequence length

**Used By:**
- **`embedding.py`** - Initialization stage
  - Extracts `vocab_size` and `hidden_size` to verify embedding weight dimensions
  
- **`self_attention.py`** - Initialization stage
  - Extracts all architecture parameters to configure layer dimensions
  - Used to determine weight matrix shapes and attention head configuration
  
- **`decode.py`** - Initialization stage
  - Extracts `num_hidden_layers` to determine how many transformer layers to instantiate
  - Extracts `eos_token_id` to detect end-of-sequence during generation

**Pipeline Stage:** Initialization (before inference begins)

---

### 2. `vocab.json`

**Purpose:** Maps string tokens to integer token IDs for tokenization.

**Format:** JSON file with string keys and integer values

**Structure:**
```json
{
  "Hello": 9906,
  "Ġworld": 1917,
  "<|endoftext|>": 151643,
  ...
}
```

**Key Information:**
- **Token-to-ID mapping**: String token → Integer ID (e.g., `"Hello"` → `9906`)
- **Special tokens**: Includes special tokens like `<|endoftext|>`, `<|im_start|>`, etc.
- **Total vocabulary size**: 151,936 entries

**Used By:**
- **`tokenizer.py`** - Initialization and encoding stages
  - Loaded into `self.encoder` dictionary during initialization
  - Used during `encode()` to convert BPE subword tokens to integer IDs
  - Reverse mapping (`self.decoder`) created for `decode()` function

**Pipeline Stage:** 
- **Initialization**: Load vocabulary mapping
- **Tokenization Stage**: Convert BPE tokens to token IDs
- **Decoding Stage**: Convert token IDs back to text

---

### 3. `merges.txt`

**Purpose:** Contains Byte Pair Encoding (BPE) merge rules in priority order.

**Format:** Plain text file, one merge rule per line

**Structure:**
```
Ġ Ġ
ĠĠ ĠĠ
Ġ t
ĠT he
...
```

**Key Information:**
- **Merge rules**: Each line contains two tokens to merge (e.g., `"Ġ"` + `"Ġ"` → `"ĠĠ"`)
- **Priority order**: Lower line numbers = higher priority merges
- **Rank system**: Line number (0-indexed) becomes the merge rank

**Used By:**
- **`tokenizer.py`** - Initialization and encoding stages
  - Loaded into `self.bpe_ranks` dictionary during initialization
  - Format: `{('Ġ', 'Ġ'): 0, ('ĠĠ', 'ĠĠ'): 1, ...}`
  - Used in `bpe()` method to iteratively merge character pairs
  - Lower rank number = higher priority merge

**Pipeline Stage:**
- **Initialization**: Load merge rules into lookup table
- **Tokenization Stage**: Apply BPE algorithm to split words into subword tokens

---

### 4. `model.safetensors`

**Purpose:** Contains all model weights in SafeTensors format (efficient, safe tensor storage).

**Format:** SafeTensors binary file

**Weight Keys Structure:**
```
model.embed_tokens.weight                    # [vocab_size, hidden_size] = [151936, 896]
model.layers.{i}.input_layernorm.weight      # [hidden_size] = [896]
model.layers.{i}.self_attn.q_proj.weight    # [hidden_size, hidden_size] = [896, 896]
model.layers.{i}.self_attn.q_proj.bias      # [hidden_size] = [896] (optional)
model.layers.{i}.self_attn.k_proj.weight    # [num_kv_heads * head_dim, hidden_size] = [128, 896]
model.layers.{i}.self_attn.k_proj.bias      # [num_kv_heads * head_dim] = [128] (optional)
model.layers.{i}.self_attn.v_proj.weight    # [num_kv_heads * head_dim, hidden_size] = [128, 896]
model.layers.{i}.self_attn.v_proj.bias      # [num_kv_heads * head_dim] = [128] (optional)
model.layers.{i}.self_attn.o_proj.weight    # [hidden_size, hidden_size] = [896, 896]
model.layers.{i}.self_attn.o_proj.bias      # [hidden_size] = [896] (optional)
model.layers.{i}.post_attention_layernorm.weight  # [hidden_size] = [896]
model.layers.{i}.mlp.gate_proj.weight       # [intermediate_size, hidden_size] = [4864, 896]
model.layers.{i}.mlp.gate_proj.bias         # [intermediate_size] = [4864] (optional)
model.layers.{i}.mlp.up_proj.weight         # [intermediate_size, hidden_size] = [4864, 896]
model.layers.{i}.mlp.up_proj.bias           # [intermediate_size] = [4864] (optional)
model.layers.{i}.mlp.down_proj.weight       # [hidden_size, intermediate_size] = [896, 4864]
model.layers.{i}.mlp.down_proj.bias         # [hidden_size] = [896] (optional)
model.norm.weight                           # [hidden_size] = [896]
lm_head.weight                              # [vocab_size, hidden_size] = [151936, 896]
                                           # OR model.embed_tokens.weight (if tied)
```

Where `{i}` ranges from `0` to `num_hidden_layers - 1` (0 to 23 for Qwen-0.5B).

**Used By:**
- **`embedding.py`** - Initialization stage
  - Extracts `model.embed_tokens.weight`
  - Shape: `[vocab_size, hidden_size]` = `[151936, 896]`
  - Used in `forward()` to perform embedding lookup: `embeddings = weights[token_ids, :]`

- **`self_attention.py`** - Initialization stage (per layer)
  - For each layer `i` (0 to 23), extracts:
    - **Attention weights:**
      - `model.layers.{i}.self_attn.q_proj.weight` and `.bias`
      - `model.layers.{i}.self_attn.k_proj.weight` and `.bias`
      - `model.layers.{i}.self_attn.v_proj.weight` and `.bias`
      - `model.layers.{i}.self_attn.o_proj.weight` and `.bias`
    - **Normalization weights:**
      - `model.layers.{i}.input_layernorm.weight`
      - `model.layers.{i}.post_attention_layernorm.weight`
    - **FFN weights:**
      - `model.layers.{i}.mlp.gate_proj.weight` and `.bias`
      - `model.layers.{i}.mlp.up_proj.weight` and `.bias`
      - `model.layers.{i}.mlp.down_proj.weight` and `.bias`
  - Also extracts final layer norm and LM head:
    - `model.norm.weight` - Final RMS normalization
    - `lm_head.weight` or `model.embed_tokens.weight` (if tied) - Language model head

**Pipeline Stage:**
- **Initialization**: Load all weights into memory
- **Prefill Stage**: Use weights for forward pass through all layers
- **Decode Stage**: Use weights for single-token forward pass (with KV cache)

---

### 5. `generation_config.json`

**Purpose:** Contains default generation hyperparameters.

**Format:** JSON file

**Key Fields:**
- `max_new_tokens`: 2048 - Maximum tokens to generate
- `eos_token_id`: 151643 - End-of-sequence token
- `bos_token_id`: 151643 - Beginning-of-sequence token
- `do_sample`: false - Whether to use sampling

**Used By:**
- Currently **not used** by our custom implementation
- Could be used to set default generation parameters
- Our implementation uses hardcoded defaults or function parameters

**Pipeline Stage:** Not currently used (optional for future enhancements)

---

### 6. `tokenizer_config.json` and `tokenizer.json`

**Purpose:** Full tokenizer configuration and state in HuggingFace format.

**Format:** JSON files

**Used By:**
- Currently **not used** by our custom implementation
- Our `SimpleBPETokenizer` uses `vocab.json` and `merges.txt` directly
- These files are used by HuggingFace's `AutoTokenizer` in `test_qwen_model.py`

**Pipeline Stage:** Not used by our custom pipeline (used only for validation)

---

### 7. `LICENSE` and `README.md`

**Purpose:** Model licensing and documentation.

**Used By:**
- Reference only
- Not parsed or used in inference pipeline

---

## 🔄 Inference Pipeline File Usage Flow

### Initialization Phase

```
1. Load config.json
   ├─ Extract architecture parameters
   └─ Determine number of layers, dimensions, etc.

2. Load vocab.json + merges.txt
   ├─ Build tokenizer encoder/decoder maps
   └─ Build BPE merge priority table

3. Load model.safetensors
   ├─ Extract embedding weights (embedding.py)
   ├─ Extract layer weights for each of 24 layers (self_attention.py)
   └─ Extract final norm + LM head weights
```

### Prefill Stage

```
Input Text
    ↓
tokenizer.py: encode()
    ├─ Uses: vocab.json, merges.txt
    └─ Output: [token_id_1, token_id_2, ...]
        ↓
embedding.py: forward()
    ├─ Uses: model.safetensors (embed_tokens.weight)
    └─ Output: [embedding_1, embedding_2, ...]
        ↓
self_attention.py: forward() × 24 layers
    ├─ Uses: model.safetensors (all layer weights)
    ├─ Uses: config.json (for dimensions)
    └─ Output: [hidden_state_1, hidden_state_2, ...]
        ↓
Final Norm + LM Head
    ├─ Uses: model.safetensors (model.norm.weight, lm_head.weight)
    └─ Output: logits [vocab_size]
        ↓
Sample first token
```

### Decode Stage (Autoregressive)

```
Generated Token ID
    ↓
embedding.py: forward([token_id])
    ├─ Uses: model.safetensors (embed_tokens.weight)
    └─ Output: embedding vector
        ↓
self_attention.py: forward_decode() × 24 layers
    ├─ Uses: model.safetensors (all layer weights)
    ├─ Uses: KV cache (from prefill)
    └─ Output: hidden_state
        ↓
Final Norm + LM Head
    ├─ Uses: model.safetensors (model.norm.weight, lm_head.weight)
    └─ Output: logits [vocab_size]
        ↓
Sample next token
    ↓
Repeat until EOS or max_new_tokens
    └─ Uses: config.json (eos_token_id)
```

---

## 📊 Summary Table

| File | Format | Used By | Pipeline Stage | Key Information |
|------|--------|---------|----------------|-----------------|
| `config.json` | JSON | embedding.py, self_attention.py, decode.py | Initialization | Architecture params, dimensions, token IDs |
| `vocab.json` | JSON | tokenizer.py | Initialization, Tokenization, Decoding | Token string → ID mapping |
| `merges.txt` | Text | tokenizer.py | Initialization, Tokenization | BPE merge rules and priorities |
| `model.safetensors` | Binary | embedding.py, self_attention.py | Initialization, Prefill, Decode | All model weights |
| `generation_config.json` | JSON | (Not used) | - | Default generation params |
| `tokenizer_config.json` | JSON | (Not used) | - | HuggingFace tokenizer config |
| `tokenizer.json` | JSON | (Not used) | - | HuggingFace tokenizer state |

---

## 🔍 Weight Key Patterns

For hardware implementation, understanding the weight key naming pattern is crucial:

### Pattern: `model.layers.{layer_idx}.{component}.{operation}.{type}`

- `layer_idx`: 0 to 23 (for 24 layers)
- `component`: `self_attn` or `mlp`
- `operation`: 
  - For attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - For MLP: `gate_proj`, `up_proj`, `down_proj`
- `type`: `weight` or `bias`

### Special Keys:
- `model.embed_tokens.weight` - Embedding layer
- `model.layers.{i}.input_layernorm.weight` - Pre-attention norm
- `model.layers.{i}.post_attention_layernorm.weight` - Pre-FFN norm
- `model.norm.weight` - Final layer norm
- `lm_head.weight` - Language model head (or tied to embed_tokens)

---

## 💡 Implementation Notes

1. **Weight Loading**: All weights are loaded once during initialization. For FPGA implementation, this would be done via host-to-FPGA transfer.

2. **Memory Requirements**: 
   - Embedding weights: ~151,936 × 896 × 4 bytes ≈ 544 MB
   - Per-layer weights: ~896 × 896 × 4 bytes × multiple matrices ≈ several MB per layer
   - Total model size: ~1-2 GB (depending on precision)

3. **KV Cache**: Not stored in files - dynamically generated during prefill and updated during decode.

4. **Tied Embeddings**: Qwen uses tied embeddings (`tie_word_embeddings: true`), meaning `lm_head.weight` may be the same as `model.embed_tokens.weight`.

---

## 🚀 Hardware Implementation Considerations

For FPGA implementation, consider:

1. **Weight Storage**: Model weights must be transferred to FPGA memory (BRAM/DRAM)
2. **Weight Format**: Convert from float32 to fixed-point or custom precision
3. **Streaming**: Large weight matrices may need to be streamed from external memory
4. **Caching**: Frequently accessed weights (like embeddings) should be cached in fast memory
5. **Parallel Access**: Multiple layers may need parallel weight access for pipelining

---

**Last Updated:** Based on software implementation in `tokenizer.py`, `embedding.py`, `self_attention.py`, and `decode.py`

