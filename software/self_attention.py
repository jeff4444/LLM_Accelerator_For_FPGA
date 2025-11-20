import json
import math
import os
import numpy as np

from safetensors.torch import load_file


# ==========================================
# 🧮 FPGA MATH KERNELS (Hardware Primitives)
# ==========================================

def vec_dot(a, b):
    """
    Hardware: MAC (Multiply-Accumulate) Unit
    Computes: sum(a[i] * b[i])
    
    FPGA Implementation:
    - Instantiate N parallel multipliers
    - Feed into adder tree (log N depth)
    - Final accumulator register
    """
    acc = 0.0
    for i in range(len(a)):
        acc += a[i] * b[i]
    return acc


def vec_add(a, b):
    """
    Hardware: Parallel Adder Array
    Element-wise addition: c[i] = a[i] + b[i]
    
    FPGA Implementation:
    - N parallel floating-point adders
    - Single cycle latency with pipelining
    """
    return [x + y for x, y in zip(a, b)]


def vec_scale(a, factor):
    """
    Hardware: Parallel Multiplier Array
    Scalar multiplication: b[i] = a[i] * factor
    
    FPGA Implementation:
    - Broadcast scalar to N multipliers
    - Parallel execution
    """
    return [x * factor for x in a]


def mat_vec_mul(matrix, vector):
    """
    Hardware: Matrix-Vector Multiplication Engine
    Input: Matrix [Rows][Cols], Vector [Cols]
    Output: Vector [Rows]
    
    FPGA Implementation:
    - Each output element computed by MAC unit
    - Can parallelize across rows
    - Systolic array architecture for efficiency
    """
    result = []
    # first make sure the cols are the same as the vector
    if len(matrix[0]) != len(vector):
        raise ValueError(f"Matrix columns ({len(matrix[0])}) do not match vector length ({len(vector)})")
    for row in matrix:
        result.append(vec_dot(row, vector))
    return result


def mat_vec_mul_with_bias(matrix, vector, bias=None):
    """
    Hardware: Matrix-Vector Multiplication Engine with Bias
    Input: Matrix [Rows][Cols], Vector [Cols], Bias [Rows] (optional)
    Output: Vector [Rows]
    
    Performs: output = matrix @ vector + bias (if bias is provided)
    """
    result = mat_vec_mul(matrix, vector)
    if bias is not None:
        result = vec_add(result, bias)
    return result


def softmax_kernel(x):
    """
    Hardware: Softmax Unit
    
    Steps:
    1. Find max value (for numerical stability)
    2. Subtract max and exponentiate
    3. Sum all exponentials
    4. Divide each by sum
    
    FPGA Implementation:
    - Max-finder: Tree reduction (log N depth)
    - Exponential: LUT or CORDIC algorithm
    - Division: Iterative or Newton-Raphson
    """
    # Numerical stability: subtract max before exp
    max_val = max(x)
    exps = [math.exp(val - max_val) for val in x]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]


def rms_norm_kernel(x, weight, eps=1e-6):
    """
    Hardware: RMS Normalization Block
    
    RMS Norm formula:
    y = (x / RMS(x)) * weight
    where RMS(x) = sqrt(mean(x^2))
    
    FPGA Implementation:
    1. Square all elements (parallel multipliers)
    2. Sum squares (adder tree)
    3. Divide by N, add epsilon
    4. Inverse square root (LUT or iterative)
    5. Multiply by weight (parallel)
    """
    sum_sq = sum(v**2 for v in x)
    mean_sq = sum_sq / len(x)
    inv_rms = 1.0 / math.sqrt(mean_sq + eps)
    
    # Normalize and apply learned weight
    return [val * inv_rms * w for val, w in zip(x, weight)]


def silu_kernel(x):
    """
    Hardware: SiLU (Sigmoid Linear Unit) Activation
    SiLU(x) = x * sigmoid(x)
    
    FPGA Implementation:
    - Sigmoid: LUT or CORDIC algorithm
    - Element-wise multiplication
    """
    def sigmoid(val):
        # Numerical stability: clip to prevent overflow
        if val > 10:
            return 1.0
        if val < -10:
            return 0.0
        return 1.0 / (1.0 + math.exp(-val))
    
    return [val * sigmoid(val) for val in x]


def vec_mul(a, b):
    """
    Hardware: Parallel Multiplier Array
    Element-wise multiplication: c[i] = a[i] * b[i]
    
    FPGA Implementation:
    - N parallel floating-point multipliers
    - Single cycle latency with pipelining
    """
    return [x * y for x, y in zip(a, b)]

def apply_rotary_pos_emb(q_or_k, position, head_dim, rope_theta=1000000.0):
    """
    Pure Software implementation of RoPE (recomputing cos/sin on the fly).
    
    CRITICAL CHANGE:
    This uses 'Split-Half' pairing (index i pairs with index i + half_dim).
    This is required to match the weights of Qwen, Llama, and HuggingFace models.
    
    Args:
        q_or_k: Query or Key vector [head_dim]
        position: Token position in sequence (0-indexed)
        head_dim: Dimension of each attention head
        rope_theta: Base frequency
    """
    result = list(q_or_k) # Create a copy to avoid modifying input in place
    half_dim = head_dim // 2

    # We iterate 0 to half_dim (e.g., 0 to 31 for size 64)
    for i in range(half_dim):
        # 1. Calculate Frequency on the fly
        # formula: theta ^ (-2i/dim)
        freq = 1.0 / (rope_theta ** (2 * i / head_dim))
        angle = position * freq
        
        # 2. Calculate Trig on the fly
        cos_val = math.cos(angle)
        sin_val = math.sin(angle)
        
        # 3. Identify the Correct Pair (Split-Half Strategy)
        idx_1 = i
        idx_2 = i + half_dim
        
        val_1 = q_or_k[idx_1]
        val_2 = q_or_k[idx_2]
        
        # 4. Apply Rotation Matrix
        # Corresponds to: (x * cos) + (rotate_half(x) * sin)
        # Where rotate_half transforms [x1, x2] -> [-x2, x1]
        
        # Output 1: x1*cos - x2*sin
        result[idx_1] = val_1 * cos_val - val_2 * sin_val
        
        # Output 2: x1*sin + x2*cos
        result[idx_2] = val_1 * sin_val + val_2 * cos_val
        
    return result

# def apply_rotary_pos_emb(q_or_k, position, head_dim, rope_theta=1000000.0):
#     """
#     Hardware: Rotary Position Embedding (RoPE) Unit
    
#     Applies rotary position embeddings to Q or K vectors.
#     This implementation uses the standard pair-wise rotation approach, which is mathematically
#     equivalent to the transformers library's rotate_half approach.
    
#     FPGA Implementation:
#     - Precompute cos/sin LUTs for common positions
#     - Apply rotation: [x, y] -> [x*cos - y*sin, x*sin + y*cos]
#     - Parallel rotation units for each dimension pair
    
#     Args:
#         q_or_k: Query or Key vector [head_dim]
#         position: Token position in sequence (0-indexed)
#         head_dim: Dimension of each attention head
#         rope_theta: Base frequency (model-specific, default 1M for Qwen2.5)
        
#     Returns:
#         Rotated vector [head_dim]
#     """
#     result = list(q_or_k)  # Make a copy
    
#     # Process pairs of dimensions
#     # Standard RoPE formula: inv_freq = 1.0 / (base ** (i / dim)) where i = 0, 2, 4, ...
#     # This is equivalent to: 1.0 / (base ** (2 * pair_idx / dim)) where pair_idx = 0, 1, 2, ...
#     for i in range(0, head_dim, 2):
#         pair_idx = i // 2
#         inv_freq = 1.0 / (rope_theta ** (2.0 * pair_idx / head_dim))
        
#         # Angle for this position
#         angle = position * inv_freq
        
#         # Precompute cos and sin
#         cos_val = math.cos(angle)
#         sin_val = math.sin(angle)
        
#         # Apply 2D rotation to the pair (i, i+1)
#         # Standard rotation matrix: [x, y] -> [x*cos - y*sin, x*sin + y*cos]
#         x = q_or_k[i]
#         y = q_or_k[i + 1]
        
#         result[i] = x * cos_val - y * sin_val
#         result[i + 1] = x * sin_val + y * cos_val
    
#     return result


# ==========================================
# 🎯 SELF-ATTENTION MODULE
# ==========================================

class SelfAttentionLayer:
    
    def __init__(self, layer_idx, model_path=None, config_path=None, model_dir=None):
        """
        INITIALIZATION - Self-Attention Layer
        
        Loads Q, K, V, and Output projection weights for a single transformer layer.
        Implements Grouped Query Attention (GQA) as used in Qwen2.5.
        
        Args:
            layer_idx: Which transformer layer (0-indexed)
            model_path: Path to model.safetensors file (optional if model_dir provided)
            config_path: Path to config.json file (optional if model_dir provided)
            model_dir: Path to model directory (e.g., "../Qwen2.5-0.5B")
        """
        # Determine file paths
        if model_dir:
            if model_path is None:
                model_path = os.path.join(model_dir, "model.safetensors")
            if config_path is None:
                config_path = os.path.join(model_dir, "config.json")
        
        if model_path is None or config_path is None:
            raise ValueError("Must provide either (model_path and config_path) or model_dir")
        
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Extract architecture parameters
        self.hidden_size = self.config.get("hidden_size")  # 896
        self.num_attention_heads = self.config.get("num_attention_heads")  # 14 (Q heads)
        self.num_key_value_heads = self.config.get("num_key_value_heads")  # 2 (K, V heads)
        self.head_dim = self.hidden_size // self.num_attention_heads  # 64
        self.layer_idx = layer_idx
        self.rope_theta = self.config.get("rope_theta", 1000000.0)  # Default to 1000000.0 if not found
        self.rms_norm_eps = self.config.get("rms_norm_eps", 1e-6)  # RMS norm epsilon
        
        # Calculate Grouped Query Attention parameters
        # Each KV head services multiple Q heads
        self.num_queries_per_kv = self.num_attention_heads // self.num_key_value_heads  # 7
        
        # Load model weights
        self.model_path = model_path
        state_dict = load_file(model_path)
        
        # Weight key prefix for this layer
        prefix = f"model.layers.{layer_idx}"
        
        # Helper function to extract and convert weights to list of lists
        def get_weight_matrix(key_name, weight_type="self_attn", transpose=False):
            """
            Extracts weight matrix and converts to Python list of lists.
            PyTorch Linear weights are stored as [Out_Features, In_Features].
            """
            full_key = f"{prefix}.{weight_type}.{key_name}.weight" if weight_type != "" else f"{prefix}.{key_name}.weight"
            if full_key not in state_dict:
                raise KeyError(f"Weight key '{full_key}' not found in model!")
            
            tensor = state_dict[full_key]
            # Convert to numpy - ensure float32
            numpy_array = tensor.detach().float().numpy()
            
            if transpose:
                numpy_array = numpy_array.T
            
            return numpy_array.tolist()
        
        # Helper function to extract bias vectors
        def get_bias_vector(key_name, weight_type="self_attn"):
            """
            Extracts bias vector and converts to Python list.
            Returns None if bias doesn't exist (some layers may not have bias).
            """
            full_key = f"{prefix}.{weight_type}.{key_name}.bias" if weight_type != "" else f"{prefix}.{key_name}.bias"
            if full_key not in state_dict:
                return None
            
            tensor = state_dict[full_key]
            # Convert to numpy - ensure float32
            numpy_array = tensor.detach().float().numpy()
            
            return numpy_array.flatten().tolist()
        
        # Load projection matrices
        # Q_proj: [hidden_size, hidden_size] = [896, 896]
        # K_proj: [num_kv_heads * head_dim, hidden_size] = [128, 896]
        # V_proj: [num_kv_heads * head_dim, hidden_size] = [128, 896]
        # O_proj: [hidden_size, hidden_size] = [896, 896]
        
        self.w_q = get_weight_matrix("q_proj", transpose=False)
        self.b_q = get_bias_vector("q_proj")
        
        self.w_k = get_weight_matrix("k_proj", transpose=False)
        self.b_k = get_bias_vector("k_proj")
        
        self.w_v = get_weight_matrix("v_proj", transpose=False)
        self.b_v = get_bias_vector("v_proj")
        
        self.w_o = get_weight_matrix("o_proj", transpose=False)
        self.b_o = get_bias_vector("o_proj")
        self.w_gate = get_weight_matrix("gate_proj", weight_type="mlp", transpose=False)
        self.b_gate = get_bias_vector("gate_proj", weight_type="mlp")
        self.w_up = get_weight_matrix("up_proj", weight_type="mlp", transpose=False)
        self.b_up = get_bias_vector("up_proj", weight_type="mlp")
        self.w_down = get_weight_matrix("down_proj", weight_type="mlp", transpose=False)
        self.b_down = get_bias_vector("down_proj", weight_type="mlp")
        
        # Norm Weights (1D Lists)
        # Load norm weights carefully - they should be 1D
        norm1_key = f"{prefix}.input_layernorm.weight"
        norm2_key = f"{prefix}.post_attention_layernorm.weight"
        
        norm1_tensor = state_dict[norm1_key].detach().float().numpy()
        norm2_tensor = state_dict[norm2_key].detach().float().numpy()
        
        # Flatten to 1D if needed
        self.norm1_w = norm1_tensor.flatten().tolist()
        self.norm2_w = norm2_tensor.flatten().tolist()
        
        # Clean up
        del state_dict
        
        # Initialize KV cache storage
        # Structure: k_cache[kv_head_idx][seq_pos][head_dim]
        #            v_cache[kv_head_idx][seq_pos][head_dim]
        self.k_cache = None  # Will be initialized on first forward pass
        self.v_cache = None  # Will be initialized on first forward pass
    
    
    def forward(self, x_seq, apply_norm=False):
        """
        THE SELF-ATTENTION FORWARD PASS
        
        Implements Multi-Head Self-Attention with Grouped Query Attention (GQA).
        
        Args:
            x_seq: Input embeddings as list of lists [Seq_Len][Hidden_Dim]
                   Can be numpy arrays or nested lists
            apply_norm: Whether to apply RMS normalization before attention
            
        Returns:
            List of lists [Seq_Len][Hidden_Dim] - attention output
        """
        # Convert numpy arrays to lists if needed
        if hasattr(x_seq, 'tolist'):
            x_seq = x_seq.tolist()
        
        seq_len = len(x_seq)
        
        # --- STEP 1: PRE-ATTENTION NORMALIZATION ---
        if apply_norm and self.norm1_w is not None:
            x_norm = [rms_norm_kernel(token, self.norm1_w, eps=self.rms_norm_eps) for token in x_seq]
        else:
            x_norm = x_seq
        
        # --- STEP 2: Q, K, V PROJECTIONS ---
        # Q projection: [Seq_Len][Hidden_Size]
        # Each token gets projected to full hidden_size dimension
        Q_flat = [mat_vec_mul_with_bias(self.w_q, token, self.b_q) for token in x_norm]
        
        # K, V projections: [Seq_Len][num_kv_heads * head_dim]
        # Smaller dimension due to Grouped Query Attention
        K_flat = [mat_vec_mul_with_bias(self.w_k, token, self.b_k) for token in x_norm]
        V_flat = [mat_vec_mul_with_bias(self.w_v, token, self.b_v) for token in x_norm]
        
        # --- STEP 2.3: APPLY ROTARY POSITION EMBEDDINGS ---
        
        # Apply RoPE to each head separately
        # For Q: we have num_attention_heads heads
        # For K: we have num_key_value_heads heads
        
        # Apply RoPE to Q heads
        Q_rope = []
        for t in range(seq_len):
            q_token = []
            for q_head_idx in range(self.num_attention_heads):
                q_start = q_head_idx * self.head_dim
                q_end = q_start + self.head_dim
                q_head = Q_flat[t][q_start:q_end]
                # Apply RoPE with position t
                q_head_rope = apply_rotary_pos_emb(q_head, t, self.head_dim, rope_theta=self.rope_theta)
                q_token.extend(q_head_rope)
            Q_rope.append(q_token)
        Q_flat = Q_rope
        
        # Apply RoPE to K heads
        K_rope = []
        for t in range(seq_len):
            k_token = []
            for kv_head_idx in range(self.num_key_value_heads):
                kv_start = kv_head_idx * self.head_dim
                kv_end = kv_start + self.head_dim
                k_head = K_flat[t][kv_start:kv_end]
                # Apply RoPE with position t
                k_head_rope = apply_rotary_pos_emb(k_head, t, self.head_dim, rope_theta=self.rope_theta)
                k_token.extend(k_head_rope)
            K_rope.append(k_token)
        K_flat = K_rope
        
        # --- STEP 2.5: STORE KV CACHE ---
        # Organize K and V by KV head and store in cache
        # Structure: k_cache[kv_head_idx][seq_pos][head_dim]
        #            v_cache[kv_head_idx][seq_pos][head_dim]
        self.k_cache = []
        self.v_cache = []
        
        for kv_head_idx in range(self.num_key_value_heads):
            kv_start = kv_head_idx * self.head_dim
            kv_end = kv_start + self.head_dim
            
            # Extract K and V for this KV head across all sequence positions
            k_head_cache = [token[kv_start:kv_end] for token in K_flat]
            v_head_cache = [token[kv_start:kv_end] for token in V_flat]
            
            self.k_cache.append(k_head_cache)
            self.v_cache.append(v_head_cache)
        
        # --- STEP 3: MULTI-HEAD ATTENTION COMPUTATION ---
        
        # Initialize output container
        attn_output_seq = [[0.0] * self.hidden_size for _ in range(seq_len)]
        
        # Scaling factor for attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        
        # Process each attention head
        # In GQA: Multiple Q heads share the same K, V head
        for q_head_idx in range(self.num_attention_heads):
            # Determine which KV head this Q head uses
            kv_head_idx = q_head_idx // self.num_queries_per_kv
            
            # Calculate dimension slices for this head
            # Q heads are contiguous in the full hidden_size
            q_start = q_head_idx * self.head_dim
            q_end = q_start + self.head_dim
            
            # KV heads are in the smaller KV dimension space
            kv_start = kv_head_idx * self.head_dim
            kv_end = kv_start + self.head_dim
            
            # Extract Q, K, V for this head across all sequence positions
            # Q_head: [Seq_Len][Head_Dim]
            Q_head = [token[q_start:q_end] for token in Q_flat]
            K_head = [token[kv_start:kv_end] for token in K_flat]
            V_head = [token[kv_start:kv_end] for token in V_flat]
            
            # Compute attention for each query position
            for t_q in range(seq_len):
                # --- Calculate Attention Scores ---
                # Score[t_q, t_k] = (Q[t_q] · K[t_k]) / sqrt(head_dim)
                scores = []
                for t_k in range(seq_len):
                    score = vec_dot(Q_head[t_q], K_head[t_k]) * scale
                    scores.append(score)
                
                # --- Apply Causal Mask (for autoregressive generation) ---
                # Mask out future positions by setting their scores to -inf
                # This ensures we only attend to current and past tokens
                for t_k in range(t_q + 1, seq_len):
                    scores[t_k] = float('-inf')
                
                # --- Softmax to get attention probabilities ---
                attn_probs = softmax_kernel(scores)
                
                # --- Weighted sum of Values ---
                # context[t_q] = sum(attn_probs[t_k] * V[t_k])
                head_context = [0.0] * self.head_dim
                for t_v in range(seq_len):
                    # Skip if attention probability is zero (for efficiency)
                    if attn_probs[t_v] == 0.0:
                        continue
                    
                    weighted_v = vec_scale(V_head[t_v], attn_probs[t_v])
                    head_context = vec_add(head_context, weighted_v)
                
                # --- Write head output to concatenated output ---
                # All heads are concatenated along the hidden dimension
                for i in range(self.head_dim):
                    attn_output_seq[t_q][q_start + i] = head_context[i]
        
        # --- STEP 4: OUTPUT PROJECTION ---
        post_attn = [mat_vec_mul_with_bias(self.w_o, token, self.b_o) for token in attn_output_seq]
        
        # --- STEP 5: RESIDUAL CONNECTION ---
        x_resid = [vec_add(x_seq[i], post_attn[i]) for i in range(seq_len)]
        
        # --- BLOCK 2: FFN (SwiGLU) ---
        
        # 1. Norm
        x_norm2 = [rms_norm_kernel(token, self.norm2_w, eps=self.rms_norm_eps) for token in x_resid]
        
        # 2. Gate & Up Projections
        gate_out = [mat_vec_mul_with_bias(self.w_gate, token, self.b_gate) for token in x_norm2]
        up_out = [mat_vec_mul_with_bias(self.w_up, token, self.b_up) for token in x_norm2]
        
        # 3. Activation (SiLU) & Element-wise Mul
        # SwiGLU = (SiLU(Gate) * Up)
        mlp_hidden = []
        for i in range(seq_len):
            act_gate = silu_kernel(gate_out[i])
            mlp_hidden.append(vec_mul(act_gate, up_out[i]))
        
        # 4. Down Projection
        mlp_out = [mat_vec_mul_with_bias(self.w_down, token, self.b_down) for token in mlp_hidden]
        
        # 5. Final Residual
        final_out = [vec_add(x_resid[i], mlp_out[i]) for i in range(seq_len)]
        
        return final_out
    
    def get_kv_cache(self):
        """
        Retrieve the stored KV cache.
        
        Returns:
            tuple: (k_cache, v_cache) where:
                - k_cache: List of lists [kv_head_idx][seq_pos][head_dim]
                - v_cache: List of lists [kv_head_idx][seq_pos][head_dim]
            Returns (None, None) if cache hasn't been populated yet.
        """
        if self.k_cache is None or self.v_cache is None:
            return None, None
        return self.k_cache, self.v_cache
    
    def clear_kv_cache(self):
        """
        Clear the KV cache. Useful for resetting state between different sequences.
        """
        self.k_cache = None
        self.v_cache = None
        
    def get_lm_head_w_and_final_norm_w(self):
        state_dict = load_file(self.model_path)
        final_norm_w = state_dict["model.norm.weight"].tolist()
        if "lm_head.weight" in state_dict:
            lm_head_w = state_dict["lm_head.weight"].tolist()
        elif "model.embed_tokens.weight" in state_dict:
            lm_head_w = state_dict["model.embed_tokens.weight"].tolist()
        else:
            raise KeyError("Critical Error: Could not find output projection weights!")
        return final_norm_w, lm_head_w


# ==========================================
# 🏁 DRIVER CODE FOR TESTING
# ==========================================

if __name__ == "__main__":
    import sys
    
    # Add parent directory to path to import embedding module
    sys.path.append(os.path.dirname(__file__))
    from tokenizer import SimpleBPETokenizer
    from embedding import EmbeddingLayer
    
    # Path to model directory
    model_dir = os.path.join(os.path.dirname(__file__), "..", "Qwen2.5-0.5B")
    
    # --- STEP 1: Initialize Components ---
    tokenizer = SimpleBPETokenizer(model_dir=model_dir)
    embedding_layer = EmbeddingLayer(model_dir=model_dir)
    
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    attention_layers = [SelfAttentionLayer(layer_idx=i, model_dir=model_dir) for i in range(config.get("num_hidden_layers"))]
    
    # --- STEP 2: Process Input ---
    user_input = input("Enter text to process through self-attention: ")
    
    # Tokenization
    token_ids = tokenizer.encode(user_input)
    
    # Embedding
    embeddings = embedding_layer.forward(token_ids)
    
    # Self-Attention
    for attention_layer in attention_layers:
        embeddings = attention_layer.forward(embeddings, apply_norm=True)
        
    final_norm_w, lm_head_w = attention_layers[-1].get_lm_head_w_and_final_norm_w()
        
    embeddings = [rms_norm_kernel(token, final_norm_w) for token in embeddings]
    last_token_vector = embeddings[-1]
    logits = mat_vec_mul(lm_head_w, last_token_vector)
        
    
    # now predict the next token
    max_val = -float('inf')
    max_id = -1
    for i, val in enumerate(logits):
        if val > max_val:
            max_val = val
            max_id = i
    
    decoded_token = tokenizer.decoder.get(max_id, f"<unk:{max_id}>")
