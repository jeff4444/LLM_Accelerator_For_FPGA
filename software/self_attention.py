import json
import math
import os

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
    for row in matrix:
        result.append(vec_dot(row, vector))
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
        print(f"Loading config from {config_path}...")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Extract architecture parameters
        self.hidden_size = self.config.get("hidden_size")  # 896
        self.num_attention_heads = self.config.get("num_attention_heads")  # 14 (Q heads)
        self.num_key_value_heads = self.config.get("num_key_value_heads")  # 2 (K, V heads)
        self.head_dim = self.hidden_size // self.num_attention_heads  # 64
        self.layer_idx = layer_idx
        
        # Calculate Grouped Query Attention parameters
        # Each KV head services multiple Q heads
        self.num_queries_per_kv = self.num_attention_heads // self.num_key_value_heads  # 7
        
        print(f"\n--- Layer {layer_idx} Configuration ---")
        print(f"Hidden Size: {self.hidden_size}")
        print(f"Num Attention Heads (Q): {self.num_attention_heads}")
        print(f"Num Key-Value Heads: {self.num_key_value_heads}")
        print(f"Head Dimension: {self.head_dim}")
        print(f"Queries per KV head (GQA): {self.num_queries_per_kv}")
        
        # Load model weights
        print(f"Loading weights from {model_path}...")
        state_dict = load_file(model_path)
        
        # Weight key prefix for this layer
        prefix = f"model.layers.{layer_idx}"
        
        # Helper function to extract and convert weights to list of lists
        def get_weight_matrix(key_name, weight_type="self_attn"):
            
            """
            Extracts weight matrix and converts to Python list of lists.
            PyTorch Linear weights are stored as [Out_Features, In_Features].
            """
            full_key = f"{prefix}.{weight_type}.{key_name}.weight" if weight_type != "" else f"{prefix}.{key_name}.weight"
            if full_key not in state_dict:
                raise KeyError(f"Weight key '{full_key}' not found in model!")
            
            tensor = state_dict[full_key]
            # Convert to numpy first, then to list of lists
            numpy_array = tensor.detach().float().numpy()
            return numpy_array.tolist()
        
        # Load projection matrices
        # Q_proj: [hidden_size, hidden_size] = [896, 896]
        # K_proj: [num_kv_heads * head_dim, hidden_size] = [128, 896]
        # V_proj: [num_kv_heads * head_dim, hidden_size] = [128, 896]
        # O_proj: [hidden_size, hidden_size] = [896, 896]
        
        print(f"Loading Q projection weights...")
        self.w_q = get_weight_matrix("q_proj")
        
        print(f"Loading K projection weights...")
        self.w_k = get_weight_matrix("k_proj")
        
        print(f"Loading V projection weights...")
        self.w_v = get_weight_matrix("v_proj")
        
        print(f"Loading Output projection weights...")
        self.w_o = get_weight_matrix("o_proj")

        print(f"Loading FFN weights...")
        self.w_gate = get_weight_matrix("gate_proj", weight_type="mlp")
        self.w_up = get_weight_matrix("up_proj", weight_type="mlp")
        self.w_down = get_weight_matrix("down_proj", weight_type="mlp")
        
        # Norm Weights (1D Lists)
        self.norm1_w = get_weight_matrix("input_layernorm", weight_type="")
        self.norm2_w = get_weight_matrix("post_attention_layernorm", weight_type="")
        print(f"Loaded norm weights (norm1_w and norm2_w)")
        
        # Convert Norm & Head
        self.final_norm_w = state_dict["model.norm.weight"].tolist()
        # print(state_dict.keys())
        # self.lm_head_w = state_dict["lm_head.weight"].tolist()
        
        # Clean up
        del state_dict
        
        # Initialize KV cache storage
        # Structure: k_cache[kv_head_idx][seq_pos][head_dim]
        #            v_cache[kv_head_idx][seq_pos][head_dim]
        self.k_cache = None  # Will be initialized on first forward pass
        self.v_cache = None  # Will be initialized on first forward pass
        
        print(f"✓ Self-Attention Layer {layer_idx} initialized successfully\n")
    
    
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
        
        print(f"\n=== Self-Attention Forward Pass ===")
        print(f"Input sequence length: {seq_len}")
        print(f"Hidden dimension: {self.hidden_size}")
        
        # --- STEP 1: PRE-ATTENTION NORMALIZATION ---
        if apply_norm and self.norm1_w is not None:
            print("Step 1: Applying RMS Normalization...")
            x_norm = [rms_norm_kernel(token, self.norm1_w) for token in x_seq]
        else:
            print("Step 1: Skipping normalization")
            x_norm = x_seq
        
        # --- STEP 2: Q, K, V PROJECTIONS ---
        print("Step 2: Computing Q, K, V projections...")
        
        # Q projection: [Seq_Len][Hidden_Size]
        # Each token gets projected to full hidden_size dimension
        Q_flat = [mat_vec_mul(self.w_q, token) for token in x_norm]
        
        # K, V projections: [Seq_Len][num_kv_heads * head_dim]
        # Smaller dimension due to Grouped Query Attention
        K_flat = [mat_vec_mul(self.w_k, token) for token in x_norm]
        V_flat = [mat_vec_mul(self.w_v, token) for token in x_norm]
        
        print(f"  Q shape: [{seq_len}][{len(Q_flat[0])}]")
        print(f"  K shape: [{seq_len}][{len(K_flat[0])}]")
        print(f"  V shape: [{seq_len}][{len(V_flat[0])}]")
        
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
        
        print(f"  KV cache stored: {self.num_key_value_heads} heads, {seq_len} positions, {self.head_dim} dims per head")
        
        # --- STEP 3: MULTI-HEAD ATTENTION COMPUTATION ---
        print(f"Step 3: Computing {self.num_attention_heads}-head attention (GQA)...")
        
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
        print("Step 4: Output projection...")
        post_attn = [mat_vec_mul(self.w_o, token) for token in attn_output_seq]
        
        # --- STEP 5: RESIDUAL CONNECTION ---
        print("Step 5: Residual connection...")
        x_resid = [vec_add(x_seq[i], post_attn[i]) for i in range(seq_len)]
        
        # --- BLOCK 2: FFN (SwiGLU) ---
        print("Step 6: FFN (SwiGLU) block...")
        
        # 1. Norm
        x_norm2 = [rms_norm_kernel(token, self.norm2_w) for token in x_resid]
        
        # 2. Gate & Up Projections
        gate_out = [mat_vec_mul(self.w_gate, token) for token in x_norm2]
        up_out = [mat_vec_mul(self.w_up, token) for token in x_norm2]
        
        # 3. Activation (SiLU) & Element-wise Mul
        # SwiGLU = (SiLU(Gate) * Up)
        mlp_hidden = []
        for i in range(seq_len):
            act_gate = silu_kernel(gate_out[i])
            mlp_hidden.append(vec_mul(act_gate, up_out[i]))
        
        # 4. Down Projection
        mlp_out = [mat_vec_mul(self.w_down, token) for token in mlp_hidden]
        
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
    
    print("=" * 80)
    print("SELF-ATTENTION PIPELINE TEST")
    print("=" * 80)
    print()
    
    # --- STEP 1: Initialize Components ---
    print("STEP 1: Initializing Tokenizer...")
    print("-" * 80)
    tokenizer = SimpleBPETokenizer(model_dir=model_dir)
    print(f"✓ Tokenizer ready with {len(tokenizer.encoder)} vocab items\n")
    
    print("STEP 2: Initializing Embedding Layer...")
    print("-" * 80)
    embedding_layer = EmbeddingLayer(model_dir=model_dir)
    print(f"✓ Embedding layer ready\n")
    
    print("STEP 3: Initializing Self-Attention Layer (Layer 0)...")
    print("-" * 80)
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    attention_layers = [SelfAttentionLayer(layer_idx=i, model_dir=model_dir) for i in range(config.get("num_hidden_layers"))]
    print(f"✓ Self-attention layer ready\n")
    
    # --- STEP 2: Process Input ---
    print("STEP 4: Input Processing")
    print("-" * 80)
    user_input = input("Enter text to process through self-attention: ")
    print(f"Input: '{user_input}'\n")
    
    # Tokenization
    print("STEP 5: Tokenization")
    print("-" * 80)
    token_ids = tokenizer.encode(user_input)
    print(f"Token IDs: {token_ids}")
    print(f"Num tokens: {len(token_ids)}\n")
    
    # Embedding
    print("STEP 6: Embedding Lookup")
    print("-" * 80)
    embeddings = embedding_layer.forward(token_ids)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"(Sequence: {embeddings.shape[0]}, Hidden: {embeddings.shape[1]})\n")
    
    # Self-Attention
    print("STEP 7: Self-Attention Forward Pass")
    print("-" * 80)
    for attention_layer in attention_layers:
        embeddings = attention_layer.forward(embeddings, apply_norm=True)
        
    embeddings = [rms_norm_kernel(token, attention_layers[-1].final_norm_w) for token in embeddings]
    last_token_vector = embeddings[-1]
    # logits = mat_vec_mul(attention_layers[-1].lm_head_w, last_token_vector)
        
    
    # now predict the next token
    max_val = -float('inf')
    max_id = -1
    for i, val in enumerate(last_token_vector):
        if val > max_val:
            max_val = val
            max_id = i
    
    print(f"Next token: {max_id}")
    print()
