import math
import sys
import os
import random
import numpy as np

# Import your existing hardware primitives and layer
from self_attention import (
    vec_dot, vec_add, vec_scale, mat_vec_mul, mat_vec_mul_with_bias,
    softmax_kernel, rms_norm_kernel, silu_kernel, vec_mul,
    apply_rotary_pos_emb,
    SelfAttentionLayer
)
from tokenizer import SimpleBPETokenizer
from embedding import EmbeddingLayer


# ==========================================
# 🧠 DECODE SPECIFIC KERNELS
# ==========================================

def sample_next_token(logits, temperature=1.0):
    """
    Hardware: Sampling Unit
    Diagram Step: "Pick next token using... Greedy or Sampling"
    
    FPGA Implementation:
    - Greedy: Tree comparator to find Max Index.
    - Sampling: Random Number Generator (LFSR) + CDF Comparator.
    
    Args:
        logits: List of logit values [vocab_size]
        temperature: Sampling temperature (0.0 = greedy, >0.0 = sampling)
        
    Returns:
        Integer token ID
    """
    # 1. Apply Temperature
    if temperature == 0.0:
        # Greedy Decoding (Argmax)
        max_val = -float('inf')
        max_idx = -1
        for i, val in enumerate(logits):
            if val > max_val:
                max_val = val
                max_idx = i
        return max_idx
    else:
        # Sampling (Softmax + Random Choice)
        # Apply temp
        scaled_logits = [l / temperature for l in logits]
        probs = softmax_kernel(scaled_logits)
        
        # Simple cumulative distribution sampling
        r = random.random()
        cum_prob = 0.0
        for i, p in enumerate(probs):
            cum_prob += p
            if r < cum_prob:
                return i
        return len(probs) - 1


# ==========================================
# 🚀 EXTENDED ATTENTION LAYER (DECODE CAPABLE)
# ==========================================

class DecodeSelfAttentionLayer(SelfAttentionLayer):
    """
    Extends your existing SelfAttentionLayer to support the Decode Stage.
    Inherits initialization and weight loading.
    """
    
    def forward_decode(self, x_token_embedding, apply_norm=False):
        """
        SINGLE TOKEN PASS (The "Decode" Step)
        
        Diagram Step: "Calculate Q, K, V for new token"
        Diagram Step: "Cache K, V... into existing KV cache"
        Diagram Step: "Attention(Q, K, V) = softmax(...) * V"
        
        Args:
            x_token_embedding: Single vector [Hidden_Dim] (NOT a sequence)
            apply_norm: Whether to apply RMS normalization before attention
            
        Returns:
            Single output vector [Hidden_Dim]
        """
        # Convert to list if needed
        if hasattr(x_token_embedding, 'tolist'):
            x_token_embedding = x_token_embedding.tolist()
        
        # --- STEP 1: INPUT NORM ---
        if apply_norm and self.norm1_w is not None:
            x_norm = rms_norm_kernel(x_token_embedding, self.norm1_w, eps=self.rms_norm_eps)
        else:
            x_norm = x_token_embedding
        
        # --- STEP 2: CALCULATE Q, K, V FOR NEW TOKEN ---
        # Hardware: Matrix-Vector Multiplication
        # Input is 1 vector. Weights are matrices.
        q_new = mat_vec_mul_with_bias(self.w_q, x_norm, self.b_q)  # [Hidden]
        k_new = mat_vec_mul_with_bias(self.w_k, x_norm, self.b_k)   # [KV_Heads * Head_Dim]
        v_new = mat_vec_mul_with_bias(self.w_v, x_norm, self.b_v)   # [KV_Heads * Head_Dim]
        
        # --- STEP 2.5: APPLY ROTARY POSITION EMBEDDINGS ---
        # Position = current cache length (this is the new token's position)
        if self.k_cache is None or self.v_cache is None:
            raise ValueError("KV Cache is empty! Run prefill first.")
        
        current_position = len(self.k_cache[0])  # Cache length = position of new token
        
        # Apply RoPE to Q heads
        q_new_rope = []
        for q_head_idx in range(self.num_attention_heads):
            q_start = q_head_idx * self.head_dim
            q_end = q_start + self.head_dim
            q_head = q_new[q_start:q_end]
            q_head_rope = apply_rotary_pos_emb(q_head, current_position, self.head_dim, rope_theta=self.rope_theta)
            q_new_rope.extend(q_head_rope)
        q_new = q_new_rope
        
        # Apply RoPE to K heads
        k_new_rope = []
        for kv_head_idx in range(self.num_key_value_heads):
            kv_start = kv_head_idx * self.head_dim
            kv_end = kv_start + self.head_dim
            k_head = k_new[kv_start:kv_end]
            k_head_rope = apply_rotary_pos_emb(k_head, current_position, self.head_dim, rope_theta=self.rope_theta)
            k_new_rope.extend(k_head_rope)
        k_new = k_new_rope
        
        # --- STEP 3: KV CACHE UPDATE ---
        # We must split k_new/v_new into heads and append to self.k_cache
        
        # Update Cache Loop
        for kv_head_idx in range(self.num_key_value_heads):
            start = kv_head_idx * self.head_dim
            end = start + self.head_dim
            
            # Extract the slice for this head
            k_head_vec = k_new[start:end]
            v_head_vec = v_new[start:end]
            
            # APPEND to the existing list of vectors (The "Cache")
            # Hardware: Write to BRAM/DRAM at current_token_index
            self.k_cache[kv_head_idx].append(k_head_vec)
            self.v_cache[kv_head_idx].append(v_head_vec)
        
        # --- STEP 4: ATTENTION WITH CACHE ---
        # Q_new attends to ALL K (Past + Present)
        
        # Result container
        attn_output = [0.0] * self.hidden_size
        scale = 1.0 / math.sqrt(self.head_dim)
        
        # Loop over Query Heads
        for q_head_idx in range(self.num_attention_heads):
            kv_head_idx = q_head_idx // self.num_queries_per_kv
            
            q_start = q_head_idx * self.head_dim
            q_end = q_start + self.head_dim
            
            # Extract the single query vector for this head
            q_head_vec = q_new[q_start:q_end]
            
            # Retrieve the FULL cache for the corresponding KV head
            # List of vectors: [[head_dim], [head_dim], ... [head_dim]]
            K_cache_head = self.k_cache[kv_head_idx] 
            V_cache_head = self.v_cache[kv_head_idx]
            
            # 1. Calculate Scores against History
            # Hardware: Streaming Dot Product. 
            # Stream K vectors from memory, dot with q_head_vec register.
            scores = []
            for k_vec_past in K_cache_head:
                scores.append(vec_dot(q_head_vec, k_vec_past) * scale)
            
            # 2. Softmax
            probs = softmax_kernel(scores)
            
            # 3. Weighted Sum of Values
            # Hardware: Accumulator. 
            # Stream V vectors, scale by prob, add to accumulator.
            head_context = [0.0] * self.head_dim
            for t, v_vec_past in enumerate(V_cache_head):
                # Optimization: Skip zero probs
                if probs[t] == 0.0: 
                    continue
                
                weighted_v = vec_scale(v_vec_past, probs[t])
                head_context = vec_add(head_context, weighted_v)
            
            # 4. Concatenate result
            for i in range(self.head_dim):
                attn_output[q_start + i] = head_context[i]
        
        # --- STEP 5: OUTPUT PROJECTION ---
        post_attn = mat_vec_mul_with_bias(self.w_o, attn_output, self.b_o)
        
        # --- STEP 6: RESIDUAL 1 ---
        x_resid = vec_add(x_token_embedding, post_attn)
        
        # --- STEP 7: FFN (SwiGLU) ---
        # Norm
        x_norm2 = rms_norm_kernel(x_resid, self.norm2_w, eps=self.rms_norm_eps)
        
        # Projections
        gate = mat_vec_mul_with_bias(self.w_gate, x_norm2, self.b_gate)
        up = mat_vec_mul_with_bias(self.w_up, x_norm2, self.b_up)
        
        # Activation
        act_gate = silu_kernel(gate)
        mlp_hidden = vec_mul(act_gate, up)
        
        # Down Projection
        mlp_out = mat_vec_mul_with_bias(self.w_down, mlp_hidden, self.b_down)
        
        # Final Residual
        final_out = vec_add(x_resid, mlp_out)
        
        return final_out


# ==========================================
# 🔄 THE GENERATION LOOP
# ==========================================

def generate(model_dir, prompt_text, max_new_tokens=10, temperature=0.7, num_layers=None):
    """
    Full Pipeline: Tokenizer -> Prefill -> Decode Loop -> Detokenizer
    
    Args:
        model_dir: Path to model directory (e.g., "../Qwen2.5-0.5B")
        prompt_text: Input prompt string
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 = greedy, >0.0 = sampling)
        num_layers: Number of transformer layers (if None, loads from config)
        
    Returns:
        List of token IDs (prompt + generated tokens)
    """
    # 1. Initialize
    tokenizer = SimpleBPETokenizer(model_dir=model_dir)
    embedding_layer = EmbeddingLayer(model_dir=model_dir)
    
    # Load config to get number of layers
    import json
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    if num_layers is None:
        num_layers = config.get("num_hidden_layers", 24)
    
    # Initialize our NEW Decode-capable layers
    layers = [DecodeSelfAttentionLayer(layer_idx=i, model_dir=model_dir) for i in range(num_layers)]
    
    # Load Final Norm and Head weights (only once, at the end)
    # Use the method from any layer instance since they all share the same model_path
    final_norm_w, lm_head_w = layers[0].get_lm_head_w_and_final_norm_w()
    
    # 2. Tokenize
    input_tokens = tokenizer.encode(prompt_text)
    current_tokens = list(input_tokens)  # Copy
    print(f"Token IDs: {input_tokens}")
    
    # ==========================================
    # 🟢 STAGE 1: PREFILL (Process all Prompt tokens)
    # ==========================================
    # Embed
    embeddings_seq = embedding_layer.forward(input_tokens)
    
    # Print embedding output (last token, first 10 values)
    embeddings_np = embeddings_seq if isinstance(embeddings_seq, np.ndarray) else np.array(embeddings_seq)
    last_token_embed = embeddings_np[-1]
    print(f"\nEmbedding (shape={embeddings_np.shape}):")
    for j in range(min(10, len(last_token_embed))):
        print(f"  [{j}] = {last_token_embed[j]:.8f}")
    
    # Convert numpy array to list of lists if needed
    if hasattr(embeddings_seq, 'tolist'):
        embeddings_seq = embeddings_seq.tolist()
    
    # Run all layers (Prefill Mode)
    # This POPULATES the KV CACHE inside each layer
    x_out_seq = embeddings_seq
    for i, layer in enumerate(layers):
        x_out_seq = layer.forward(x_out_seq, apply_norm=True)
        
        # Print layer output (last token, first 10 values)
        x_out_np = np.array(x_out_seq) if not isinstance(x_out_seq, np.ndarray) else x_out_seq
        last_token_out = x_out_np[-1]
        print(f"\nLayer {i} (shape={x_out_np.shape}):")
        for j in range(min(10, len(last_token_out))):
            print(f"  [{j}] = {last_token_out[j]:.8f}")
    
    # Get the last vector to predict the first NEW token
    last_vector = x_out_seq[-1]
    
    # Final Norm (use epsilon from first layer)
    rms_eps = layers[0].rms_norm_eps
    last_vector = rms_norm_kernel(last_vector, final_norm_w, eps=rms_eps)
    
    # Logits & Sample
    # Optimization: We only need dot product for the last token
    logits = mat_vec_mul(lm_head_w, last_vector) 
    next_token_id = sample_next_token(logits, temperature=temperature if temperature > 0 else 0.0)
    
    print(f"\nGenerated token: {next_token_id} ('{tokenizer.decoder.get(next_token_id)}')")
    current_tokens.append(next_token_id)
    
    # ==========================================
    # 🔵 STAGE 2: DECODE LOOP (Token-by-Token)
    # ==========================================
    for i in range(max_new_tokens):
        # Print KV cache BEFORE processing the new token (to match test_qwen_model.py behavior)
        print(f"\n=== KV Cache at Decode Iteration {i} ===")
        for layer_idx, layer in enumerate(layers):
            if layer.k_cache is not None and layer.v_cache is not None:
                # Get cache length (same for all heads)
                seq_len = len(layer.k_cache[0]) if len(layer.k_cache) > 0 else 0
                num_kv_heads = len(layer.k_cache)
                head_dim = len(layer.k_cache[0][0]) if seq_len > 0 else 0
                
                print(f"Layer {layer_idx}:")
                print(f"  K cache shape: [{num_kv_heads} heads, {seq_len} tokens, {head_dim} dims]")
                print(f"  V cache shape: [{num_kv_heads} heads, {seq_len} tokens, {head_dim} dims]")
                print(f"  K cache length (seq_len): {seq_len}")
                
                # Print first few values of the last token's K and V for first 2 heads
                if seq_len > 0:
                    last_token_idx = seq_len - 1
                    for head_idx in range(min(2, num_kv_heads)):
                        k_head_last = layer.k_cache[head_idx][last_token_idx]
                        v_head_last = layer.v_cache[head_idx][last_token_idx]
                        print(f"    Head {head_idx} (last token):")
                        print(f"      K first 5 values: {[f'{x:.8f}' for x in k_head_last[:5]]}")
                        print(f"      V first 5 values: {[f'{x:.8f}' for x in v_head_last[:5]]}")
        
        # 1. Embed ONLY the new token
        # Input is [1] list, output is numpy array [1][Hidden]
        # We extract the single vector [0]
        next_embed_seq = embedding_layer.forward([next_token_id])
        
        # Convert to list if needed
        if hasattr(next_embed_seq, 'tolist'):
            next_embed_seq = next_embed_seq.tolist()
        
        next_embed_vec = next_embed_seq[0]
        
        # 2. Run all layers (Decode Mode)
        # Uses the 'forward_decode' method we wrote above
        # This updates the cache inside each layer and attends to history
        x_out_vec = next_embed_vec
        for layer in layers:
            x_out_vec = layer.forward_decode(x_out_vec, apply_norm=True)
        
        # 3. Final Norm
        x_out_vec = rms_norm_kernel(x_out_vec, final_norm_w, eps=rms_eps)
        
        # 4. Logits (LM Head)
        # Hardware: Large Matrix-Vector Multiply
        logits = mat_vec_mul(lm_head_w, x_out_vec)
        
        # 5. Sample
        next_token_id = sample_next_token(logits, temperature=temperature)
        
        # 6. Append & Check EOS
        current_tokens.append(next_token_id)
        token_str = tokenizer.decoder.get(next_token_id, f"<unk:{next_token_id}>")
        print(f"\nGenerated token: {next_token_id} ('{token_str}')")
        
        # Check for EOS token (Qwen uses 151643 for <|endoftext|>)
        eos_token_id = config.get("eos_token_id", 151643)
        if next_token_id == eos_token_id:
            break
    
    return current_tokens


if __name__ == "__main__":
    # Point this to your actual model directory
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "Qwen2.5-0.5B")
    
    if os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
        # Test with a simple prompt
        prompt = input("Enter a prompt: ")
        generate(MODEL_DIR, prompt, max_new_tokens=5, temperature=0.0)
    else:
        print(f"Error: Model not found at {MODEL_DIR}")

