#!/usr/bin/env python3
"""
Test script for Qwen2.5-0.5B model
Tests the model's text generation capabilities using Hugging Face transformers.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)


def load_model(model_dir):
    """Load the Qwen model and tokenizer from local directory."""
    # Check if model directory exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Use float16 for efficiency
        device_map="auto"  # Automatically handle device placement
    )
    
    return model, tokenizer


def inspect_model_internals(model, tokenizer, prompt, max_new_tokens=50, temperature=0.0, save_outputs=False):
    """
    Inspect intermediate values in the model for comparison with custom implementation.
    Extracts: tokens, embeddings, hidden states, logits, attention weights.
    """
    device = next(model.parameters()).device
    
    # ==========================================
    # STEP 1: TOKENIZATION
    # ==========================================
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    token_ids = input_ids[0].cpu().tolist()
    
    # Generate position_ids if needed
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
    
    # Print tokenization result
    print("=== Tokenization ===")
    print(token_ids)
    print("====")
    
    # ==========================================
    # STEP 2: EMBEDDINGS
    # ==========================================
    with torch.no_grad():
        # Get embedding layer
        embeddings = model.model.embed_tokens(input_ids)
        embedding_vectors = embeddings[0].cpu().numpy()  # [seq_len, hidden_size]
    
    # Print embedding result (first 10 values only)
    print("\n=== Embedding ===")
    embedding_list = embedding_vectors.tolist()
    # Print first 10 values of each token's embedding
    print([[vec[i] for i in range(min(10, len(vec)))] for vec in embedding_list])
    print("====")
    
    # ==========================================
    # STEP 3: HIDDEN STATES & KV CACHE (Layer by Layer)
    # ==========================================
    hidden_states = []
    kv_caches = []  # Store KV cache for each layer
    
    with torch.no_grad():
        # Use the model's forward pass with output_hidden_states=True
        # This is more reliable than manually forwarding through layers
        outputs = model.model(
            input_ids,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract all hidden states
        # outputs.hidden_states is a tuple of (embedding + all layers)
        # Shape: tuple of [batch_size, seq_len, hidden_size]
        all_hidden_states = outputs.hidden_states
        
        # Convert to numpy and store
        for i, h_state in enumerate(all_hidden_states):
            h_state_np = h_state[0].cpu().numpy()  # [seq_len, hidden_size]
            hidden_states.append(h_state_np)
        
        # Now extract KV cache for each layer by manually computing Q, K, V
        # We'll use the hidden state input to each layer
        for i, layer in enumerate(model.model.layers):
            try:
                # Get the hidden state that goes INTO this layer
                # hidden_states[0] is embedding output, which goes into layer 0
                # hidden_states[i] is the output of layer i-1, which goes into layer i
                layer_input = torch.from_numpy(hidden_states[i]).unsqueeze(0).to(device)
                
                # Apply input layer norm to get attention input
                if hasattr(layer, 'input_layernorm'):
                    attn_input = layer.input_layernorm(layer_input)
                else:
                    attn_input = layer_input
                
                # Compute K, V from attention module
                if hasattr(layer, 'self_attn'):
                    attn_module = layer.self_attn
                    # Get K and V projections
                    if hasattr(attn_module, 'k_proj') and hasattr(attn_module, 'v_proj'):
                        k = attn_module.k_proj(attn_input)
                        v = attn_module.v_proj(attn_input)
                        
                        # Store K and V for KV cache
                        k_cache_np = k[0].cpu().numpy()  # [seq_len, num_kv_heads * head_dim]
                        v_cache_np = v[0].cpu().numpy()  # [seq_len, num_kv_heads * head_dim]
                        
                        kv_caches.append({
                            'k': k_cache_np,
                            'v': v_cache_np
                        })
                    else:
                        kv_caches.append(None)
                else:
                    kv_caches.append(None)
                    
            except Exception as e:
                kv_caches.append(None)
        
        # Store the final hidden state from outputs for next step
        hidden_state = outputs.last_hidden_state
    
    # ==========================================
    # STEP 4: FINAL NORM AND LOGITS
    # ==========================================
    with torch.no_grad():
        # Ensure we have a valid hidden_state
        if hidden_state is None or len(hidden_states) == 0:
            # Fallback: use model forward pass
            outputs = model.model(input_ids)
            hidden_state = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        
        # Apply final layer norm
        final_hidden = model.model.norm(hidden_state)
        final_hidden_np = final_hidden[0].cpu().numpy()
        
        # Get logits from LM head
        logits = model.lm_head(final_hidden)  # [seq_len, vocab_size]
        logits_np = logits[0].cpu().numpy()
    
    # ==========================================
    # STEP 5: GENERATION (Token by Token)
    # ==========================================
    generated_tokens = []
    generated_logits = []
    generated_embeddings = []
    
    with torch.no_grad():
        # Start with input
        current_ids = input_ids.clone()
        past_key_values = None  # Initialize KV cache
        
        for step in range(max_new_tokens):
            # Forward pass with KV cache
            outputs = model(current_ids, use_cache=True, past_key_values=past_key_values)
            next_token_logits = outputs.logits[0, -1, :]  # Last token logits
            next_token_logits_np = next_token_logits.cpu().numpy()
            generated_logits.append(next_token_logits_np)
            
            # Sample next token
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, 1).item()
            else:
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            
            generated_tokens.append(next_token_id)
            token_str = tokenizer.decode([next_token_id])
            
            # Print first token after prefill as Attention output
            if step == 0:
                print("\n=== Attention ===")
                print(next_token_id)
                print("=====")
            elif step == 1:
                print("\n=== Decode ===")
                print(f"{next_token_id} {token_str}")
            else:
                print(f"{next_token_id} {token_str}")
            
            # Get embedding for this token (for comparison)
            token_embedding = model.model.embed_tokens(torch.tensor([[next_token_id]], device=device))
            generated_embeddings.append(token_embedding[0, 0].cpu().numpy())
            
            # Check for EOS
            if next_token_id == tokenizer.eos_token_id:
                break
            
            # Update past_key_values for next iteration
            past_key_values = outputs.past_key_values
            
            # For decode stage, only pass the new token
            current_ids = torch.tensor([[next_token_id]], device=device)
    
    # Decode the final result
    full_token_ids = token_ids + generated_tokens
    result_text = tokenizer.decode(full_token_ids)
    print("\n=== Result ===")
    print(result_text)
    print("====")
    
    # ==========================================
    # STEP 6: SUMMARY & SAVE
    # ==========================================
    # Save outputs if requested
    if save_outputs:
        output_dir = Path(__file__).parent / "model_inspection_outputs"
        output_dir.mkdir(exist_ok=True)
        
        # Save as numpy arrays
        np.save(output_dir / "input_token_ids.npy", np.array(token_ids))
        np.save(output_dir / "input_embeddings.npy", embedding_vectors)
        np.save(output_dir / "final_hidden_state.npy", final_hidden_np)
        np.save(output_dir / "final_logits.npy", logits_np)
        np.save(output_dir / "generated_token_ids.npy", np.array(generated_tokens))
        np.save(output_dir / "generated_logits.npy", np.array(generated_logits))
        np.save(output_dir / "generated_embeddings.npy", np.array(generated_embeddings))
        
        # Save hidden states
        for i, h_state in enumerate(hidden_states):
            layer_name = 'embed' if i == 0 else f"layer_{i-1}"
            np.save(output_dir / f"hidden_state_{layer_name}.npy", h_state)
        
        # Save KV caches
        for i, kv_cache in enumerate(kv_caches):
            if kv_cache is not None:
                np.save(output_dir / f"kv_cache_layer_{i}_k.npy", kv_cache['k'])
                np.save(output_dir / f"kv_cache_layer_{i}_v.npy", kv_cache['v'])
        
        # Save metadata
        metadata = {
            "prompt": prompt,
            "input_token_ids": token_ids,
            "generated_token_ids": generated_tokens,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "hidden_size": embedding_vectors.shape[1],
            "num_layers": len(model.model.layers),
            "num_hidden_states": len(hidden_states),
            "num_kv_caches": sum(1 for kv in kv_caches if kv is not None),
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    return {
        "input_token_ids": token_ids,
        "input_embeddings": embedding_vectors,
        "hidden_states": hidden_states,  # Embeddings after each layer
        "kv_caches": kv_caches,  # KV cache for each layer
        "final_hidden_state": final_hidden_np,
        "final_logits": logits_np,
        "generated_token_ids": generated_tokens,
        "generated_logits": generated_logits,
        "generated_embeddings": generated_embeddings,
    }


def test_generation(model, tokenizer, prompt, max_new_tokens=50, temperature=0.0, top_p=0.9, inspect=False):
    """Test text generation with the model."""
    if inspect:
        return inspect_model_internals(model, tokenizer, prompt, max_new_tokens, temperature, save_outputs=True)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


def main():
    """Main test function."""
    # Get model directory (relative to script location)
    script_dir = Path(__file__).parent
    model_dir = script_dir.parent / "Qwen2.5-0.5B"
    model_dir = str(model_dir.resolve())
    
    # Load model
    try:
        model, tokenizer = load_model(model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Ask for prompt and inspection mode
    try:
        prompt = input("Prompt: ").strip()
        
        if not prompt:
            sys.exit(0)
        
        # Ask if user wants detailed inspection
        inspect_input = input("Inspect model internals? (y/n, default=n): ").strip().lower()
        inspect_mode = inspect_input == 'y'
        
        # Run inference
        test_generation(model, tokenizer, prompt, max_new_tokens=50, temperature=0.0, inspect=inspect_mode)
        
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

