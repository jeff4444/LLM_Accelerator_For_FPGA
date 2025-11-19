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
    print("Error: Required packages not installed.")
    print("Please install: pip install transformers torch")
    print(f"Error: {e}")
    sys.exit(1)


def load_model(model_dir):
    """Load the Qwen model and tokenizer from local directory."""
    print(f"Loading model from: {model_dir}")
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Check for required files
    required_files = ["config.json", "model.safetensors", "tokenizer.json"]
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            print(f"Warning: {file} not found in {model_dir}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    # Load model
    print("Loading model (this may take a moment)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Use float16 for efficiency
        device_map="auto"  # Automatically handle device placement
    )
    
    print("✓ Model loaded successfully!")
    return model, tokenizer


def inspect_model_internals(model, tokenizer, prompt, max_new_tokens=50, temperature=0.0, save_outputs=False):
    """
    Inspect intermediate values in the model for comparison with custom implementation.
    Extracts: tokens, embeddings, hidden states, logits, attention weights.
    """
    print(f"\n{'='*60}")
    print("INSPECTING MODEL INTERNALS")
    print(f"{'='*60}")
    print(f"Prompt: '{prompt}'")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print(f"{'='*60}\n")
    
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
    
    print("=" * 60)
    print("1. TOKENIZATION")
    print("=" * 60)
    print(f"Input text: '{prompt}'")
    print(f"Token IDs: {token_ids}")
    print(f"Number of tokens: {len(token_ids)}")
    
    # Show token strings
    token_strings = [tokenizer.decode([tid]) for tid in token_ids]
    print(f"Token strings: {token_strings}")
    print()
    
    # ==========================================
    # STEP 2: EMBEDDINGS
    # ==========================================
    with torch.no_grad():
        # Get embedding layer
        embeddings = model.model.embed_tokens(input_ids)
        embedding_vectors = embeddings[0].cpu().numpy()  # [seq_len, hidden_size]
    
    print("=" * 60)
    print("2. EMBEDDINGS")
    print("=" * 60)
    print(f"Embedding shape: {embedding_vectors.shape}")
    print(f"Hidden size: {embedding_vectors.shape[1]}")
    print(f"First token embedding (first 10 values): {embedding_vectors[0][:10]}")
    print(f"Embedding stats - Mean: {embedding_vectors.mean():.6f}, Std: {embedding_vectors.std():.6f}")
    print()
    
    # ==========================================
    # STEP 3: HIDDEN STATES & KV CACHE (Layer by Layer)
    # ==========================================
    print("=" * 60)
    print("3. HIDDEN STATES & KV CACHE (Through Layers)")
    print("=" * 60)
    
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
        
        print(f"Total hidden states (including embedding): {len(all_hidden_states)}")
        
        # Convert to numpy and store
        for i, h_state in enumerate(all_hidden_states):
            h_state_np = h_state[0].cpu().numpy()  # [seq_len, hidden_size]
            hidden_states.append(h_state_np)
            
            # Print layer name and first 10 values
            layer_name = "Embedding" if i == 0 else f"Layer {i-1}"
            last_token_hidden = h_state_np[-1]  # [hidden_size]
            
            print(f"\n{layer_name} (shape={h_state_np.shape}):")
            for j in range(10):
                print(f"  [{j}] = {last_token_hidden[j]:.8f}")
        
        # Print intermediate values for Layer 0 for debugging
        if len(all_hidden_states) > 1:
            print("\n[DEBUG] Layer 0 intermediate values:")
            layer0 = model.model.layers[0]
            layer0_input = all_hidden_states[0]  # Embedding output
            
            # Apply input layernorm
            with torch.no_grad():
                norm_out = layer0.input_layernorm(layer0_input)
                print(f"  After norm (last token, first 10): {norm_out[0, -1, :10].tolist()}")
                
                # Apply Q projection
                q_out = layer0.self_attn.q_proj(norm_out)
                print(f"  After Q proj (last token, first 10): {q_out[0, -1, :10].tolist()}")
                
                # Apply K projection  
                k_out = layer0.self_attn.k_proj(norm_out)
                print(f"  After K proj (last token, first 10): {k_out[0, -1, :10].tolist()}")
        
        # Now extract KV cache for each layer by manually computing Q, K, V
        # We'll use the hidden state input to each layer
        print("\nExtracting KV caches...")
        
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
                        print(f"Layer {i} KV cache: K shape={k_cache_np.shape}, V shape={v_cache_np.shape}")
                    else:
                        kv_caches.append(None)
                        print(f"Layer {i}: No K/V projections found")
                else:
                    kv_caches.append(None)
                    print(f"Layer {i}: No self_attn module found")
                    
            except Exception as e:
                print(f"Error extracting KV cache for layer {i}: {e}")
                kv_caches.append(None)
        
        # Store the final hidden state from outputs for next step
        hidden_state = outputs.last_hidden_state
    
    print(f"\nTotal layers processed: {len(hidden_states) - 1}")  # -1 for embedding
    print(f"Layers with KV cache: {sum(1 for kv in kv_caches if kv is not None)}")
    print()
    
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
    
    print("=" * 60)
    print("4. FINAL NORM & LOGITS")
    print("=" * 60)
    print(f"Final hidden state shape: {final_hidden_np.shape}")
    print(f"Logits shape: {logits_np.shape}")
    print(f"Logits stats - Mean: {logits_np.mean():.6f}, Std: {logits_np.std():.6f}, "
          f"Min: {logits_np.min():.6f}, Max: {logits_np.max():.6f}")
    
    # Get top-k predictions for last token
    last_token_logits = logits_np[-1]
    top_k = 10
    top_k_indices = np.argsort(last_token_logits)[-top_k:][::-1]
    top_k_values = last_token_logits[top_k_indices]
    
    print(f"\nTop {top_k} predictions for last token:")
    for idx, (token_id, score) in enumerate(zip(top_k_indices, top_k_values)):
        token_str = tokenizer.decode([token_id])
        print(f"  {idx+1}. Token {token_id:6d} ({token_str:20s}): {score:8.4f}")
    print()
    
    # ==========================================
    # STEP 5: GENERATION (Token by Token)
    # ==========================================
    print("=" * 60)
    print("5. GENERATION (Token-by-Token)")
    print("=" * 60)
    
    generated_tokens = []
    generated_logits = []
    generated_embeddings = []
    
    with torch.no_grad():
        # Start with input
        current_ids = input_ids.clone()
        
        for step in range(max_new_tokens):
            # Forward pass
            outputs = model(current_ids, use_cache=False)
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
            print(f"Step {step+1}: Token {next_token_id:6d} ({token_str:20s}), "
                  f"logit={next_token_logits[next_token_id].item():8.4f}")
            
            # Get embedding for this token (for comparison)
            token_embedding = model.model.embed_tokens(torch.tensor([[next_token_id]], device=device))
            generated_embeddings.append(token_embedding[0, 0].cpu().numpy())
            
            # Check for EOS
            if next_token_id == tokenizer.eos_token_id:
                print("  -> EOS token detected")
                break
            
            # Append to current_ids for next iteration
            current_ids = torch.cat([current_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
    
    print()
    
    # ==========================================
    # STEP 6: SUMMARY & SAVE
    # ==========================================
    print("=" * 60)
    print("6. SUMMARY")
    print("=" * 60)
    print(f"Input tokens: {len(token_ids)}")
    print(f"Generated tokens: {len(generated_tokens)}")
    print(f"Total tokens: {len(token_ids) + len(generated_tokens)}")
    
    full_text = tokenizer.decode(token_ids + generated_tokens, skip_special_tokens=True)
    print(f"\nFull generated text:")
    print("-" * 60)
    print(full_text)
    print("-" * 60)
    print()
    
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
        
        print(f"Saved inspection outputs to: {output_dir}")
        print()
    
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
    
    print(f"\n{'='*60}")
    print("TESTING TEXT GENERATION")
    print(f"{'='*60}")
    print(f"Prompt: '{prompt}'")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Top-p: {top_p}")
    print(f"{'='*60}\n")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print(f"Input tokens: {inputs['input_ids'].shape[1]}")
    print(f"Device: {device}")
    print("\nGenerating...\n")
    
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
    
    print("Generated text:")
    print("-" * 60)
    print(generated_text)
    print("-" * 60)
    
    # Show token breakdown
    input_tokens = inputs['input_ids'].shape[1]
    output_tokens = outputs.shape[1]
    new_tokens = output_tokens - input_tokens
    
    print(f"\nToken count: {input_tokens} input + {new_tokens} generated = {output_tokens} total")
    
    return generated_text


def main():
    """Main test function."""
    # Get model directory (relative to script location)
    script_dir = Path(__file__).parent
    model_dir = script_dir.parent / "Qwen2.5-0.5B"
    model_dir = str(model_dir.resolve())
    
    print("Qwen2.5-0.5B Model Test Script")
    print("="*60)
    
    # Load model
    try:
        model, tokenizer = load_model(model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Ask for prompt and inspection mode
    print("\n" + "="*60)
    print("Enter a prompt for the model:")
    print("="*60)
    
    try:
        prompt = input("Prompt: ").strip()
        
        if not prompt:
            print("No prompt provided. Exiting.")
            sys.exit(0)
        
        # Ask if user wants detailed inspection
        inspect_input = input("Inspect model internals? (y/n, default=n): ").strip().lower()
        inspect_mode = inspect_input == 'y'
        
        # Run inference
        test_generation(model, tokenizer, prompt, max_new_tokens=50, temperature=0.0, inspect=inspect_mode)
        
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

