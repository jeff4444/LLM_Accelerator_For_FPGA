import json
import os

from safetensors.torch import load_file


class EmbeddingLayer:

    def __init__(self, model_path=None, config_path=None, model_dir=None):
        """
        INITIALIZATION
        
        Loads the massive weight matrix from the .safetensors file.
        Also loads config.json to verify dimensions.
        
        Args:
            model_path: Path to model.safetensors file (optional if model_dir is provided)
            config_path: Path to config.json file (optional if model_dir is provided)
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
        
        # Load config.json to get expected dimensions
        print(f"Loading config from {config_path}...")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.vocab_size = self.config.get("vocab_size")
        self.hidden_dim = self.config.get("hidden_size")
        
        print(f"Expected vocab_size: {self.vocab_size}, hidden_size: {self.hidden_dim}")
        
        print(f"Loading weights from {model_path}...")

        # 1. Load the safetensors file
        # This dictionary acts like a map of { "layer_name": TensorData }
        state_dict = load_file(model_path)

        # 2. Extract ONLY the embedding weights
        # For Qwen/Llama models, the key is usually 'model.embed_tokens.weight'
        # Shape: [Vocab Size, Hidden Size] -> e.g. [151936, 896] for Qwen-0.5B
        embedding_key = "model.embed_tokens.weight"
        
        if embedding_key not in state_dict:
            # Try alternative keys that might be used
            possible_keys = [k for k in state_dict.keys() if "embed" in k.lower()]
            if possible_keys:
                print(f"Warning: '{embedding_key}' not found. Available embedding keys: {possible_keys}")
                embedding_key = possible_keys[0]
                print(f"Using '{embedding_key}' instead.")
            else:
                raise KeyError(f"Could not find embedding weights in file! Available keys: {list(state_dict.keys())[:10]}...")
        
        weights_tensor = state_dict[embedding_key]
        
        # Convert torch tensor to numpy array for barebone implementation
        # Use detach() to remove from computation graph, convert to float32 (numpy supports this),
        # then convert to numpy array
        self.weights = weights_tensor.detach().float().numpy()
        
        # Verify dimensions match config
        actual_vocab_size, actual_hidden_dim = self.weights.shape
        if actual_vocab_size != self.vocab_size:
            print(f"Warning: Vocab size mismatch! Config: {self.vocab_size}, Actual: {actual_vocab_size}")
        if actual_hidden_dim != self.hidden_dim:
            print(f"Warning: Hidden dim mismatch! Config: {self.hidden_dim}, Actual: {actual_hidden_dim}")
        
        # Update to actual dimensions
        self.vocab_size = actual_vocab_size
        self.hidden_dim = actual_hidden_dim

        # Clean up memory (optional, but good practice in barebones implementations)
        del state_dict

        print(f"Embedding Matrix Loaded. Shape: {self.vocab_size} rows x {self.hidden_dim} cols")
        print(f"Total parameters: {self.vocab_size * self.hidden_dim:,}")

    def forward(self, token_ids):
        """
        THE LOOKUP OPERATION (Barebone Implementation)
        
        Input: List of integers [100, 52, 101]
        Output: numpy array of shape [Seq_Len, Hidden_Dim]
        
        Args:
            token_ids: List of token IDs (integers)
            
        Returns:
            Embedding array of shape [Seq_Len, Hidden_Dim] as numpy array
        """
        # Ensure token_ids is a list of integers
        if not isinstance(token_ids, list):
            token_ids = list(token_ids)
        
        # Perform the lookup: index into the weights matrix
        # This effectively does: [ self.weights[100], self.weights[52], ... ]
        embeddings = self.weights[token_ids, :]

        return embeddings


# --- DRIVER CODE (Full Pipeline: Text -> Tokens -> Embeddings) ---
if __name__ == "__main__":
    # Import the tokenizer from the same directory
    from tokenizer import SimpleBPETokenizer
    
    # Path to the Qwen2.5-0.5B model directory
    model_dir = os.path.join(os.path.dirname(__file__), "..", "Qwen2.5-0.5B")
    
    print("=" * 70)
    print("FULL PIPELINE: Text -> Tokenization -> Embedding")
    print("=" * 70)
    print()
    
    # Step 1: Initialize Tokenizer
    print("Step 1: Loading Tokenizer...")
    print("-" * 70)
    tokenizer = SimpleBPETokenizer(model_dir=model_dir)
    print(f"Tokenizer loaded: {len(tokenizer.encoder)} vocab items")
    print()
    
    # Step 2: Initialize Embedding Layer
    print("Step 2: Loading Embedding Layer...")
    print("-" * 70)
    embedding_layer = EmbeddingLayer(model_dir=model_dir)
    print()
    
    # Step 3: Get user input
    print("Step 3: Input Processing")
    print("-" * 70)
    user_input = input("Enter a string to tokenize and embed: ")
    print(f"Input String: '{user_input}'")
    print()
    
    # Step 4: Tokenization (Text -> Token IDs)
    print("Step 4: Tokenization")
    print("-" * 70)
    token_ids = tokenizer.encode(user_input)
    print(f"Token IDs: {token_ids}")
    print(f"Number of tokens: {len(token_ids)}")
    print()
    
    # Step 5: Embedding Lookup (Token IDs -> Embedding Vectors)
    print("Step 5: Embedding Lookup")
    print("-" * 70)
    embeddings = embedding_layer.forward(token_ids)
    
    print(f"\n--- Resulting Embedding Vectors (E^0) ---")
    print(f"Shape: {embeddings.shape}")
    print(f"(Sequence Length: {embeddings.shape[0]}, Hidden Dimension: {embeddings.shape[1]})")
    print()
    
    # Print first few values of each token's vector
    print("Sample embeddings (first 5 values of each token):")
    for i, token_id in enumerate(token_ids):
        token_str = tokenizer.decoder.get(token_id, f"<unk:{token_id}>")
        print(f"  Token {i} (ID: {token_id}, Text: '{token_str}'): {embeddings[i][:5].tolist()}")
    print()
    
    # Summary
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Input text: '{user_input}'")
    print(f"→ Tokenized to {len(token_ids)} tokens")
    print(f"→ Embedded to array of shape {embeddings.shape}")
    print(f"→ This array is ready for your accelerator's Prefill stage!")

