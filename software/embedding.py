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
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.vocab_size = self.config.get("vocab_size")
        self.hidden_dim = self.config.get("hidden_size")

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
                embedding_key = possible_keys[0]
            else:
                raise KeyError(f"Could not find embedding weights in file! Available keys: {list(state_dict.keys())[:10]}...")
        
        weights_tensor = state_dict[embedding_key]
        
        # Convert torch tensor to numpy array for barebone implementation
        # Use detach() to remove from computation graph, convert to float32 (numpy supports this),
        # then convert to numpy array
        self.weights = weights_tensor.detach().float().numpy()
        
        # Verify dimensions match config
        actual_vocab_size, actual_hidden_dim = self.weights.shape
        # Update to actual dimensions
        self.vocab_size = actual_vocab_size
        self.hidden_dim = actual_hidden_dim

        # Clean up memory (optional, but good practice in barebones implementations)
        del state_dict

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
    
    # Step 1: Initialize Tokenizer
    tokenizer = SimpleBPETokenizer(model_dir=model_dir)
    
    # Step 2: Initialize Embedding Layer
    embedding_layer = EmbeddingLayer(model_dir=model_dir)
    
    # Step 3: Get user input
    user_input = input("Enter a string to tokenize and embed: ")
    
    # Step 4: Tokenization (Text -> Token IDs)
    token_ids = tokenizer.encode(user_input)
    
    # Step 5: Embedding Lookup (Token IDs -> Embedding Vectors)
    embeddings = embedding_layer.forward(token_ids)

