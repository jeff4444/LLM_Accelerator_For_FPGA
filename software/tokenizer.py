import json
import os


class SimpleBPETokenizer:

    def __init__(self, vocab_path=None, merges_path=None, model_dir=None):
        """
        INITIALIZATION PHASE
        
        This runs once when you load the model.
        It prepares the hash maps needed for the pipeline.
        
        Args:
            vocab_path: Path to vocab.json file (optional if model_dir is provided)
            merges_path: Path to merges.txt file (optional if model_dir is provided)
            model_dir: Path to model directory (e.g., "../Qwen2.5-0.5B")
        """
        self.encoder = {}  # Maps string token -> Integer ID (from vocab.json)
        self.bpe_ranks = {}  # Maps tuple (char, char) -> Priority Rank (from merges.txt)

        # Determine file paths
        if model_dir:
            vocab_path = os.path.join(model_dir, "vocab.json")
            merges_path = os.path.join(model_dir, "merges.txt")
        
        if vocab_path is None or merges_path is None:
            raise ValueError("Must provide either (vocab_path and merges_path) or model_dir")

        # --- 1. LOAD VOCABULARY (The Dictionary) ---
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.encoder = json.load(f)
        
        # Create reverse mapping for debugging (optional)
        self.decoder = {v: k for k, v in self.encoder.items()}

        # --- 2. LOAD MERGES (The Assembly Instructions) ---
        with open(merges_path, 'r', encoding='utf-8') as f:
            merges_data = f.read().strip().split('\n')

        # Parse merges into a dictionary for fast lookup
        # bpe_ranks will look like: {('Ġ', 'Ġ'): 0, ('ĠĠ', 'ĠĠ'): 1, ...}
        # The integer value is the 'rank'. Lower number = higher priority.
        rank = 0
        for line in merges_data:
            # Skip empty lines
            if not line.strip():
                continue
            
            # Split the line into parts (e.g., "Ġ Ġ" -> ["Ġ", "Ġ"])
            parts = line.strip().split()
            if len(parts) != 2:
                # Skip malformed lines (though there shouldn't be any)
                print(f"Warning: Malformed line: {line}")
                continue
            
            # Store as tuple for hashing
            parts_tuple = tuple(parts)
            self.bpe_ranks[parts_tuple] = rank
            rank += 1

    def bpe(self, token):
        """
        THE BPE ALGORITHM
        
        Repeatedly merges characters based on the rules in bpe_ranks.
        
        Args:
            token: Input string token (e.g., "Hello")
            
        Returns:
            Tuple of BPE subword tokens (e.g., ("Hello",) or ("He", "ll", "o"))
        """
        # 1. Split word into individual characters
        word = tuple(token)
        # Example: ("H", "e", "l", "l", "o")

        while True:
            # Find all adjacent pairs in the current word tuple
            # e.g. [('H','e'), ('e','l'), ('l','l'), ('l','o')]
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            if not pairs:
                break

            # 2. Find the pair with the lowest rank (highest priority)
            # We look up each pair in self.bpe_ranks.
            # If a pair isn't in ranks, we ignore it (float('inf')).
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))

            # If the best pair found isn't in our merge list, we are done.
            if bigram not in self.bpe_ranks:
                break

            # 3. Merge the pair!
            # We create a new list of tokens where the pair is combined.
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                # Check if we found the pair at this position
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)  # Merge: "H"+"e" -> "He"
                    i += 2  # Skip the next character since we just merged it
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            # Loop repeats. Next iteration 'word' might be ("He", "l", "l", "o")

        return word

    def encode(self, text):
        """
        PIPELINE STEP A: Tokenization
        
        Input: "Hello, how are you"
        Output: [100, 52, 101, 102, 103] (List of Integers)
        
        Args:
            text: Input string to tokenize
            
        Returns:
            List of integer token IDs
        """
        # 1. Pre-tokenization (Simple Version)
        # Real tokenizers use complex Regex here.
        # We will simply add the space token 'Ġ' and split by space.

        # In Qwen/RoBERTa, spaces are usually attached to the start of the word.
        # "Hello, how are you" -> ["Hello,", "Ġhow", "Ġare", "Ġyou"]
        words = []
        raw_words = text.split(' ')
        for i, w in enumerate(raw_words):
            if not w:  # Skip empty strings from multiple spaces
                continue
            if i == 0:
                words.append(w)
            else:
                words.append("Ġ" + w)  # Add special space character

        tokens_ids = []

        # 2. Run BPE on each word
        for word in words:
            # Run the BPE algorithm loop
            bpe_tokens = self.bpe(word)

            # 3. Map the resulting string tokens to Integers using vocab.json
            for token in bpe_tokens:
                if token in self.encoder:
                    tokens_ids.append(self.encoder[token])
                else:
                    # Handle unknown tokens (usually mapped to an <unk> ID)
                    # Qwen doesn't have unk_token, so we'll print a warning
                    print(f"Warning: Token '{token}' not found in vocab. Skipping.")

        return tokens_ids

    def decode(self, token_ids):
        """
        Decode a list of token IDs back to text.
        
        Args:
            token_ids: List of integer token IDs
            
        Returns:
            Decoded string
        """
        tokens = [self.decoder.get(token_id, f"<unk:{token_id}>") for token_id in token_ids]
        text = ''.join(tokens)
        # Replace Ġ with spaces (except at the beginning)
        if text.startswith('Ġ'):
            text = text[1:]
        text = text.replace('Ġ', ' ')
        return text


# --- DRIVER CODE FOR YOUR TESTING ---
if __name__ == "__main__":
    # Path to the Qwen2.5-0.5B model directory
    model_dir = os.path.join(os.path.dirname(__file__), "..", "Qwen2.5-0.5B")
    
    # 1. Initialize (Loads the JSON and TXT files)
    print(f"Loading tokenizer from: {model_dir}")
    tokenizer = SimpleBPETokenizer(model_dir=model_dir)

    print("--- Initialization Complete ---")
    print(f"Loaded {len(tokenizer.encoder)} vocab items.")
    print(f"Loaded {len(tokenizer.bpe_ranks)} merge rules.")
    print("-" * 50)

    # 2. Test Input String
    user_input = input("Enter a string: ")
    print(f"Input String: '{user_input}'")

    # 3. Run the Pipeline
    output_ids = tokenizer.encode(user_input)

    # 4. Show Result
    print(f"Output Tokens (IDs): {output_ids}")
    print(f"Number of tokens: {len(output_ids)}")
    
    # 5. Test decode
    decoded = tokenizer.decode(output_ids)
    print(f"Decoded back: '{decoded}'")