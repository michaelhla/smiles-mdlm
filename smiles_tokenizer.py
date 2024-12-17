import regex
import torch
from typing import List, Union
from collections import Counter
import os
from datasets import load_dataset


class SMILESTokenizer:
    def __init__(self, max_len: int = 256, min_freq: int = 1):
        self.max_len = max_len
        self.min_freq = min_freq
        
        # Regex pattern for tokenization
        self.pattern = r"\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|\d|\(|\)|\."
        self.regex = regex.compile(self.pattern)
        
        # Special tokens with UNK token added
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[SOS]': 2,
            '[EOS]': 3,
        }
        
        # Initialize vocabulary with special tokens
        self.token_to_id = None
        self.id_to_token = None
        self.vocab_size = None
        self.is_trained = False
        self.pad_token_id = 0
        self.eos_token_id = 3
        
    def train(self, data_dir: str = None, smiles_list: List[str] = None):
        """Train tokenizer on SMILES data to build vocabulary"""
        assert data_dir is not None or smiles_list is not None, "Either data_dir or smiles_list must be provided"
        
        # Collect all tokens and their frequencies
        token_freq = Counter()
        
        if data_dir is not None:
            # Read SMILES from files in directory
            for filename in os.listdir(data_dir):
                if filename.endswith('.txt'):
                    with open(os.path.join(data_dir, filename), 'r') as f:
                        for line in f:
                            smiles = line.strip().split('\t')[1] if '\t' in line else line.strip()
                            tokens = self.tokenize(smiles)
                            token_freq.update(tokens)
        
        if smiles_list is not None:
            # Process provided SMILES list
            for smiles in smiles_list:
                tokens = self.tokenize(smiles)
                token_freq.update(tokens)
        
        # Initialize vocabulary with special tokens
        self.token_to_id = self.special_tokens.copy()
        current_id = len(self.special_tokens)
        
        # Add tokens that meet minimum frequency requirement
        for token, freq in token_freq.items():
            if freq >= self.min_freq and token not in self.token_to_id:
                self.token_to_id[token] = current_id
                current_id += 1
        
        # Create reverse mapping
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)
        self.is_trained = True
        
        return self
    
    def save_vocabulary(self, save_path: str):
        """Save vocabulary to file"""
        assert self.is_trained, "Tokenizer must be trained before saving vocabulary"
        with open(save_path, 'w') as f:
            for token, idx in sorted(self.token_to_id.items(), key=lambda x: x[1]):
                f.write(f"{token}\t{idx}\n")
    
    def load_vocabulary(self, load_path: str):
        """Load vocabulary from file"""
        self.token_to_id = {}
        with open(load_path, 'r') as f:
            for line in f:
                token, idx = line.strip().split('\t')
                self.token_to_id[token] = int(idx)
        
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)
        self.is_trained = True
        return self
    
    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize SMILES string into list of tokens"""
        return self.regex.findall(smiles)
    
    def encode_one(self, smiles: str) -> List[int]:
        """Encode single SMILES string to list of token IDs"""
        assert self.is_trained, "Tokenizer must be trained before encoding"
        
        tokens = self.tokenize(smiles)
        
        # Convert tokens to IDs with SOS and EOS
        ids = [self.special_tokens['[SOS]']]
        for token in tokens:
            # Use UNK token for unknown tokens
            ids.append(self.token_to_id.get(token, self.special_tokens['[UNK]']))
        ids.append(self.special_tokens['[EOS]'])
        
        # Pad or truncate to max_len
        if len(ids) < self.max_len:
            ids.extend([self.special_tokens['[PAD]']] * (self.max_len - len(ids)))
        else:
            ids = ids[:self.max_len-1] + [self.special_tokens['[EOS]']]
        
        return ids
    
    def __call__(self, smiles: Union[str, List[str]]) -> List[List[int]]:
        """Encode one or more SMILES strings"""
        if isinstance(smiles, str):
            return self.encode_one(smiles)
        
        return [self.encode_one(s) for s in smiles]
    
    def decode_one(self, ids: Union[torch.Tensor, List[int]]) -> str:
        """Decode single tensor or list of token IDs back to SMILES string"""
        assert self.is_trained, "Tokenizer must be trained before decoding"
        
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        tokens = []
        for id in ids:
            token = self.id_to_token[id]
            if token in ['[PAD]', '[SOS]', '[EOS]']:
                continue
            if token == '[UNK]':
                tokens.append('?')  # Replace UNK with ? in output
            else:
                tokens.append(token)
        return ''.join(tokens)
    
    def decode(self, ids: Union[torch.Tensor, List[List[int]]]) -> Union[str, List[str]]:
        """Decode tensor(s) or list(s) of token IDs back to SMILES string(s)"""
        if isinstance(ids, torch.Tensor):
            if ids.dim() == 1:
                return self.decode_one(ids)
            return [self.decode_one(id_tensor) for id_tensor in ids]
        
        if isinstance(ids[0], int):
            return self.decode_one(ids)
        return [self.decode_one(id_list) for id_list in ids]
    
    def __len__(self) -> int:
        assert self.is_trained, "Tokenizer must be trained before getting length"
        return self.vocab_size

# Example usage
if __name__ == "__main__":
    # Load the ChEBI dataset
    cache_dir = '/root/smiles-mdlm/cache/chebi'
    mode = 'train'
    dataset = load_dataset("liupf/chEBI-20-MM")

    # Initialize the SMILES tokenizer
    smiles_tokenizer = SMILESTokenizer(max_len=256, min_freq=1)

    # Extract SMILES strings from the dataset
    train_smiles = dataset[mode]['SMILES']

    # Remove white space in the train_smiles
    train_smiles = [smiles.replace(" ", "") for smiles in train_smiles]

    # Train the tokenizer on the extracted SMILES strings
    smiles_tokenizer.train(smiles_list=train_smiles)

    # Save the trained vocabulary
    smiles_tokenizer.save_vocabulary('/root/smiles-mdlm/smiles_vocab.txt')

    # Test tokenization with unknown tokens
    test_smiles = [
        'C(C(=O)O)[NH3+]',  # Known SMILES
        'C[Sc]C(=O)O',      # Contains unknown token [Sc]
    ]

    # Test encoding and decoding
    for smiles in test_smiles:
        print(f"\nOriginal: {smiles}")
        tokens = smiles_tokenizer.tokenize(smiles)
        print(f"Tokens: {tokens}")
        encoded = smiles_tokenizer(smiles)
        print('encoded:', encoded)
        decoded = smiles_tokenizer.decode(encoded)
        print(f"Decoded: {decoded}")