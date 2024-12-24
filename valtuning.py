import torch
import torch.nn.functional as F
from smiles_tokenizer import SMILESTokenizer

class SMILESValidityLosses:
    def __init__(self, vocab=None):
        # Vocabulary mappings
        if vocab is None:
            self.tokenizer = SMILESTokenizer()
            self.tokenizer.load_vocabulary('/root/smiles-mdlm/smiles_vocab.txt')
            self.vocab = self.tokenizer.token_to_id
        else:
            self.vocab = vocab
        
        # Define special tokens
        self.open_brackets = ['(']
        self.close_brackets = [')']
        self.ring_numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.percent_ring_numbers = [f'%{i}' for i in range(10, 33)]
        
        # Valence rules {atom: [possible_valences]}
        self.valence_rules = {
            'C': [4],
            'N': [3, 5],
            'O': [2],
            'F': [1],
            'P': [3, 5],
            'S': [2, 4, 6],
            'Cl': [1],
            'Br': [1],
            'I': [1]
        }

    def bracket_loss(self, logits):
        """Loss for bracket matching"""
        probs = F.softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
        
        # Sum probabilities for all bracket types
        open_probs = sum([probs[:, :, self.vocab[b]] for b in self.open_brackets])
        close_probs = sum([probs[:, :, self.vocab[b]] for b in self.close_brackets])
        
        # Running counts
        open_count = torch.cumsum(open_probs, dim=1)
        close_count = torch.cumsum(close_probs, dim=1)
        
        # Penalties
        negative_balance = torch.relu(close_count - open_count)  # Can't close before opening
        final_imbalance = torch.abs(open_count[:, -1] - close_count[:, -1])  # Must close all
        running_imbalance = torch.mean(torch.abs(open_count - close_count), dim=1)  # Prefer balanced
        
        return torch.mean(negative_balance) + torch.mean(final_imbalance) + torch.mean(running_imbalance)

    def ring_number_loss(self, logits):
        """Loss for ring number consistency"""
        probs = F.softmax(logits, dim=-1)
        
        # For each ring number, count occurrences
        ring_losses = []
        for num in self.ring_numbers + self.percent_ring_numbers:
            num_probs = probs[:, :, self.vocab[num]]
            count = torch.sum(num_probs, dim=1)
            # Each number should appear exactly 0 or 2 times
            ring_losses.append(torch.min(
                torch.abs(count - 0.0),  # Cost for not using this number
                torch.abs(count - 2.0)   # Cost for using this number properly
            ))
        
        ring_loss = torch.stack(ring_losses, dim=1)
        return torch.mean(ring_loss)

    def valence_loss(self, logits):
        """Loss for atomic valence"""
        probs = F.softmax(logits, dim=-1)
        
        # Get bond probabilities
        single_bond_probs = probs[:, :, self.vocab['-']]
        double_bond_probs = probs[:, :, self.vocab['=']]
        triple_bond_probs = probs[:, :, self.vocab['#']]
        
        # Calculate expected number of bonds
        expected_bonds = (
            torch.cumsum(single_bond_probs, dim=1) + 
            2.0 * torch.cumsum(double_bond_probs, dim=1) + 
            3.0 * torch.cumsum(triple_bond_probs, dim=1)
        )
        
        # Check each atom type
        valence_violations = []
        for atom, valid_valences in self.valence_rules.items():
            atom_mask = (probs[:, :, self.vocab[atom]] > 0)
            if not atom_mask.any():
                continue
                
            # Calculate violation for each possible valence state
            violations = []
            for valence in valid_valences:
                violation = torch.abs(expected_bonds[atom_mask] - float(valence))
                violations.append(violation)
            
            # Take minimum violation (atom can be in any valid state)
            min_violation = torch.min(torch.stack(violations, dim=0), dim=0)[0]
            valence_violations.append(min_violation)
        
        if valence_violations:
            return torch.mean(torch.cat(valence_violations))
        return torch.tensor(0.0, device=logits.device)

    def combined_loss(self, logits, weights={'bracket': 1.0, 'ring': 1.0, 'valence': 0.25}):
        """Combine all validity losses"""
        losses = {
            'bracket': self.bracket_loss(logits),
            'ring': self.ring_number_loss(logits),
            'valence': self.valence_loss(logits)
        }
        
        total_loss = sum(w * losses[k] for k, w in weights.items())
        return total_loss, losses