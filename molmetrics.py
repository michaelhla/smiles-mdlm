from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import Levenshtein
from smiles_tokenizer import SMILESTokenizer
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


class MoleculeMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {
            'bleu': [],
            'exact_match': [],
            'levenshtein': [],
            'maccs_similarity': [],
            'validity': []
        }
    
    def update(self, generated_smiles, target_smiles):
        # BLEU score
        tokenizer = SMILESTokenizer()
        tokenizer.load_vocabulary('/root/smiles-mdlm/smiles_vocab.txt')
        gen_smiles = tokenizer.encode_one(generated_smiles)
        target_smiles = tokenizer.encode_one(target_smiles)
        gen_smiles = [str(token) for token in gen_smiles]
        target_smiles = [str(token) for token in target_smiles]
        bleu_score = sentence_bleu([target_smiles], gen_smiles)   
        print(bleu_score)
        self.metrics['bleu'].append(bleu_score)
        
        # Exact match
        exact = float(generated_smiles == target_smiles)
        self.metrics['exact_match'].append(exact)
        
        # Levenshtein
        lev = Levenshtein.distance(generated_smiles, target_smiles)
        self.metrics['levenshtein'].append(lev)
        
        # Validity
        try:
            mol = Chem.MolFromSmiles(generated_smiles)
            valid = mol is not None
        except Exception as e:
            valid = False
        self.metrics['validity'].append(float(valid))
        
        # MACCS similarity (if both molecules are valid)
        try:
            if valid and Chem.MolFromSmiles(target_smiles) is not None:
                gen_maccs = MACCSkeys.GenMACCSKeys(mol)
                target_maccs = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(target_smiles))
                similarity = DataStructs.TanimotoSimilarity(gen_maccs, target_maccs)
                self.metrics['maccs_similarity'].append(similarity)
        except Exception as e:
            pass
    
    def compute(self):
        return {
            k: np.mean(v) if v else 0.0 
            for k, v in self.metrics.items()
        }