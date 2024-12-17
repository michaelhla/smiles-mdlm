import torch
import hydra
from omegaconf import OmegaConf
from main import generate_samples
from utils import get_logger
from dataloader import get_tokenizer
from smiles_tokenizer import SMILESTokenizer
from transformers import BertTokenizer, BertModel

# Load the configuration
config_path = 'configs/config.yaml'  # Path to your config file
config = OmegaConf.load(config_path)

# Update the config with the checkpoint path
config.eval.checkpoint_path = '/root/smiles-mdlm/outputs/chebi/2024.12.16/215831/checkpoints/best.ckpt'  # Update with your checkpoint path
config.mode = 'sample_eval'  # Set mode to sample_eval

config.noise = OmegaConf.load('configs/noise/loglinear.yaml')

# Initialize logger and tokenizer
logger = get_logger(__name__)  

tokenizer = SMILESTokenizer()
tokenizer.load_vocabulary('/root/smiles-mdlm/smiles_vocab.txt')

# Load pre-trained BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Define a function to get the text embedding of the CLS token
def get_text_embedding(text_prompt):
    inputs = bert_tokenizer(text_prompt, return_tensors='pt', padding=True, truncation=True)
    outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token is at index 0
    return cls_embedding

# Example usage
text_prompt = "The molecule is an epoxy(hydroxy)icosatrienoate that is the conjugate base of 11 hydroxy-(14R,15S)-epoxy-(5Z,8Z,12E)-icosatrienoic acid, obtained by deprotonation of the carboxy group; major species at pH 7.3. It is a conjugate base of an 11 hydroxy-(14R,15S)-epoxy-(5Z,8Z,12E)-icosatrienoic acid."
text_embedding = get_text_embedding(text_prompt)

text_embeddings = torch.stack([text_embedding]).to('cuda')

# Generate samples
smiles_samples = generate_samples(config, logger, tokenizer, text_embeddings)

# # Print the generated samples
# print("Generated Samples:")
# for sample in smiles_samples:
#     print(sample)   

print(smiles_samples[0])