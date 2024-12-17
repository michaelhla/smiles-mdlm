import torch
from dataloader import get_tokenizer, get_dataset
from molmetrics import MoleculeMetrics  # Assuming this is the correct import path
from main import generate_samples  # Assuming this is the correct import path
import hydra
from omegaconf import OmegaConf
from omegaconf import OmegaConf
from utils import get_logger
from smiles_tokenizer import SMILESTokenizer
from tqdm import tqdm  # Import tqdm for the progress bar
from transformers import BertTokenizer, BertModel

# Load the configuration
config_path = 'configs/config.yaml'  # Path to your config file
config = OmegaConf.load(config_path)

# Update the config with the checkpoint path
config.eval.checkpoint_path = '/root/smiles-mdlm/outputs/chebi/2024.12.16/215831/checkpoints/best.ckpt'  # Update with your checkpoint path
config.mode = 'sample_eval'  # Set mode to sample_eval

config.noise = OmegaConf.load('configs/noise/loglinear.yaml')
config.data = OmegaConf.load('configs/data/chebi.yaml')

# Initialize logger and tokenizer
logger = get_logger(__name__)  

tokenizer = SMILESTokenizer()
tokenizer.load_vocabulary('/root/smiles-mdlm/smiles_vocab.txt')

text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the validation dataset
valid_set = get_dataset(
    config.data.valid,
    text_tokenizer,
    wrap=config.data.wrap,
    mode='validation',
    cache_dir=config.data.cache_dir,
    block_size=config.model.length
)

# Initialize MoleculeMetrics
metrics = MoleculeMetrics()

# Evaluate the model on the validation dataset
batch_size = 32
text_embeddings_list = []
target_smiles_list = []

for batch in tqdm(valid_set, desc="Evaluating"):
    text_embeddings_list.append(batch['text_embeddings'])
    target_smiles_list.append(batch['input_ids'])

    if len(text_embeddings_list) == batch_size:
        text_embeddings = torch.stack(text_embeddings_list).to('cuda')
        smiles_samples = generate_samples(config, logger, tokenizer, text_embeddings)

        # Update metrics
        for i, generated_smiles in enumerate(smiles_samples):
            target_smiles = text_tokenizer.decode(target_smiles_list[i], skip_special_tokens=True)
            metrics.update(generated_smiles, target_smiles)

        # Reset lists for the next batch
        text_embeddings_list = []
        target_smiles_list = []

# Compute and print the final metrics
final_metrics = metrics.compute()
print("Evaluation Metrics:", final_metrics)
