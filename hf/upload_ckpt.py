import subprocess
import sys

# Install a package dynamically
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install huggingface_hub
install_package("huggingface_hub")
from huggingface_hub import login, upload_file

# Log in to Hugging Face
login(token=keys.json['huggingface_token'])  # Enter your API token when prompted

# Upload the checkpoint
upload_file(
    path_or_fileobj="/root/smiles-mdlm/outputs/chebi/2024.12.16/215831/checkpoints/best.ckpt",
    path_in_repo="best.ckpt",
    repo_id="mhla/smiles-mdlm",
    commit_message="Upload best checkpoint"
)