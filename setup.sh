# Install Python 3.9 if not already present
apt-get update && apt-get install -y python3.9 python3.9-venv

# Create virtual environment
python3.9 -m venv venv

# Activate it
source venv/bin/activate

# Install requirements
pip install -r requirements-pip.txt