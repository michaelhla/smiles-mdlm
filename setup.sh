# Install Python 3.9 if not already present
apt-get update && apt-get install -y python3.9 python3.9-venv

# Create virtual environment
python3.9 -m venv venv

# Activate it
source venv/bin/activate

# Install requirements
pip install -r requirements-pip.txt

git config --global user.email michaelhla@college.harvard.edu
git config --global user.name michaelhla

echo "set up done, now run 'source venv/bin/activate' to activate the virtual environment"