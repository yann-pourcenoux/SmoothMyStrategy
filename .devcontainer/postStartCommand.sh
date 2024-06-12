# Add .local/bin to PATHm needed to find the packages
export PATH=$PATH:/home/vscode/.local/bin
# wandb login 52084ad3dd36b2a1f4181e861ebfdb83e2376c72

# Install the package and its dependencies
uv venv --python=3.10
source .venv/bin/activate
uv pip install -e .

# Install pre-commit hooks
pre-commit install