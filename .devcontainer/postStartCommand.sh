# Add .local/bin to PATH needed to find the packages
export PATH=$PATH:/home/vscode/.local/bin

# Install the package and its dependencies
uv venv --python=3.10
source .venv/bin/activate
uv pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
