set -eo pipefail

# Add .local/bin to PATH needed to find the packages
export PATH=$PATH:/home/vscode/.local/bin

# Add this to avoid git errors
git config --global --add safe.directory /workspaces/Finance

# Download the torch_rl examples
bash tools/install_utils/download_torchrl_implementations.sh

# Create useful folders
mkdir -p data
mkdir -p outputs
mkdir -p wandb