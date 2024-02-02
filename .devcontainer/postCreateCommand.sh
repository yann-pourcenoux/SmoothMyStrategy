set -eo pipefail

# Add .local/bin to PATH needed to find the packages
export PATH=$PATH:/home/vscode/.local/bin

# Update pip
python -m pip install --upgrade pip

# Install mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
mv mujoco210 /home/vscode/.mujoco/
rm -r mujoco210-linux-x86_64.tar.gz
pip install "cython<3"

# Install the package and its dependencies
pip install -e .

# Add this to avoid git errors
git config --global --add safe.directory /workspaces/Finance

# Install pre-commit hooks
pre-commit install
