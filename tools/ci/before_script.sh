# Install git in the docker image
apt-get update -y
apt-get install -y git

# Rebase branch
git fetch
git rebase origin/main

# Install and setup uv
pip install uv
uv venv --python=3.10 
source .venv/bin/activate


# Debugging
python --version
uv pip -V
uv pip list

# Install the package
uv pip install -e .
