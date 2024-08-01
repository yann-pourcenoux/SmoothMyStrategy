# Use the official PyTorch base image
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime AS base

# Install git
RUN apt-get update -y
RUN apt-get install -y git

# Install the package and its dependencies
COPY pyproject.toml /tmp/tmp-finance/pyproject.toml
RUN pip install /tmp/tmp-finance[dev]
RUN rm -rf /tmp/tmp-finance


FROM base AS devcontainer

# Add a non root user
RUN adduser vscode

# Install needed dependencies for mujoco
RUN apt-get install -y wget gcc libosmesa6-dev libgl1-mesa-glx libglfw3