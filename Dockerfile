# Use the official PyTorch base image from https://hub.docker.com/r/pytorch/pytorch/tags
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime AS base

# Install git
RUN apt-get update -y
RUN apt-get install -y git

# Install the package and its dependencies
COPY pyproject.toml /tmp/tmp-finance/pyproject.toml
RUN pip install /tmp/tmp-finance[dev,ubuntu]
RUN rm -rf /tmp/tmp-finance

FROM base AS devcontainer

# Add a non root user
RUN adduser vscode

# Install needed dependencies for mujoco
RUN apt-get install -y wget gcc libosmesa6-dev libgl1-mesa-glx libglfw3