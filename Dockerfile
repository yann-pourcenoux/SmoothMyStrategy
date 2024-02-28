# Use the official PyTorch base image
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Install git
RUN apt update -y
RUN apt install -y git

# Add a non root user
RUN adduser vscode

# Install needed dependencies for mujoco
RUN apt install -y wget gcc libosmesa6-dev libgl1-mesa-glx libglfw3