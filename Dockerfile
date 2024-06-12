# Use the official PyTorch base image
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Install git
RUN apt-get update -y
RUN apt-get install -y git

# Install pip
RUN apt-get install -y python3-pip

# Install uv
RUN pip3 install uv

# Add a non root user
RUN adduser vscode

# Install needed dependencies for mujoco
RUN apt-get install -y wget gcc libosmesa6-dev libgl1-mesa-glx libglfw3