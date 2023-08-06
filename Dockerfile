# Use the official PyTorch base image
FROM pytorch/pytorch:latest

# Install git
RUN apt update -y
RUN apt install -y git

# Install the package
RUN pip install -e .