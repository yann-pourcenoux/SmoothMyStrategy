# Use the official PyTorch base image
FROM pytorch/pytorch:latest

# Install git
RUN apt update -y
RUN apt install -y git

# Install the package
COPY setup.py .
ADD src src
RUN pip install -e .

# Add a non root user
RUN adduser vscode

# Add .local/bin to PATH
RUN export PATH=$PATH:/home/vscode/.local/bin
