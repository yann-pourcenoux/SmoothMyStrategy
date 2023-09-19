# Use the official PyTorch base image
FROM pytorch/pytorch:latest

# Install git
RUN apt update -y
RUN apt install -y git

# Install the package
COPY setup.py .
RUN pip install -e .

# Add .local/bin to PATH
RUN export PATH="$PATH:/home/vscode/.local/bin"

# Add a non root user
RUN adduser vscod
