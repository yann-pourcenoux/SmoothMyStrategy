# Finance

## Description

This project implements two distinct approaches to manage investment portfolios:

1. **Quantitative (Quant) Approach**: A traditional quantitative strategy that uses statistical methods and financial models for portfolio evaluation and optimization. This approach is currently used for evaluation runs and performance analysis.

2. **Reinforcement Learning (RL) Approach**: An advanced approach using Deep Learning and Reinforcement Learning with PyTorch to create autonomous trading agents that can learn optimal trading strategies through interaction with the market environment.

The project combines both methodologies to provide comprehensive portfolio management solutions, with quant methods serving as a baseline for evaluation and RL methods exploring more sophisticated trading strategies.

## Project Structure

```
finance/
├── src/                    # Source code
│   ├── cfg/               # Configuration files (YAML)
│   ├── config/            # Python configuration modules
│   ├── data/              # Data handling and processing
│   ├── environment/       # Trading environment definitions
│   ├── evaluation/        # Evaluation metrics and tools
│   ├── quant/            # Quantitative trading strategies
│   ├── rl/               # Reinforcement learning implementations
│   ├── test/             # Test suite
│   └── visualization/    # Visualization tools and dashboards
├── data/                  # Raw and processed data storage
├── outputs/              # Model outputs and results
├── tools/                # Utility scripts and tools
├── resources/            # Additional resources and documentation
├── .cursor/             # Cursor IDE configuration
├── .devcontainer/       # VS Code dev container configuration
```

### Key Directories

- **src/cfg/**: Contains YAML configuration files for different experiments and models
- **src/config/**: Python modules for configuration management
- **src/data/**: Data processing pipelines, data loaders, and feature engineering
- **src/environment/**: Trading environment implementations for RL training
- **src/evaluation/**: Performance metrics and evaluation tools
- **src/quant/**: Implementation of quantitative trading strategies
- **src/rl/**: Reinforcement learning models and training pipelines
- **src/visualization/**: Tools for visualizing results and analysis

### Development Directories

- **.cursor/**: Contains Cursor IDE specific configurations and settings

## Setup

### Install the package

You can install the package by doing the following command:

```shell
pip install -e .
```

For development purposes, use:

```shell
pip install -e ".[dev]"
```

### Download TorchRL examples

This is only needed to run TorchRL tutorials.

```shell
bash tools/install_utils/download_torchrl_implementations.sh
```

You can install the dependencies to run on mujoco env by doing:

```shell
source tools/install_utils/install_mujoco.sh
```

## Data

To download the data, run:

```shell
python src/data/download.py
```

## Train a model

To train a model, run:

```shell
run_training.py --config-name=xxx
```

## Test a model

To test a model, run:

```shell
run_testing --config-name=xxx
```

Don't forget top provide the `agent.model_path` element of the config.

## Visualization and analysis of test runs

Run

```shell
streamlit run src/visualization/main.py
```
