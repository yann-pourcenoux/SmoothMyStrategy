# Finance

## Description

This project aims to create agents to manage investment portfolios.
This will be achieved by using Deep Learning and Reinforcement Learning using Pytorch.

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
