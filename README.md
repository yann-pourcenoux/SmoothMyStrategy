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

### Download TorchRL examples

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
python src/run_training.py --config-name=xxx
```

### Artificial data

It is just a sigmoid that is generated in the downloading script

You can run a training on it using the `debug` config, i.e using the `--config-name=debug`
argument.

## Coverage

[![Coverage Status](https://gitlab.com/yannpourcenoux/Finance/badges/main/coverage.svg)](https://gitlab.com/yannpourcenoux/Finance/-/pipelines)
