# Finance

## Description

This project aims to create agents to manage investment portfolios.
This will be achieved by using Deep Learning and Reinforcement Learning using Pytorch.

## Setup

You can install the package by doing the following command:

```shell
pip install -e .
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
