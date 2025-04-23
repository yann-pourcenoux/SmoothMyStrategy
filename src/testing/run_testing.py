"""Module to run a testing."""

import hydra
import loguru
import numpy as np
import omegaconf
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig

import data.container
import data.preprocessing
import environments.trading
import testing.analysis as analysis
import training.evaluate as evaluate
import training.logger as logger
import training.utils as utils
from common.config import ExperimentConfigSchema
from environments.trading import TradingEnv


@hydra.main(version_base=None, config_path="../config", config_name="base_experiment")
def main(cfg: DictConfig):
    """Wrapper to start the testing and interact with hydra."""
    config: ExperimentConfigSchema = omegaconf.OmegaConf.to_object(cfg)
    loguru.logger.info(
        "Running training with the config...\n" + omegaconf.OmegaConf.to_yaml(config)
    )
    return run_testing(config)


def run_testing(
    config: ExperimentConfigSchema, model: torch.nn.Module | None = None
) -> pd.DataFrame:
    # Find device
    device = utils.get_device(config.device)

    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Initialize wandb
    wandb.init(
        project=config.logging.project,
        name=config.logging.experiment,
        config=config,
        dir=config.logging.logging_directory,
        mode="offline" if not config.logging.online else "online",
        tags=["testing"],
    )

    # Get environments
    data_container = data.container.DataContainer(
        loading_config=config.loading, preprocessing_config=config.preprocessing
    )

    eval_env = environments.trading.apply_transforms(
        env=TradingEnv(
            config=config.eval_environment,
            data_container=data_container,
            seed=config.seed,
            device=device,
        ),
    )
    # Load the model
    if model is None:
        model = torch.load(config.agent.model_path, weights_only=False)

    exploration_policy = model[0]

    # Compute metrics
    metrics_to_log, eval_df = evaluate.evaluate(
        eval_env,
        exploration_policy,
        config.evaluation,
    )

    logger.log_df(eval_df, "eval_df")

    analysis.log_report(eval_df["daily_returns"], output_path=wandb.run.dir)

    # Log the metrics
    wandb.log(metrics_to_log)

    wandb.finish()
    return eval_df


if __name__ == "__main__":
    run_testing()
