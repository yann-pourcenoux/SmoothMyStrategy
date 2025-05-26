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
import environment.trading
import evaluation.analysis as analysis
import evaluation.evaluate as evaluate
import logger as logger
import rl.utils as utils
from config.base import BasePolicyConfigSchema
from config.quant import QuantPolicyConfigSchema
from config.rl import RLPolicyConfigSchema
from config.run import EvaluationRunConfigSchema
from environment.trading import TradingEnv
from quant.buy_everyday import BuySharesPolicy


@hydra.main(version_base=None, config_path="../cfg", config_name="evaluation")
def main(cfg: DictConfig):
    """Wrapper to start the testing and interact with hydra."""
    config: EvaluationRunConfigSchema = omegaconf.OmegaConf.to_object(cfg)
    loguru.logger.info(
        "Running training with the config...\n" + omegaconf.OmegaConf.to_yaml(config)
    )
    return run_testing(config)


def load_policy(policy_config: BasePolicyConfigSchema):
    # Load the model
    if isinstance(policy_config, RLPolicyConfigSchema):
        model = torch.load(policy_config.model_path, weights_only=False)
        return model[0]
    # Handle quant models
    elif isinstance(policy_config, QuantPolicyConfigSchema):
        # Create the quant algorithm based on configuration
        if policy_config.algorithm_name == "BuyOneShareEveryDay":
            return BuySharesPolicy()
        else:
            # Add more algorithm types here as needed
            raise ValueError(f"Unknown quant algorithm: {policy_config.algorithm_name}")
    else:
        raise ValueError("Unknown config type.")


def run_testing(
    config: EvaluationRunConfigSchema, model: torch.nn.Module | None = None
) -> pd.DataFrame:
    # Find device
    device = utils.get_device(config.run_parameters.device)

    # Set seed
    torch.manual_seed(config.run_parameters.seed)
    np.random.seed(config.run_parameters.seed)

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
        loading_config=config.evaluation.loading,
        preprocessing_config=config.evaluation.preprocessing,
    )

    eval_env = environment.trading.apply_transforms(
        env=TradingEnv(
            config=config.evaluation.environment,
            data_container=data_container,
            seed=config.run_parameters.seed,
            device=device,
        ),
    )

    if model is None:
        exploration_policy = load_policy(config.policy)
    else:
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
