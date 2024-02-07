"""Module to run a training."""

# import os
# os.environ['MKL_THREADING_LAYER'] = 'GNU'


import hydra
import loguru
import numpy as np
import omegaconf
import torch

import agents
import data.loader
import data.preprocessing
import environment
import losses
import optimizers
import train
import utils
from config.schemas import ExperimentConfigSchema


@hydra.main(version_base=None, config_path="config", config_name="base_training")
def main(cfg: omegaconf.DictConfig):
    """Wrapper to start the training and interact with hydra."""
    config: ExperimentConfigSchema = omegaconf.OmegaConf.to_object(cfg)
    loguru.logger.info(
        "Running training with the config...\n" + omegaconf.OmegaConf.to_yaml(config)
    )
    return run_training(config)


def run_training(config: ExperimentConfigSchema):
    """Train an agent."""

    # Find device
    device = utils.get_device(config.rest.device)

    # Set seed
    torch.manual_seed(config.rest.seed)
    np.random.seed(config.rest.seed)

    # Create logger
    logger = utils.get_logger(config.logging)

    # Get environments
    preprocessed_data = data.preprocessing.preprocess_data(
        stock_df_iterator=data.loader.load_data(config.loading),
        config=config.preprocessing,
    )
    # Create environments
    train_env = environment.get_sac_environment(
        config=config.environment,
        num_tickers=len(config.loading.tickers),
        env_data=preprocessed_data,
        seed=config.rest.seed,
        device=device,
    )
    eval_env = environment.get_sac_environment(
        config=config.environment,
        num_tickers=len(config.loading.tickers),
        env_data=preprocessed_data,
        seed=config.rest.seed,
        device=device,
    )

    # Create agent
    model, exploration_policy = agents.make_sac_agent(
        train_env=train_env,
        eval_env=eval_env,
        config=config.agent,
        device=device,
    )
    # Create SAC loss
    loss_module, target_net_updater = losses.make_loss_module(
        model=model,
        config=config.loss,
    )

    # Create off-policy collector
    collector = utils.make_collector(
        train_env=train_env,
        actor_model_explore=exploration_policy,
        config=config.collector,
        device=device,
        seed=config.rest.seed,
    )

    # Create replay buffer
    replay_buffer = utils.make_replay_buffer(config=config.replay_buffer)

    # Create optimizers
    model_optimizers = optimizers.make_sac_optimizer(
        loss_module=loss_module,
        config=config.optimizer,
    )

    # Main loop
    train.train_and_eval(
        collector=collector,
        eval_env=eval_env,
        actor=exploration_policy,
        replay_buffer=replay_buffer,
        loss_module=loss_module,
        optimizers=model_optimizers,
        target_net_updater=target_net_updater,
        logger=logger,
        config=config,
    )


if __name__ == "__main__":
    main()
