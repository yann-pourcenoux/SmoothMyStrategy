"""Module to run a training."""

import math
from typing import Any, Dict

import hydra
import loguru
import numpy as np
import omegaconf
import torch
import tqdm
import wandb

import data.container
import data.preprocessing
import environments.trading
import models.agents as agents
import training.evaluate as evaluate
import training.logger as logger
import training.losses as losses
import training.optimizers as optimizers
import training.train as train
import training.utils as utils
from common.config import ExperimentConfigSchema
from environments.trading import TradingEnv


@hydra.main(version_base=None, config_path="../config", config_name="base_experiment")
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
        tags=["training"],
    )

    # Get environments
    data_container = data.container.DataContainer(
        loading_config=config.loading, preprocessing_config=config.preprocessing
    )

    # Create environments
    train_env = environments.trading.apply_transforms(
        env=TradingEnv(
            config=config.train_environment,
            data_container=data_container,
            seed=config.seed,
            device=device,
        ),
    )
    eval_env = environments.trading.apply_transforms(
        env=TradingEnv(
            config=config.eval_environment,
            data_container=data_container,
            seed=config.seed,
            device=device,
        ),
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
        seed=config.seed,
    )

    # Create replay buffer
    replay_buffer = utils.make_replay_buffer(config=config.replay_buffer)

    # Create optimizers
    model_optimizers = optimizers.make_sac_optimizer(
        loss_module=loss_module,
        config=config.optimizer,
    )

    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.optimizer.T_max,
            eta_min=config.optimizer.eta_min,
        )
        for optimizer in model_optimizers
    ]

    # Compile
    torch.compile(exploration_policy)
    torch.compile(model)
    torch.compile(loss_module)

    # Main loop
    for epoch in tqdm.tqdm(
        range(config.training.num_epochs),
        unit="epoch",
        leave=True,
        desc="Training status",
    ):
        metrics_to_log: Dict[str, Any] = {
            "train/learning_rate": schedulers[0].get_last_lr()
        }
        # Collect data
        metrics_to_log.update(
            utils.collect_data(
                collector,
                replay_buffer,
                num_steps_per_episode=math.ceil(
                    collector.env._num_time_steps
                    * config.train_environment.batch_size
                    / config.collector.frames_per_batch
                ),
            )
        )

        # Update the models
        metrics_to_log.update(
            train.train(
                config.training,
                replay_buffer,
                collector,
                loss_module,
                model_optimizers,
                target_net_updater,
                schedulers,
            )
        )

        # Evaluation
        metrics_to_log.update(
            evaluate.evaluate(
                eval_env,
                exploration_policy,
                config.evaluation,
            )[0]
        )

        # Log the metrics
        wandb.log(metrics_to_log, step=epoch)

    collector.shutdown()

    # Save the exploration policy to file
    logger.log_model(model)

    wandb.finish()

    return model


if __name__ == "__main__":
    main()
