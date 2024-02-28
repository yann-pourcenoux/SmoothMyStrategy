"""Module to run a training."""

import math

import hydra
import loguru
import numpy as np
import omegaconf
import torch
import tqdm

import agents
import analysis
import data.container
import data.preprocessing
import environment
import evaluate
import logger
import losses
import optimizers
import train
import utils
from config.schemas import ExperimentConfigSchema


@hydra.main(version_base=None, config_path="config", config_name="base_experiment")
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
    wandb_logger = logger.get_logger(config.logging)

    # Get environments
    data_container = data.container.DataContainer(
        loading_config=config.loading, preprocessing_config=config.preprocessing
    )

    # Create environments
    train_env = environment.get_sac_environment(
        config=config.train_environment,
        data_container=data_container,
        seed=config.rest.seed,
        device=device,
    )
    eval_env = environment.get_sac_environment(
        config=config.eval_environment,
        data_container=data_container,
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
    for epoch in tqdm.tqdm(range(config.training.num_epochs), unit="epoch", leave=True):
        # Collect data
        metrics_to_log = utils.collect_data(
            collector,
            replay_buffer,
            num_steps_per_episode=math.ceil(
                collector.env._num_time_steps
                * config.train_environment.batch_size
                / config.collector.frames_per_batch
            ),
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
            )
        )

        # Evaluation
        metrics_to_log.update(
            evaluate.evaluate(
                eval_env,
                exploration_policy,
                config.evaluation,
            )
        )

        # Analysis
        metrics_to_log.update(analysis.analyse(config.evaluation.output_path))
        logger.log_metrics(wandb_logger, metrics_to_log, epoch)

    collector.shutdown()


if __name__ == "__main__":
    main()
