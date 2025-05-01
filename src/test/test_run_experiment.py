"""Module to test the complete training, testing, and visualization pipeline."""

import unittest

import hydra
import omegaconf
import training.run_training as run_training

import evaluation.run_testing as run_testing
import visualization.visualize as visualize
from config import QuantExperimentConfigSchema, RLExperimentConfigSchema


class TestPipeline(unittest.TestCase):
    """Tests for the training, testing, and visualization pipeline."""

    def test_rl_pipeline(self):
        """Test that the entire RL pipeline runs as expected."""
        with hydra.initialize(
            version_base=None,
            config_path="config",
            job_name="run_rl_pipeline_test",
        ):
            config = hydra.compose(config_name="rl_pipeline")
        config: RLExperimentConfigSchema = omegaconf.OmegaConf.to_object(config)
        model = run_training.run_training(config)
        eval_df = run_testing.run_testing(config, model)
        visualize.visualize(data=eval_df)

    def test_quant_pipeline(self):
        """Test that the entire quant pipeline runs as expected."""
        with hydra.initialize(
            version_base=None,
            config_path="config",
            job_name="run_quant_pipeline_test",
        ):
            config = hydra.compose(config_name="quant_pipeline")
        config: QuantExperimentConfigSchema = omegaconf.OmegaConf.to_object(config)
        eval_df = run_testing.run_testing(config)
        visualize.visualize(data=eval_df)


if __name__ == "__main__":
    unittest.main()
