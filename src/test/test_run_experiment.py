"""Module to test the complete training, testing, and visualization pipeline."""

import unittest

import hydra
import omegaconf

import testing.run_testing as run_testing
import testing.visualize as visualize
import training.run_training as run_training
from common.config import ExperimentConfigSchema


class TestPipeline(unittest.TestCase):
    """Tests for the training, testing, and visualization pipeline."""

    def test_pipeline(self):
        """Test that the entire pipeline runs as expected."""
        with hydra.initialize(
            version_base=None,
            config_path="config",
            job_name="run_training_unit_test",
        ):
            config = hydra.compose(config_name="unittest_training")
        config: ExperimentConfigSchema = omegaconf.OmegaConf.to_object(config)
        model = run_training.run_training(config)
        eval_df = run_testing.run_testing(config, model)
        visualize.visualize(data=eval_df)


if __name__ == "__main__":
    unittest.main()
