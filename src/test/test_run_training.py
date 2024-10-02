"""Module to test that run_training works as expected."""

import unittest

import hydra
import omegaconf

import training.run_training as run_training
from common.config import ExperimentConfigSchema


class TestRunTraining(unittest.TestCase):
    """Tests for run_training function."""

    def test_run_training(self):
        """Test that run_training works as expected."""
        with hydra.initialize(
            version_base=None,
            config_path="config",
            job_name="run_training_unit_test",
        ):
            config = hydra.compose(config_name="unittest_training")
        config: ExperimentConfigSchema = omegaconf.OmegaConf.to_object(config)
        run_training.run_training(config)


if __name__ == "__main__":
    unittest.main()
