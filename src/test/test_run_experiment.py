"""Module to test the complete training, testing, and visualization pipeline."""

import unittest

import hydra
import omegaconf

import evaluation.run_evaluation as run_evaluation
import rl.run_training as run_training
import visualization.visualize as visualize
from config.run import CalibrationRunConfigSchema, TrainingConfigRunSchema


class TestPipeline(unittest.TestCase):
    """Tests for the training, testing, and visualization pipeline."""

    def test_rl_pipeline(self):
        """Test that the entire RL pipeline runs as expected."""
        with hydra.initialize(
            version_base=None,
            config_path="cfg",
            job_name="run_rl_pipeline_test",
        ):
            config = hydra.compose(config_name="training")
        config: TrainingConfigRunSchema = omegaconf.OmegaConf.to_object(config)
        model = run_training.run_training(config)
        eval_df = run_evaluation.run_testing(config, model)
        visualize.visualize(data=eval_df)

    def test_quant_pipeline(self):
        """Test that the entire quant pipeline runs as expected."""
        with hydra.initialize(
            version_base=None,
            config_path="cfg",
            job_name="run_quant_pipeline_test",
        ):
            config = hydra.compose(config_name="calibration")
        config: CalibrationRunConfigSchema = omegaconf.OmegaConf.to_object(config)
        eval_df = run_evaluation.run_testing(config)
        visualize.visualize(data=eval_df)


if __name__ == "__main__":
    unittest.main()
