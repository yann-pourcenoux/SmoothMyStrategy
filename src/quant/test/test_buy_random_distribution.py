"""Unit tests for src/quant/buy_random_distribution.py."""

import torch

from quant.buy_random_distribution import BuyRandomDistributionModule


class TestBuyRandomDistributionModule:
    """Test class for BuyRandomDistributionModule."""

    def test_initialization(self):
        """Test that the module initializes correctly."""
        module = BuyRandomDistributionModule()
        assert isinstance(module, torch.nn.Module)

    def test_output_is_probability_distribution(self):
        """Test that output sums to 1 for each batch."""
        module = BuyRandomDistributionModule()

        # Test single batch
        num_shares_owned = torch.tensor([1.0, 2.0, 3.0])
        output = module.forward(num_shares_owned)
        assert torch.allclose(torch.sum(output), torch.tensor(1.0), atol=1e-6)

        # Test multiple batches
        num_shares_owned = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        output = module.forward(num_shares_owned)
        expected_sums = torch.ones(2)
        assert torch.allclose(torch.sum(output, dim=-1), expected_sums, atol=1e-6)

    def test_output_is_non_negative(self):
        """Test that all output values are non-negative."""
        module = BuyRandomDistributionModule()
        num_shares_owned = torch.tensor([1.0, 2.0, 3.0])
        output = module.forward(num_shares_owned)
        assert torch.all(output >= 0)

    def test_output_shape_matches_input(self):
        """Test that output shape matches input shape."""
        module = BuyRandomDistributionModule()

        # Test 1D tensor
        num_shares_owned = torch.tensor([1.0, 2.0, 3.0])
        output = module.forward(num_shares_owned)
        assert output.shape == num_shares_owned.shape

        # Test 2D tensor
        num_shares_owned = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        output = module.forward(num_shares_owned)
        assert output.shape == num_shares_owned.shape

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        module = BuyRandomDistributionModule()

        for batch_size in [1, 5, 10]:
            num_tickers = 3
            num_shares_owned = torch.ones((batch_size, num_tickers))
            output = module.forward(num_shares_owned)

            assert output.shape == (batch_size, num_tickers)
            assert torch.allclose(torch.sum(output, dim=-1), torch.ones(batch_size), atol=1e-6)

    def test_randomness(self):
        """Test that multiple calls produce different outputs."""
        module = BuyRandomDistributionModule()
        num_shares_owned = torch.tensor([1.0, 2.0, 3.0])

        outputs = []
        for _ in range(10):
            output = module.forward(num_shares_owned)
            outputs.append(output)

        # Check that not all outputs are identical
        all_same = True
        first_output = outputs[0]
        for output in outputs[1:]:
            if not torch.allclose(output, first_output, atol=1e-6):
                all_same = False
                break

        assert not all_same, "All outputs should not be identical due to randomness"
