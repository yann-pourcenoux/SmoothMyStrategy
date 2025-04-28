# Quantitative Trading Models

This directory contains traditional quantitative trading algorithms that can be used for backtesting and evaluation.

## Available Models

1. **BuyOneShareEveryDay** - A simple algorithm that buys one share of each asset every day without regard to price or portfolio value.
2. **BuyOneShareEverydayTDM** - TensorDictModule implementation of the buy-one-share strategy.

## How to Use in Testing

### Running Tests with a Quant Model

To test a quantitative model:

```bash
# Navigate to the project root
cd /path/to/project

# Run the testing with the traditional implementation
python -m src.testing.run_testing --config-name quant_test

# Run the testing with the TensorDictModule implementation
python -m src.testing.run_testing --config-name tdm_test
```

### TensorDictModule Implementation

The TensorDictModule implementation has some advantages over the traditional approach:

1. **Direct Integration** - Works directly with the evaluation framework without requiring a wrapper
2. **Simplified Interface** - Uses a declarative approach to define input and output keys
3. **Better Compatibility** - Integrates seamlessly with TorchRL's ecosystem
4. **Performance** - Can be optimized with TorchRL's compile and optimization tools

Example usage:

```python
from quant.buy_everyday import BuyOneShareEverydayTDM
from tensordict import TensorDict
import torch

# Create a model
model = BuyOneShareEverydayTDM(scaling_factor=100.0)

# Create a sample input TensorDict
td = TensorDict({
    "adj_close": torch.ones(3, 4),  # 3 batch items, 4 tickers
}, batch_size=[3])

# Apply the model
result_td = model(td)

# The result contains the original inputs plus the action
print(result_td["action"])  # Shows the buy actions (1/100 for each ticker)
```

### Creating Your Own Quant Model

To create a new quantitative trading model:

1. Create a new Python file in the `src/quant` directory
2. You can either:
   - Implement a class that inherits from `TraditionalAlgorithm`
   - Implement a class that inherits from `TensorDictModule` (recommended)
3. Update the `run_testing.py` to support your new algorithm type

Example of a custom TensorDictModule model:

```python
# src/quant/my_strategy.py
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

# First, create a standard PyTorch module that processes the inputs
class MyStrategyModule(torch.nn.Module):
    def __init__(self, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor

    def forward(self, adj_close, ma_5):
        # Buy 1 share when price is below MA, sell 1 when above
        action = torch.where(
            adj_close < ma_5,
            torch.ones_like(adj_close),  # Buy
            -torch.ones_like(adj_close)  # Sell
        ) / self.scaling_factor
        return action

# Then wrap it with a TensorDictModule to handle the TensorDict integration
class MyStrategyTDM(TensorDictModule):
    """A custom trading strategy using TensorDictModule."""

    def __init__(self, scaling_factor=1.0):
        # Create the underlying module
        module = MyStrategyModule(scaling_factor)

        # Initialize the TensorDictModule with the module and keys
        super().__init__(
            module=module,
            in_keys=["adj_close", "moving_average_5"],
            out_keys=["action"]
        )
```

### Configuration

Create a config file based on the `tdm_test.yaml` template:

```yaml
# @package _global_
defaults:
  - quant_experiment
  - eval_environment: default_eval
  - _self_

agent:
  algorithm_name: MyStrategyTDM # Your algorithm name

# Other configuration parameters...
```

## Evaluation

Quant models are evaluated using the same metrics as reinforcement learning models, including:

- Sharpe Ratio
- Calmar Ratio
- Daily Returns
- Portfolio Value Over Time

The results are logged to WandB for easy visualization and comparison.
