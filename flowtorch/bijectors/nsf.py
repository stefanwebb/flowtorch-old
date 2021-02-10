from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nflows.transforms.autoregressive import (
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.splines import rational_quadratic

import flowtorch


class NSFParams(flowtorch.Params):
    def __init__(
        self,
        hidden_features=10,
        context_features=None,
        num_bins=10,
        tails="linear",
        tail_bound=10.0,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        min_bin_width=rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=rational_quadratic.DEFAULT_MIN_DERIVATIVE,
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        self.num_bins = num_bins
        self.tails = tails
        self.tail_bound = tail_bound
        self.num_blocks = num_blocks
        self.use_residual_blocks = use_residual_blocks
        self.random_mask = random_mask
        self.activation = activation
        self.dropout_probability = dropout_probability
        self.use_batch_norm = use_batch_norm
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

    def _build(self, input_shape, param_shapes):
        model = MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            sum(input_shape),
            self.hidden_features,
            self.context_features,
            self.num_bins,
            self.tails,
            self.tail_bound,
            self.num_blocks,
            self.use_residual_blocks,
            self.random_mask,
            self.activation,
            self.dropout_probability,
            self.use_batch_norm,
            self.min_bin_width,
            self.min_bin_height,
            self.min_derivative,
        )

        modules = nn.ModuleList([model])
        return modules, None

    def _forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        modules: Optional[nn.ModuleList] = None,
        mode: str = "forward",
    ) -> Optional[Sequence[torch.Tensor]]:
        assert isinstance(modules, nn.ModuleList)
        mod = next(iter(modules))
        if mode == "forward":
            return mod.forward(x)
        else:
            return mod.inverse(x)


class NSF(flowtorch.Bijector):
    event_dim = 1

    def __init__(
        self,
        param_fn: Optional[flowtorch.Params] = None,
    ):
        if not param_fn:
            param_fn = NSFParams()
        assert isinstance(param_fn, NSFParams)
        super().__init__(param_fn=param_fn)

    def _forward(
        self, x: torch.Tensor, params: Optional[flowtorch.ParamsModule]
    ) -> torch.Tensor:
        assert isinstance(params, flowtorch.ParamsModule)
        return params(x, mode="forward")[0]

    def _inverse(
        self, y: torch.Tensor, params: Optional[flowtorch.ParamsModule]
    ) -> torch.Tensor:
        return params(y, mode="backward")[0]

    def _log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor, params: Optional[flowtorch.ParamsModule]
    ) -> torch.Tensor:
        assert isinstance(params, flowtorch.ParamsModule)
        return params(x, mode="forward")[1]
