from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from BNAF.bnaf import BNAF as _BNAF
from BNAF.bnaf import MaskedWeight, Permutation, Sequential, Tanh

import flowtorch


class BNAFParams(flowtorch.Params):
    def __init__(self, flows=1, layers=3, hidden_dim=10, residual="gated"):
        super().__init__()
        self.flows = flows
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.residual = residual

    def _build(self, input_shape, param_shapes):
        flows = []
        n_dims = sum(input_shape)
        for f in range(self.flows):
            layers = []
            for _ in range(self.layers - 1):
                layers.append(
                    MaskedWeight(
                        n_dims * self.hidden_dim, n_dims * self.hidden_dim, dim=n_dims
                    )
                )
                layers.append(Tanh())

            flows.append(
                _BNAF(
                    *(
                        [
                            MaskedWeight(n_dims, n_dims * self.hidden_dim, dim=n_dims),
                            Tanh(),
                        ]
                        + layers
                        + [MaskedWeight(n_dims * self.hidden_dim, n_dims, dim=n_dims)]
                    ),
                    res=self.residual if f < self.flows - 1 else False
                )
            )

            if f < self.flows - 1:
                flows.append(Permutation(n_dims, "flip"))

        model = Sequential(*flows)
        modules = nn.ModuleList([model])
        return modules, None

    def _forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        modules: Optional[nn.ModuleList] = None,
    ) -> Optional[Sequence[torch.Tensor]]:
        assert isinstance(modules, nn.ModuleList)
        return next(iter(modules)).forward(x)


class BNAF(flowtorch.Bijector):
    event_dim = 1

    def __init__(
        self,
        param_fn: Optional[flowtorch.Params] = None,
    ):
        if not param_fn:
            param_fn = BNAFParams()
        assert isinstance(param_fn, BNAFParams)
        super().__init__(param_fn=param_fn)

    def _forward(
        self, x: torch.Tensor, params: Optional[flowtorch.ParamsModule]
    ) -> torch.Tensor:
        assert isinstance(params, flowtorch.ParamsModule)
        return params(x)[0]

    def _inverse(
        self, y: torch.Tensor, params: Optional[flowtorch.ParamsModule]
    ) -> torch.Tensor:
        raise NotImplementedError()

    def _log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor, params: Optional[flowtorch.ParamsModule]
    ) -> torch.Tensor:
        assert isinstance(params, flowtorch.ParamsModule)
        return params(x)[1]