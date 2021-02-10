from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from UMNN.models.UMNN import UMNNMAFFlow


import flowtorch


class UMNNParams(flowtorch.Params):
    def __init__(
        self,
        device,
        nb_flow=1,
        hidden_derivative=[100, 100, 100, 100],
        hidden_embedding=[100, 100, 100, 100],
        embedding_s=10,
        nb_steps=20,
    ):
        super().__init__()
        self.device = device
        self.nb_flow = nb_flow
        self.hidden_derivative = hidden_derivative
        self.hidden_embedding = hidden_embedding
        self.embedding_s = embedding_s
        self.nb_steps = nb_steps

    def _build(self, input_shape, param_shapes):
        model = UMNNMAFFlow(
            device=self.device,
            nb_flow=self.nb_flow,
            nb_in=sum(input_shape),
            hidden_derivative=self.hidden_derivative,
            hidden_embedding=self.hidden_embedding,
            embedding_s=self.embedding_s,
            nb_steps=self.nb_steps,
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
            return mod.forward(x), mod.compute_log_jac(x).sum(-1)
        else:
            return mod.invert(x), -mod.compute_log_jac(x).sum(-1)


class UMNN(flowtorch.Bijector):
    event_dim = 1

    def __init__(
        self,
        device,
        param_fn: Optional[flowtorch.Params] = None,
    ):
        if not param_fn:
            param_fn = UMNNParams(device=device)
        assert isinstance(param_fn, UMNNParams)
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