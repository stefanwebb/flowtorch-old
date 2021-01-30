# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import nn, optim
from torchkit import flows, nn as nn_

import flowtorch
import flowtorch.params
from flowtorch.utils import clamp_preserve_gradients


class NAFParams(flowtorch.Params):
    def __init__(
        self,
        flowtype=1,
        dim=43,
        dimh=64,
        num_hid_layers=2,
        act=nn.ELU(),
        num_flow_layers=2,
        num_ds_dim=16,
        num_ds_layers=1,
        lr=0.005,
        betas=(0.9, 0.999),
    ):
        super().__init__()

        if flowtype == 0:
            flow = flows.IAF

        elif flowtype == 1:
            flow = lambda **kwargs: flows.IAF_DSF(
                num_ds_dim=num_ds_dim, num_ds_layers=num_ds_layers, **kwargs
            )

        elif flowtype == 2:
            flow = lambda **kwargs: flows.IAF_DDSF(
                num_ds_dim=num_ds_dim, num_ds_layers=num_ds_layers, **kwargs
            )

        sequels = [
            nn_.SequentialFlow(
                flow(
                    dim=dim,
                    hid_dim=dimh,
                    context_dim=1,
                    num_layers=num_hid_layers + 1,
                    activation=act,
                    fixed_order=True,
                ),
                flows.FlipFlow(1),
            )
            for i in range(num_flow_layers)
        ] + [
            flows.LinearFlow(dim, 1),
        ]

        self.mdl = nn.Sequential(*sequels)

    def _build(
        self,
        input_shape: torch.Size,
        param_shapes: Sequence[torch.Size],
    ) -> Tuple[nn.ModuleList, Dict[str, Any]]:
        return (
            nn.ModuleList(
                [
                    self.mdl,
                ]
            ),
            {},
        )

    def _forward(
        self,
        y: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        modules: Optional[nn.ModuleList] = None,
    ) -> Sequence[torch.Tensor]:
        assert (
            isinstance(y, torch.Tensor)
            and context is None
            and isinstance(modules, nn.ModuleList)
        )
        n = y.shape[0]
        mdl = next(iter(modules))
        lgd = torch.zeros(n, device=y.device)
        context = torch.zeros((n, 1), device=y.device) + 2.0
        z, logdet, _ = mdl((y, lgd, context))
        return z, logdet


class NeuralAutoregressive(flowtorch.Bijector):
    event_dim = 1
    autoregressive = True
    default_param_fn = NAFParams()

    def __init__(
        self,
        param_fn: flowtorch.Params = default_param_fn,
    ):
        super().__init__(param_fn=param_fn)

    def _forward(
        self, x: torch.Tensor, params: Optional[flowtorch.ParamsModule]
    ) -> torch.Tensor:
        assert isinstance(params, flowtorch.ParamsModule)

        y_guess = nn.Parameter(x.clone())
        optimizer = optim.Adam([y_guess], lr=1e-1)

        for _ in range(100):
            optimizer.zero_grad()
            x_guess, _ = params(y_guess)
            loss = (x - x_guess).norm()
            loss.backward()
            optimizer.step()    
        
        return y_guess

    def _inverse(
        self, y: torch.Tensor, params: Optional[flowtorch.ParamsModule]
    ) -> torch.Tensor:
        assert isinstance(params, flowtorch.ParamsModule)
        z, _ = params(y)
        return z

    def _log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor, params: Optional[flowtorch.ParamsModule]
    ) -> torch.Tensor:
        assert isinstance(params, flowtorch.ParamsModule)
        _, logdet = params(y)
        return -logdet
