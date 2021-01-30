# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from typing import Optional, Sequence

import flowtorch


class BatchNormModule(nn.Module):
    def __init__(self, num_inputs):
        super(BatchNormModule, self).__init__()
        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, x):
        return (self.log_gamma, self.beta, self.running_mean, self.running_var)


class BatchNormParams(flowtorch.Params):
    autoregressive = False

    def _build(self, input_shape, param_shapes):
        num_inputs = sum(input_shape)
        modules = nn.ModuleList([BatchNormModule(num_inputs)])

        return modules, None

    def _forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        modules: Optional[nn.ModuleList] = None,
    ) -> Optional[Sequence[torch.Tensor]]:
        return next(iter(modules)).forward(None)


class BatchNorm(flowtorch.Bijector, nn.Module):
    """An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    event_dim = 0
    autoregressive = False

    def __init__(self, momentum=0.0, eps=1e-5):
        super(BatchNorm, self).__init__(param_fn=BatchNormParams())
        self.momentum = momentum
        self.eps = eps

    def _batch_norm(
        self,
        inputs: torch.Tensor,
        params: Sequence[torch.Tensor],
        mode: str = "direct",
    ):
        log_gamma, beta, running_mean, running_var = params
        if mode == "direct":
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (inputs - self.batch_mean).pow(2).mean(0) + self.eps

                running_mean.mul_(self.momentum)
                running_var.mul_(self.momentum)

                running_mean.add_(self.batch_mean.data * (1 - self.momentum))
                running_var.add_(self.batch_var.data * (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = running_mean
                var = running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(log_gamma) * x_hat + beta
            return y, (log_gamma - 0.5 * torch.log(var)).sum(-1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - beta) / torch.exp(log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-log_gamma + 0.5 * torch.log(var)).sum(-1, keepdim=True)

    def _forward(
        self, x: torch.Tensor, params: Optional["flowtorch.ParamsModule"]
    ) -> torch.Tensor:
        return self._batch_norm(x, params(x), mode="reverse")[0]

    def _log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor, params
    ) -> torch.Tensor:
        return self._batch_norm(x, params(x), mode="direct")[1]

    def _inverse(
        self, y: torch.Tensor, params: Optional["flowtorch.ParamsModule"]
    ) -> torch.Tensor:
        return self._batch_norm(y, params(y), mode="direct")[0]
