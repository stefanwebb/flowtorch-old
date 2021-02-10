# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT
from typing import Optional, Sequence

import torch
import torch.nn as nn

import flowtorch


class BatchNormModule(nn.Module):
    "Module to encapsulate computation of batch norm parameters."

    def __init__(self, num_inputs, momentum, eps):
        super().__init__()
        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(num_inputs))
        self.register_buffer("running_var", torch.ones(num_inputs))

    def forward(self, x, is_new_batch):
        inputs = x
        log_gamma, beta, running_mean, running_var = (
            self.log_gamma,
            self.beta,
            self.running_mean,
            self.running_var,
        )
        if self.training:
            if is_new_batch:
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

        return mean, var, log_gamma, beta


class BatchNormParams(flowtorch.Params):
    "Dumb wrapper around `BatchNormModule` implementing the `Params` interface."

    autoregressive = False

    def __init__(self, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

    def _build(self, input_shape, _param_shapes):
        num_inputs = sum(input_shape)
        modules = nn.ModuleList([BatchNormModule(num_inputs, self.momentum, self.eps)])

        return modules, None

    def _forward(
        self,
        x: torch.Tensor,
        is_new_batch: bool,
        context: Optional[torch.Tensor] = None,
        modules: Optional[nn.ModuleList] = None,
    ) -> Optional[Sequence[torch.Tensor]]:
        assert isinstance(modules, nn.ModuleList)
        return next(iter(modules)).forward(x, is_new_batch)


class BatchNorm(flowtorch.Bijector):
    """
    Batch normalization from Real NVP.

    See (https://arxiv.org/abs/1605.08803)
    """

    event_dim = 1
    autoregressive = False

    def __init__(self):
        super().__init__(param_fn=BatchNormParams())

    def _forward(
        self, x: torch.Tensor, params: Optional["flowtorch.ParamsModule"]
    ) -> torch.Tensor:
        assert isinstance(params, flowtorch.ParamsModule)
        mean, var, log_gamma, beta = params(x, is_new_batch=True)
        x_hat = (x - mean) / var.sqrt()
        y = torch.exp(log_gamma) * x_hat + beta
        return y

    def _log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor, params
    ) -> torch.Tensor:
        assert isinstance(params, flowtorch.ParamsModule)
        _, var, log_gamma, _ = params(x, is_new_batch=False)
        return (log_gamma - 0.5 * torch.log(var)).sum(-1, keepdim=True)

    def _inverse(
        self, y: torch.Tensor, params: Optional["flowtorch.ParamsModule"]
    ) -> torch.Tensor:
        assert isinstance(params, flowtorch.ParamsModule)
        mean, var, log_gamma, beta = params(None, is_new_batch=False)
        x_hat = (y - beta) / torch.exp(log_gamma)
        x = x_hat * var.sqrt() + mean
        return x
