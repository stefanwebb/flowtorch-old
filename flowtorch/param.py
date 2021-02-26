# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class ParamsModuleList(torch.nn.Module):
    params_modules: nn.ModuleList

    def __init__(
        self,
        params_modules: Sequence["ParamsModule"],
    ) -> None:
        super().__init__()
        self.params_modules = nn.ModuleList(params_modules)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Sequence[Optional[Sequence[torch.Tensor]]]:
        return [p.forward(x, context=context) for p in self.params_modules]

    def __iter__(self):
        return iter(self.params_modules)

    def __call__(self):
        return self.params_modules

    def __len__(self):
        return len(self.params_modules)

    def __reversed__(self):
        return reversed(self.params_modules)


class ParamsModule(torch.nn.Module):
    """
    Encapsulates computation of flow parameters.

    `ParamsModule` is a `torch.nn.Module` which registers the state
    (`modules`, `buffers`) used in computing flow parameters
    and provides it to the associated `Params`'s `forward()` method.
    """

    params: "Params"
    mods: Optional[nn.ModuleList]

    def __init__(
        self,
        params: "Params",
        modules: Optional[nn.ModuleList] = None,
        buffers: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Associate `params` with it stateful `modules` and `buffers`."""
        super().__init__()
        self.params = params
        self.mods = modules

        if buffers is not None:
            for n, v in buffers.items():
                self.register_buffer(n, v)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Optional[Sequence[torch.Tensor]]:
        """Compute flow parameters using input `x` and state in this `ParamsModule`."""
        return self.params.forward(x, modules=self.mods, context=context)


class Params(object):
    """
    Deferred initialization of parameters.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        input_shape: torch.Size,
        param_shapes: Sequence[torch.Size],
        context_dims: int,
    ) -> ParamsModule:
        return ParamsModule(self, *self.build(input_shape, param_shapes, context_dims))

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        modules: Optional[nn.ModuleList] = None,
    ) -> Optional[Sequence[torch.Tensor]]:
        if context is None:
            context = torch.empty(0)
        if modules is None:
            modules = nn.ModuleList()
        return self._forward(x, context=context, modules=modules)

    def _forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        modules: Optional[nn.ModuleList] = None,
    ) -> Optional[Sequence[torch.Tensor]]:
        """
        Abstract method to ***
        """
        raise NotImplementedError

    def build(
        self,
        input_shape: torch.Size,
        param_shapes: Sequence[torch.Size],
        context_dims: int,
    ) -> Tuple[nn.ModuleList, Dict[str, torch.Tensor]]:
        """
        Build stateful `modules` and `buffers` for a `ParamsModule`.

        :param input_shape: shape of the input
        :param param_shapes: shapes of the parameters
        :returns: a tuple of the initialized `modules` and `buffers`
        """
        self.input_shape = input_shape
        self.param_shapes = param_shapes
        return self._build(input_shape, param_shapes, context_dims)

    def _build(
        self,
        input_shape: torch.Size,
        param_shapes: Sequence[torch.Size],
        context_dims: int,
    ) -> Tuple[nn.ModuleList, Dict[str, torch.Tensor]]:
        """
        Abstract method to ***
        """
        raise NotImplementedError
