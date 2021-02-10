from typing import Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from torchkit import flows, nn as nn_, utils


import flowtorch


class NAFParams(flowtorch.Params):
    def __init__(
        self,
        flowtype: Union[Literal["IAF"], Union[Literal["DSF"], Literal["DDSF"]]] = "IAF",
        dimh=64,
        num_hid_layers=2,
        act=nn.ELU(),
        num_flow_layers=2,
        num_ds_dim=16,
        num_ds_layers=1,
    ):
        super().__init__()
        self.flowtype = flowtype
        self.dimh = dimh
        self.num_hid_layers = num_hid_layers
        self.act = act
        self.num_flow_layers = num_flow_layers
        self.num_ds_dim = num_ds_dim
        self.num_ds_layers = num_ds_layers

    def _build(self, input_shape, param_shapes):
        if self.flowtype == "IAF":
            flow = flows.IAF

        elif self.flowtype == "DSF":
            flow = lambda **kwargs: flows.IAF_DSF(
                num_ds_dim=self.num_ds_dim, num_ds_layers=self.num_ds_layers, **kwargs
            )

        elif self.flowtype == "DDSF":
            flow = lambda **kwargs: flows.IAF_DDSF(
                num_ds_dim=self.num_ds_dim, num_ds_layers=self.num_ds_layers, **kwargs
            )

        else:
            raise Exception("Unknown flow type for NAF")

        dim = sum(input_shape)
        sequels = [
            nn_.SequentialFlow(
                flow(
                    dim=dim,
                    hid_dim=self.dimh,
                    context_dim=1,
                    num_layers=self.num_hid_layers + 1,
                    activation=self.act,
                    fixed_order=True,
                ),
                flows.FlipFlow(1),
            )
            for i in range(self.num_flow_layers)
        ] + [
            flows.LinearFlow(dim, 1),
        ]
        model = nn.Sequential(*sequels)
        modules = nn.ModuleList([model])

        return modules, None

    def _forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        modules: Optional[nn.ModuleList] = None,
    ) -> Optional[Sequence[torch.Tensor]]:
        assert isinstance(modules, nn.ModuleList)
        mod = next(iter(modules))

        n = x.shape[0]
        context = torch.FloatTensor(n, 1).zero_().to(x.device) + 2.0
        lgd = torch.FloatTensor(n).zero_().to(x.device)

        z, logdet, _ = mod((x, lgd, context))
        return z, logdet


class NAF(flowtorch.Bijector):
    event_dim = 1

    def __init__(
        self,
        param_fn: Optional[flowtorch.Params] = None,
    ):
        if not param_fn:
            param_fn = NAFParams()
        assert isinstance(param_fn, NAFParams)
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