from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

import residual_flows.lib.layers.base as base_layers
import residual_flows.lib.layers as layers


import flowtorch

ACTIVATION_FNS = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "elu": torch.nn.ELU,
    "selu": torch.nn.SELU,
    "fullsort": base_layers.FullSort,
    "maxmin": base_layers.MaxMin,
    "swish": base_layers.Swish,
    "lcube": base_layers.LipschitzCube,
}


class ResFlowParams(flowtorch.Params):
    def __init__(
        self,
        arch="iresnet",
        dims="128-128-128-128",
        act="swish",
        actnorm=False,
        batchnorm=False,
        # nblocks=100,
        nblocks=2,
        n_dist="geometric",
        n_power_series=None,
        exact_trace=False,
        n_samples=1,
        brute_force=False,
        vnorms="222222",
        learn_p=False,
        coeff=0.9,
        n_lipschitz_iters=5,
        atol=None,
        rtol=None,
    ):
        super().__init__()
        self.arch = arch
        self.dims = dims
        self.act = act
        self.actnorm = actnorm
        self.batchnorm = batchnorm
        self.nblocks = nblocks
        self.n_dist = n_dist
        self.n_power_series = n_power_series
        self.exact_trace = exact_trace
        self.brute_force = brute_force
        self.n_samples = n_samples
        self.vnorms = vnorms
        self.learn_p = learn_p
        self.coeff = coeff
        self.n_lipschitz_iters = n_lipschitz_iters
        self.atol = atol
        self.rtol = rtol

    def _build(self, input_shape, param_shapes):
        activation_fn = ACTIVATION_FNS[self.act]

        if self.arch == "iresnet":
            dims = [2] + list(map(int, self.dims.split("-"))) + [2]
            blocks = []
            if self.actnorm:
                blocks.append(residual_flows.lib.layers.ActNorm1d(2))
            for _ in range(self.nblocks):
                blocks.append(
                    layers.iResBlock(
                        self._build_nnet(dims, activation_fn),
                        n_dist=self.n_dist,
                        n_power_series=self.n_power_series,
                        exact_trace=self.exact_trace,
                        brute_force=self.brute_force,
                        n_samples=self.n_samples,
                        neumann_grad=False,
                        grad_in_forward=False,
                    )
                )
                if self.actnorm:
                    blocks.append(layers.ActNorm1d(2))
                if self.batchnorm:
                    blocks.append(layers.MovingBatchNorm1d(2))
            model = layers.SequentialFlow(blocks)
        elif self.arch == "realnvp":
            blocks = []
            for _ in range(self.nblocks):
                blocks.append(layers.CouplingBlock(2, swap=False))
                blocks.append(layers.CouplingBlock(2, swap=True))
                if self.actnorm:
                    blocks.append(layers.ActNorm1d(2))
                if self.batchnorm:
                    blocks.append(layers.MovingBatchNorm1d(2))
            model = layers.SequentialFlow(blocks)
        modules = nn.ModuleList([model])
        return modules, None

    def _build_nnet(self, dims, activation_fn=torch.nn.ReLU):
        nnet = []
        domains, codomains = self.parse_vnorms()
        if self.learn_p:
            if self.mixed:
                domains = [torch.nn.Parameter(torch.tensor(0.0)) for _ in domains]
            else:
                domains = [torch.nn.Parameter(torch.tensor(0.0))] * len(domains)
            codomains = domains[1:] + [domains[0]]
        for i, (in_dim, out_dim, domain, codomain) in enumerate(
            zip(dims[:-1], dims[1:], domains, codomains)
        ):
            nnet.append(activation_fn())
            nnet.append(
                base_layers.get_linear(
                    int(in_dim),
                    int(out_dim),
                    coeff=self.coeff,
                    n_iterations=self.n_lipschitz_iters,
                    atol=self.atol,
                    rtol=self.rtol,
                    domain=domain,
                    codomain=codomain,
                    zero_init=(out_dim == 2),
                )
            )
        return torch.nn.Sequential(*nnet)

    def parse_vnorms(self):
        ps = []
        for p in self.vnorms:
            if p == "f":
                ps.append(float("inf"))
            else:
                ps.append(float(p))
        return ps[:-1], ps[1:]

    def _forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        modules: Optional[nn.ModuleList] = None,
        mode: str = "forward",
    ) -> Optional[Sequence[torch.Tensor]]:
        assert isinstance(modules, nn.ModuleList)
        mod = next(iter(modules))
        zero = torch.zeros(x.shape[0], 1).to(x)
        z, delta_logp = mod(x, zero)
        delta_logp = delta_logp.squeeze()
        return z, delta_logp


class ResidualFlow(flowtorch.Bijector):
    event_dim = 1

    def __init__(
        self,
        param_fn: Optional[flowtorch.Params] = None,
    ):
        if not param_fn:
            param_fn = ResFlowParams()
        assert isinstance(param_fn, ResFlowParams)
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
