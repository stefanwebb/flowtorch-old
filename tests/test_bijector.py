# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT
import pytest
import torch
import torch.distributions as dist

import flowtorch
import flowtorch.bijectors
import flowtorch.params


def test_bijector_constructor():
    param_fn = flowtorch.params.DenseAutoregressive()
    b = flowtorch.bijectors.AffineAutoregressive(param_fn=param_fn)
    assert b is not None

@pytest.mark.parametrize("base_dist", [
    dist.Normal(0.0, 1.0),
    dist.Normal(torch.tensor(0.0), torch.tensor(1.0)),
    dist.Normal(torch.tensor([0., 0.]), torch.tensor([1., 1.])),
    dist.Independent(dist.Normal(torch.tensor([0., 0.]), torch.tensor([1., 1.])), 1),
])
def test_batch_event_shapes(base_dist):
    bij = flowtorch.bijectors.AffineAutoregressive(
        flowtorch.params.DenseAutoregressive()
    )
    tdist, _ = bij(base_dist)
    assert tdist.batch_shape == base_dist.batch_shape
    assert tdist.event_shape == base_dist.event_shape

# class TestClass:
#     def test_shapes(self):
#         """
#         Tests output shapes of bijector
#         """

#         assert "h" in x
