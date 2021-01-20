# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT
import pytest
import torch
import torch.distributions as dist

import flowtorch
import flowtorch.bijectors
import flowtorch.params


@pytest.fixture
def affine_ar_bijector():
    param_fn = flowtorch.params.DenseAutoregressive()
    b = flowtorch.bijectors.AffineAutoregressive(param_fn=param_fn)
    return b


def test_bijector_constructor(affine_ar_bijector):
    assert affine_ar_bijector is not None


def test_simple(affine_ar_bijector):
    base_dist = dist.Normal(0.0, 1.0)
    new_dist, _ = affine_ar_bijector(base_dist)
    assert new_dist.rsample((100,)).shape == (100, 1)


def test_batched_base_distribution(affine_ar_bijector):
    base_dist = dist.Normal(torch.zeros(1), torch.ones(1))
    new_dist, _ = affine_ar_bijector(base_dist)
    assert new_dist.rsample((100,)).shape == (100, 1)
