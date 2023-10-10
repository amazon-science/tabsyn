# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""

import torch
import numpy as np
from scipy import integrate
from models import utils as mutils


def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, t, eps):

    grad_fn_eps_list = []
    for epsilon in eps:
      with torch.enable_grad():
        x.requires_grad_(True)
        fn_eps = torch.sum(fn(x, t) * epsilon)
        grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]

      x.requires_grad_(False)
      grad_fn_eps_list.append(torch.sum(grad_fn_eps * epsilon, dim=tuple(range(1, len(x.shape)))))

    return torch.mean(torch.stack(grad_fn_eps_list), 0)

        
  return div_fn


def get_likelihood_fn(sde, inverse_scaler, hutchinson_type='Rademacher',
                      rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5):

  def drift_fn(model, x, t):
    """The drift function of the reverse-time SDE."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=True)
    # Probability flow ODE is a special case of Reverse SDE
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  def div_fn(model, x, t, noise):
    return get_div_fn(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)

  def likelihood_fn(model, data, eps_iters=1):
    with torch.no_grad():
      shape = data.shape
      if hutchinson_type == 'Gaussian':
        epsilon = [torch.randn_like(data) for k in range(eps_iters)]
      elif hutchinson_type == 'Rademacher':
        epsilon = [torch.randint_like(data, low=0, high=2).float() * 2 - 1. for k in range(eps_iters)]
      else:
        raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")
        
      def ode_func(t, x):
        sample = mutils.from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
        vec_t = torch.ones(sample.shape[0], device=sample.device) * t
        drift = mutils.to_flattened_numpy(drift_fn(model, sample, vec_t))
        logp_grad = mutils.to_flattened_numpy(div_fn(model, sample, vec_t, epsilon))
        return np.concatenate([drift, logp_grad], axis=0)

      init = np.concatenate([mutils.to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
      solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)

      nfe = solution.nfev
      zp = solution.y[:, -1]

      z = mutils.from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float64)
      delta_logp = mutils.from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float64)      
      prior_logp = sde.prior_logp(z).view(shape[0], -1).sum(1, keepdim=False)

      ll = prior_logp + delta_logp
      return ll, z, nfe

  return likelihood_fn