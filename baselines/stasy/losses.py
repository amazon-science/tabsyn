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

"""All functions related to loss computation and optimization.
"""

from pickle import FALSE
import torch
import torch.optim as optim
import numpy as np
from baselines.stasy.models import utils as mutils
from baselines.stasy.sde_lib import VESDE, VPSDE
import logging

def get_optimizer(config, params):
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum((step+1) / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()
  return optimize_fn


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None] * z

    score = score_fn(perturbed_data, t)
    
    if not likelihood_weighting:
      losses = torch.square(score * std[:, None] + z) 
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)

    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    # loss = torch.mean(losses)
    return losses

  return loss_fn


def get_smld_loss_fn(vesde, train, reduce_mean=False):
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    
    labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
    sigmas = smld_sigma_array.to(batch.device)[labels]
    noise = torch.randn_like(batch) * sigmas[:, None]
    perturbed_data = noise + batch
    score = model_fn(perturbed_data, labels)
    target = -noise / (sigmas ** 2)[:, None]
    losses = torch.square(score - target)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)
    perturbed_data = sqrt_alphas_cumprod[labels, None] * batch + \
                     sqrt_1m_alphas_cumprod[labels, None] * noise
    score = model_fn(perturbed_data, labels)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False, workdir=False, spl=True, writer=None, alpha0=None, beta0=None):
  if continuous:
    loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")


  def step_fn(state, batch):
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      losses = loss_fn(model, batch)

      if spl:
        nll = losses
        q_alpha = torch.tensor(alpha0 + torch.log( torch.tensor(1+ 0.0001718*state['step']* (1-alpha0), dtype=torch.float32) )).clamp_(max=1).to(nll.device)
        q_beta = torch.tensor(beta0 + torch.log( torch.tensor(1+ 0.0001718*state['step']* (1-beta0), dtype=torch.float32) )).clamp_(max=1).to(nll.device)
        logging.info(f"q_alpha: {q_alpha}, q_beta: {q_beta}")

        alpha = torch.quantile(nll, q_alpha) 
        beta = torch.quantile(nll, q_beta)
        assert alpha <= beta
        v = compute_v(nll, alpha, beta)
        loss = torch.mean(v*losses)
        
        logging.info(f"alpha: {alpha}, beta: {beta}")
        logging.info(f"1 samples: {torch.sum(v == 1)} / {len(v)}")
        logging.info(f"weighted samples: { torch.sum((v != 1) * (v != 0)  )} / {len(v)}")
        logging.info(f"0 samples: {torch.sum(v == 0)} / {len(v)}")

      
      else:
        loss = torch.mean(losses)

      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())


    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        losses, score = loss_fn(model, batch)
        ema.restore(model.parameters())
        loss = torch.mean(losses)

    return loss

  return step_fn


def min_max_scaling(factor, scale=(0, 1)):

  std = (factor - factor.min()) / (factor.max() - factor.min())
  new_min = torch.tensor(scale[0]) 
  new_max = torch.tensor(scale[1])
  return std * (new_max - new_min) + new_min


def compute_v(ll, alpha, beta):

  v = -torch.ones(ll.shape).to(ll.device)
  v[torch.gt(ll, beta)] = torch.tensor(0., device=v.device) 
  v[torch.le(ll, alpha)] = torch.tensor(1., device=v.device)

  if ll[torch.eq(v, -1)].shape[0] !=0 and ll[torch.eq(v, -1)].shape[0] !=1 :
        v[torch.eq(v, -1)] = min_max_scaling(ll[torch.eq(v, -1)], scale=(1, 0)).to(v.device)
  else:
        v[torch.eq(v, -1)] = torch.tensor(0.5, device=v.device)

  return v  