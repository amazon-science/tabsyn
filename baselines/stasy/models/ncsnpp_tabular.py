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

from torch.nn.functional import embedding
from . import utils, layers, layerspp
import torch.nn as nn
import torch

get_act = layers.get_act
default_initializer = layers.default_init


NONLINEARITIES = {
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "lrelu": nn.LeakyReLU(negative_slope=0.2),
    "swish": nn.SiLU(),
    "tanh": nn.Tanh(),
    "softplus": nn.Softplus(),
}


@utils.register_model(name='ncsnpp_tabular')
class NCSNpp(nn.Module):
  """NCSN++ model"""

  def __init__(self, config):
    super().__init__()
    base_layer = {
      "ignore": layers.IgnoreLinear,
      "squash": layers.SquashLinear,
      "concat": layers.ConcatLinear,
      "concat_v2": layers.ConcatLinear_v2,
      "concatsquash": layers.ConcatSquashLinear,
      "blend": layers.BlendLinear,
      "concatcoord": layers.ConcatLinear,
    }

    self.config = config
    self.act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))
    self.hidden_dims = config.model.hidden_dims 

    self.nf = nf = config.model.nf

    self.conditional = conditional = config.model.conditional 
    self.embedding_type = embedding_type = config.model.embedding_type.lower()

    modules = []
    if embedding_type == 'fourier':
      assert config.training.continuous, "Fourier features are only used for continuous training."

      modules.append(layerspp.GaussianFourierProjection(
        embedding_size=nf, scale=config.model.fourier_scale
      ))
      embed_dim = 2 * nf

    elif embedding_type == 'positional':
      embed_dim = nf

    else:
      pass

    if conditional:
      modules.append(nn.Linear(embed_dim, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)

    dim = config.data.image_size
    for item in list(config.model.hidden_dims):
      modules += [
          base_layer[config.model.layer_type](dim, item)
      ]
      dim += item
      modules.append(NONLINEARITIES[config.model.activation])

    modules.append(nn.Linear(dim, config.data.image_size))
    self.all_modules = nn.ModuleList(modules)

  def forward(self, x, time_cond):
    modules = self.all_modules 
    m_idx = 0
    if self.embedding_type == 'fourier':
      used_sigmas = time_cond
      temb = modules[m_idx](torch.log(used_sigmas))
      m_idx += 1

    elif self.embedding_type == 'positional':
      timesteps = time_cond
      used_sigmas = self.sigmas[time_cond.long()]
      temb = layers.get_timestep_embedding(time_cond, self.nf)

    else:
      pass

    if self.conditional:
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None

    temb = x
    for _ in range(len(self.hidden_dims)):
      temb1 = modules[m_idx](t=time_cond, x=temb)
      temb = torch.cat([temb1, temb], dim=1)
      m_idx += 1
      temb = modules[m_idx](temb) 
      m_idx += 1

    h = modules[m_idx](temb)

    if self.config.model.scale_by_sigma:
      used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
      h = h / used_sigmas

    return h