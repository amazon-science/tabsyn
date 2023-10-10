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
"""Common layers for defining score networks.
"""
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

def get_act(FLAGS):
  if FLAGS.activation.lower() == 'elu':
    return nn.ELU()
  elif FLAGS.activation.lower() == 'relu':
    return nn.ReLU()
  elif FLAGS.activation.lower() == 'lrelu':
    return nn.LeakyReLU(negative_slope=0.2)
  elif FLAGS.activation.lower() == 'swish':
    return nn.SiLU()
  elif FLAGS.activation.lower() == 'tanh':
    return nn.Tanh()
  elif FLAGS.activation.lower() == 'softplus':
    return nn.Softplus()
  else:
    raise NotImplementedError('activation function does not exist!')

def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  def _compute_fans(shape, in_axis=1, out_axis=0):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init

def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  assert len(timesteps.shape) == 1  
  half_dim = embedding_dim // 2
  emb = math.log(max_positions) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1: 
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb

class Encoder(nn.Module):
  def __init__(self, encoder_dim, tdim, FLAGS):
    super(Encoder, self).__init__()
    self.encoding_blocks = nn.ModuleList()
    for i in range(len(encoder_dim)):
      if (i+1)==len(encoder_dim): break
      encoding_block = EncodingBlock(encoder_dim[i], encoder_dim[i+1], tdim, FLAGS)
      self.encoding_blocks.append(encoding_block)

  def forward(self, x, t):
    skip_connections = []
    for encoding_block in self.encoding_blocks:
      x, skip_connection = encoding_block(x, t)
      skip_connections.append(skip_connection)
    return skip_connections, x

class EncodingBlock(nn.Module):
  def __init__(self, dim_in, dim_out, tdim, FLAGS):
    super(EncodingBlock, self).__init__()
    self.layer1 = nn.Sequential( 
        nn.Linear(dim_in, dim_out),
        get_act(FLAGS)
    ) 
    self.temb_proj = nn.Sequential(
        nn.Linear(tdim, dim_out),
        get_act(FLAGS)
    )
    self.layer2 = nn.Sequential(
        nn.Linear(dim_out, dim_out),
        get_act(FLAGS)
    )
    
  def forward(self, x, t):
    x = self.layer1(x).clone()
    x += self.temb_proj(t)
    x = self.layer2(x)
    skip_connection = x
    return x, skip_connection

class Decoder(nn.Module):
  def __init__(self, decoder_dim, tdim, FLAGS):
    super(Decoder, self).__init__()
    self.decoding_blocks = nn.ModuleList()
    for i in range(len(decoder_dim)):
      if (i+1)==len(decoder_dim): break
      decoding_block = DecodingBlock(decoder_dim[i], decoder_dim[i+1], tdim, FLAGS)
      self.decoding_blocks.append(decoding_block)

  def forward(self, skip_connections, x, t):
    zipped = zip(reversed(skip_connections), self.decoding_blocks)
    for skip_connection, decoding_block in zipped:
      x = decoding_block(skip_connection, x, t)
    return x

class DecodingBlock(nn.Module):
  def __init__(self, dim_in, dim_out, tdim, FLAGS):
    super(DecodingBlock, self).__init__()
    self.layer1 = nn.Sequential( 
        nn.Linear(dim_in*2, dim_in),
        get_act(FLAGS)
    )
    self.temb_proj = nn.Sequential(
        nn.Linear(tdim, dim_in),
        get_act(FLAGS)
    )
    self.layer2 = nn.Sequential(
        nn.Linear(dim_in, dim_out),
        get_act(FLAGS)
    )
    
  def forward(self, skip_connection, x, t):
    
    x = torch.cat((skip_connection, x), dim=1)
    x = self.layer1(x).clone()
    x += self.temb_proj(t)
    x = self.layer2(x)

    return x