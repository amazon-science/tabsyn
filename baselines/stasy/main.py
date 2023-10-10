import numpy as np

import torch
from torch.utils.data import DataLoader

import baselines.stasy.datasets as datasets
from baselines.stasy.utils import save_checkpoint, restore_checkpoint, apply_activate
import baselines.stasy.losses as losses
from baselines.stasy.models import ncsnpp_tabular
from baselines.stasy.models import utils as mutils
from baselines.stasy.models.ema import ExponentialMovingAverage
import baselines.stasy.sde_lib as sde_lib
from baselines.stasy.configs.config import get_config

import os
import json
import argparse
import warnings
import time

from utils_train import preprocess

warnings.filterwarnings("ignore")


def main(args):
    dataname = args.dataname
    
    config = get_config(dataname)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    dataset_dir = f'data/{dataname}'

    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    task_type = info['task_type']

    dataset = preprocess(dataset_dir, task_type = task_type, cat_encoding = 'one-hot')    
    train_z = torch.tensor(dataset.X_num['train'])

    config.data.image_size = train_z.shape[1]
    print(config.data.image_size)
    # Initialize model.
    config.device = torch.device(f'cuda:{args.gpu}')
    score_model = mutils.create_model(config)
    print(score_model)
    num_params = sum(p.numel() for p in score_model.parameters())
    print("the number of parameters", num_params)
    

    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)

    # optimizer
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0, epoch=0)

    initial_step = int(state['epoch'])

    batch_size = config.training.batch_size 

    shuffle_buffer_size = 10000
    num_epochs = None 


    train_data = train_z
    train_iter = DataLoader(train_data,
                            batch_size=config.training.batch_size,
                            shuffle=True,
                            num_workers=4)


    scaler = datasets.get_data_scaler(config) 
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")
        logging.info(score_model)


    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting

    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting, workdir=ckpt_dir, spl=config.training.spl, 
                                    alpha0=config.model.alpha0, beta0=config.model.beta0)

    best_loss = np.inf
    

    for epoch in range(initial_step, config.training.epoch+1):
        start_time = time.time()
        state['epoch'] += 1

        batch_loss = 0
        batch_num = 0
        for iteration, batch in enumerate(train_iter): 
            batch = batch.to(config.device).float()

            num_sample = batch.shape[0]
            batch_num += num_sample
            loss = train_step_fn(state, batch)

            batch_loss += loss.item() * num_sample
       
        batch_loss = batch_loss / batch_num
        print("epoch: %d, iter: %d, training_loss: %.5e" % (epoch, iteration, batch_loss))

        if batch_loss < best_loss:
            best_loss = batch_loss
            save_checkpoint(os.path.join(ckpt_dir, 'model.pth'), state)

        if epoch % 1000 == 0:
            save_checkpoint(os.path.join(ckpt_dir, f'checkpoint_{epoch}.pth'), state)

        end_time = time.time()
        # print("training time: %.5f" % (end_time - start_time))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='STASY')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number.')

    args = parser.parse_args()