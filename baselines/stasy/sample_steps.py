import pandas as pd

import torch

import baselines.stasy.datasets as datasets
from baselines.stasy.utils import restore_checkpoint
import baselines.stasy.losses as losses
import baselines.stasy.sampling as sampling
from baselines.stasy.models import ncsnpp_tabular
from baselines.stasy.models import utils as mutils
from baselines.stasy.models.ema import ExponentialMovingAverage
import baselines.stasy.sde_lib as sde_lib
from baselines.stasy.configs.config import get_config

import os
import argparse
import warnings
import json

from utils_train import preprocess

warnings.filterwarnings("ignore")

def recover_data(syn_num, syn_cat, info):

    target_col_idx = info['target_col_idx']
    if info['task_type'] == 'regression':
        syn_target = syn_num[:, :len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx):]
    
    else:
        print(syn_cat.shape)
        syn_target = syn_cat[:, :len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx):]

    
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']


    idx_mapping = info['idx_mapping']
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()

    if info['task_type'] == 'regression':
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]] 
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]


    else:
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    return syn_df


def main(args): 
    dataname = args.dataname
    steps = args.steps
    save_path = args.save_path

    config = get_config(dataname)
    
    config.device = torch.device(f'cuda:{args.gpu}')
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

    num_inverse = dataset.num_transform.inverse_transform
    cat_inverse = dataset.cat_transform.inverse_transform

    num_features = train_z.shape[1]
    config.data.image_size = num_features

    print('Input dimension: {}'.format(num_features))
    # Initialize model.
    score_model = mutils.create_model(config)
    print(score_model)
    num_params = sum(p.numel() for p in score_model.parameters())
    print("the number of parameters", num_params)

    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)

    # optimizer
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0, epoch=0)


    # load_satate_dict

    state = restore_checkpoint(f'{ckpt_dir}/model.pth', state, config.device)
    print('Loading SAVED model at from {}/model.pth'.format(ckpt_dir))

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

    sampling_shape = (info['train_num'], config.data.image_size)

    
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    config.sampling.method = 'pc'
    sde.N = steps
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    print('Start sampling...')
    samples, n = sampling_fn(score_model, sampling_shape=sampling_shape)

    task_type = info['task_type']
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == 'regression':
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

    syn_data_num = samples[:, :n_num_feat].cpu().numpy()
    cat_sample = samples[:, n_num_feat:].cpu().numpy()


    syn_num = num_inverse(syn_data_num)
    syn_cat = cat_inverse(cat_sample)

    syn_df = recover_data(syn_num, syn_cat, info)


    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns = idx_name_mapping, inplace=True)

        
    syn_df.to_csv(save_path, index = False)

    print('Saving sampled data to {}'.format(save_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STASY')

    parser.add_argument('--gpu', type=int, default=0, help='GPU device number.')
    parser.add_argument('--steps', type=int, default=50, help='Number of steps to evaluate')

    args = parser.parse_args()