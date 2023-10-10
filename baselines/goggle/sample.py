import os
import pandas as pd
import torch

import argparse
import warnings
import json
import time
from utils_train import preprocess

from baselines.goggle.GoggleModel import GoggleModel
import json

warnings.filterwarnings('ignore')


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
    device = args.device
    save_path = args.save_path

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = f'data/{dataname}'
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    task_type = info['task_type']

    dataset = preprocess(dataset_dir, task_type = task_type, cat_encoding = 'one-hot')
    X_train = torch.tensor(dataset.X_num['train'])

    num_inverse = dataset.num_transform.inverse_transform
    cat_inverse = dataset.cat_transform.inverse_transform


    gen = GoggleModel(
        ds_name=dataname,
        input_dim=X_train.shape[1],
        encoder_dim=2048,
        encoder_l=4,
        het_encoding=True,
        decoder_dim=2048,
        decoder_l=4,
        threshold=0.1,
        decoder_arch="gcn",
        graph_prior=None,
        prior_mask=None,
        device=device,
        beta=1,
        learning_rate=0.01,
        seed=42,
    )

    gen.model.load_state_dict(torch.load(f'{ckpt_dir}/model.pt'))

    start_time = time.time()

    samples = gen.sample(X_train)

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

    syn_data_num = samples[:, :n_num_feat]
    cat_sample = samples[:, n_num_feat:]
    
    syn_num = num_inverse(syn_data_num)
    syn_cat = cat_inverse(cat_sample)

    syn_df = recover_data(syn_num, syn_cat, info)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns = idx_name_mapping, inplace=True)
    syn_df.to_csv(save_path, index = False)

    end_time = time.time()  
    print(f'Sampling time = {end_time - start_time}')
    print('Saving sampled data to {}'.format(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training ')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'