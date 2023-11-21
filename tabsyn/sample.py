import os
import argparse
import warnings
import time

import torch
import numpy as np
import pandas as pd

from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target
from tabsyn.diffusion_utils import sample

warnings.filterwarnings('ignore')


def main(args):
    dataname = args.dataname
    device = args.device
    steps = args.steps
    save_path = args.save_path
    is_cond = args.is_cond
    cond_mode = args.cond_mode

    if is_cond:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        cond_embedding_save_path = f'{curr_dir}/ckpt/{dataname}/{cond_mode}_cond_z.npy'
        train_z_cond = torch.tensor(np.load(cond_embedding_save_path)).float()
        train_z_cond = train_z_cond[:, 1:, :]
        B, num_tokens, token_dim = train_z_cond.size()
        in_dim_cond = num_tokens * token_dim
        train_z_cond = train_z_cond.view(B, in_dim_cond).to(device)
        args.no_samples = B
    else:
        train_z_cond = None
        in_dim_cond = None


    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1] 

    mean = train_z.mean(0)

    denoise_fn = MLPDiffusion(in_dim, 1024, is_cond=is_cond, d_in_cond=in_dim_cond).to(device)
    
    model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1], is_cond=is_cond).to(device)

    model.load_state_dict(torch.load(f'{ckpt_path}/model.pt'))

    '''
        Generating samples    
    '''
    start_time = time.time()

    if args.no_samples == None:
        num_samples = train_z.shape[0]
    else:
        num_samples = args.no_samples

    sample_dim = in_dim

    x_next = sample(model.denoise_fn_D, num_samples, sample_dim, device=device, z_cond=train_z_cond)
    x_next = x_next * 2 + mean.to(device)

    syn_data = x_next.float().cpu().numpy()
    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse) 

    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    # convert data type
    for col in syn_df.columns:
        datatype = info['column_info'][str(col)]['subtype']
        if datatype == 'date':
            syn_df[col] = pd.to_datetime(syn_df[col].astype('int64') * 100000000000).dt.date
            continue
        syn_df[col] = syn_df[col].astype(datatype)
    syn_df.rename(columns = idx_name_mapping, inplace=True)
    
    # add fk column if conditional
    if is_cond:
        fks = np.load(f'{curr_dir}/ckpt/{dataname}/{cond_mode}_cond_fks.npy')
        syn_df.insert(info['fk_col_idx'], info['fk_col_name'], fks)
    # add id column
    syn_df.insert(info['id_col_idx'], info['id_col_name'], range(0, len(syn_df))) 
    
    syn_df.to_csv(save_path, index = False)
    
    end_time = time.time()
    print('Time:', end_time - start_time)

    print('Saving sampled data to {}'.format(save_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch.')
    parser.add_argument('--steps', type=int, default=None, help='Number of function evaluations.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'