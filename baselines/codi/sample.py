import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

import os
import json
import warnings

import os
import time

from baselines.codi.diffusion_continuous import GaussianDiffusionTrainer, GaussianDiffusionSampler
import baselines.codi.tabular_dataload as tabular_dataload
from baselines.codi.models.tabular_unet import tabularUnet
from baselines.codi.diffusion_discrete import MultinomialDiffusion
from baselines.codi.utils import *

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

    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    device = args.device
    
    dataname = args.dataname    
    dataset_dir = f'data/{dataname}'
    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)
    task_type = info['task_type']

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}'


    train, train_con_data, train_dis_data, test, (transformer_con, transformer_dis, meta), con_idx, dis_idx = tabular_dataload.get_dataset(args) 
    _, _, categories, d_numerical = preprocess(dataset_dir, task_type = task_type)
    num_class = np.array(categories)

    train_con_data = torch.tensor(train_con_data.astype(np.float32)).float()
    train_dis_data = torch.tensor(train_dis_data.astype(np.int32)).long()

    train_iter_con = DataLoader(train_con_data, batch_size=args.training_batch_size)
    train_iter_dis = DataLoader(train_dis_data, batch_size=args.training_batch_size)
    datalooper_train_con = infiniteloop(train_iter_con)
    datalooper_train_dis = infiniteloop(train_iter_dis)

    num_class = np.array(categories)

    # Condtinuous Diffusion Model Setup
    args.input_size = train_con_data.shape[1] 
    args.cond_size = train_dis_data.shape[1]
    args.output_size = train_con_data.shape[1]
    args.encoder_dim =  list(map(int, args.encoder_dim_con.split(',')))
    args.nf =  args.nf_con
    model_con = tabularUnet(args)
    optim_con = torch.optim.Adam(model_con.parameters(), lr=args.lr_con)
    sched_con = torch.optim.lr_scheduler.LambdaLR(optim_con, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(model_con, args.beta_1, args.beta_T, args.T).to(device)
    net_sampler = GaussianDiffusionSampler(model_con, args.beta_1, args.beta_T, args.T, args.mean_type, args.var_type).to(device)

    args.input_size = train_dis_data.shape[1] 
    args.cond_size = train_con_data.shape[1]
    args.output_size = train_dis_data.shape[1]
    args.encoder_dim =  list(map(int, args.encoder_dim_dis.split(',')))
    args.nf =  args.nf_dis
    model_dis = tabularUnet(args)
    optim_dis = torch.optim.Adam(model_dis.parameters(), lr=args.lr_dis)
    sched_dis = torch.optim.lr_scheduler.LambdaLR(optim_dis, lr_lambda=warmup_lr)
    trainer_dis = MultinomialDiffusion(num_class, train_dis_data.shape, model_dis, args, timesteps=args.T,loss_type='vb_stochastic').to(device)


    num_params_con = sum(p.numel() for p in model_con.parameters())
    num_params_dis = sum(p.numel() for p in model_dis.parameters())
    print('Continuous model params: %d' % (num_params_con))
    print('Discrete model params: %d' % (num_params_dis))

    scores_max_eval = -10

    total_steps_both = args.total_epochs_both * int(train.shape[0]/args.training_batch_size+1)
    sample_step = args.sample_step * int(train.shape[0]/args.training_batch_size+1)
    print("Total steps: %d" %total_steps_both)
    print("Sample steps: %d" %sample_step)
    print("Continuous: %d, %d" %(train_con_data.shape[0], train_con_data.shape[1]))
    print("Discrete: %d, %d"%(train_dis_data.shape[0], train_dis_data.shape[1]))

    epoch = 0
    train_iter_con = DataLoader(train_con_data, batch_size=args.training_batch_size)
    train_iter_dis = DataLoader(train_dis_data, batch_size=args.training_batch_size)
    datalooper_train_con = infiniteloop(train_iter_con)
    datalooper_train_dis = infiniteloop(train_iter_dis)

    model_con.load_state_dict(torch.load(f'{ckpt_dir}/model_con.pt'))
    model_dis.load_state_dict(torch.load(f'{ckpt_dir}/model_dis.pt'))

    model_con.eval()
    model_dis.eval()
    
    print(f"Start sampling")
    start_time = time.time()
    with torch.no_grad():
        x_T_con = torch.randn(train_con_data.shape[0], train_con_data.shape[1]).to(device)
        log_x_T_dis = log_sample_categorical(torch.zeros(train_dis_data.shape, device=device), num_class).to(device)
        x_con, x_dis = sampling_with(x_T_con, log_x_T_dis, net_sampler, trainer_dis, categories, args)

    x_dis = apply_activate(x_dis, transformer_dis.output_info)

    sample_con = transformer_con.inverse_transform(x_con.detach().cpu().numpy())
    sample_dis = transformer_dis.inverse_transform(x_dis.detach().cpu().numpy())
    # sample = np.zeros([train_con_data.shape[0], len(con_idx+dis_idx)])

    sample = pd.DataFrame()

    con_num = 0
    dis_num = 0

    for i in range(len(con_idx) + len(dis_idx)):
        if i in set(con_idx):
            sample[i] = sample_con[:, con_num]
            con_num += 1
        else:
            sample[i] = sample_dis[:, dis_num]
            dis_num += 1

    syn_df = sample
    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns = idx_name_mapping, inplace=True)

    save_path = args.save_path
    syn_df.to_csv(save_path, index = False)

    end_time = time.time()

    print('Samping time:', end_time-start_time)
    print('Saving sampled data to {}'.format(save_path))