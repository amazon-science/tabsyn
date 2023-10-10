import os
import numpy as np

import torch
from torch.utils.data import DataLoader

import json
import argparse
import warnings
import time


from baselines.codi.diffusion_continuous import GaussianDiffusionTrainer, GaussianDiffusionSampler
import baselines.codi.tabular_dataload as tabular_dataload
from baselines.codi.models.tabular_unet import tabularUnet
from baselines.codi.diffusion_discrete import MultinomialDiffusion
from baselines.codi.utils import *

warnings.filterwarnings("ignore")
from utils_train import preprocess


def main(args):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    dataname = args.dataname

    dataset_dir = f'data/{dataname}'
    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)
    task_type = info['task_type']

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

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

    print('Continuous model:')
    print(model_con)

    print('Discrete model:')
    print(model_dis)
    
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

    best_loss = float('inf')
    for step in range(total_steps_both):

        start_time = time.time()
        model_con.train()
        model_dis.train()

        x_0_con = next(datalooper_train_con).to(device).float()
        x_0_dis = next(datalooper_train_dis).to(device)

        ns_con, ns_dis = make_negative_condition(x_0_con, x_0_dis)
        con_loss, con_loss_ns, dis_loss, dis_loss_ns = training_with(x_0_con, x_0_dis, trainer, trainer_dis, ns_con, ns_dis, categories, args)

        loss_con = con_loss + args.lambda_con * con_loss_ns
        loss_dis = dis_loss + args.lambda_dis * dis_loss_ns

        optim_con.zero_grad()
        loss_con.backward()
        torch.nn.utils.clip_grad_norm_(model_con.parameters(), args.grad_clip)
        optim_con.step()
        sched_con.step()

        optim_dis.zero_grad()
        loss_dis.backward()
        torch.nn.utils.clip_grad_value_(trainer_dis.parameters(), args.grad_clip)#, self.args.clip_value)
        torch.nn.utils.clip_grad_norm_(trainer_dis.parameters(), args.grad_clip)#, self.args.clip_norm)
        optim_dis.step()
        sched_dis.step()
        
        total_loss = loss_con.item() + loss_dis.item()
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model_con.state_dict(), f'{ckpt_dir}/model_con.pt')
            torch.save(model_dis.state_dict(), f'{ckpt_dir}/model_dis.pt')
            
        if (step+1) % int(train.shape[0]/args.training_batch_size+1) == 0:

            print(f"Epoch:{epoch}, step = {step}, diffusion continuous loss: {con_loss:.3f}, discrete loss: {dis_loss:.3f}")
            print(f"Epoch:{epoch}, step = {step}, CL continuous loss: {con_loss_ns:.3f}, discrete loss: {dis_loss_ns:.3f}")
            print(f"Epoch:{epoch}, step = {step}, Total continuous loss: {loss_con:.3f}, discrete loss: {loss_dis:.3f}")
            epoch +=1
        
            if epoch % 1000 == 0:
                torch.save(model_con.state_dict(), f'{ckpt_dir}/model_con_{epoch}.pt')
                torch.save(model_dis.state_dict(), f'{ckpt_dir}/model_dis_{epoch}.pt')
        
        end_time = time.time()
        print(f"Time taken: {end_time-start_time:.3f}")