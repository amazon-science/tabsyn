import os
import torch
import time
from torch.utils.data import DataLoader

import argparse
import warnings
import json

from utils_train import preprocess

from baselines.goggle.GoggleModel import GoggleModel

warnings.filterwarnings('ignore')


def main(args):
    
    dataname = args.dataname
    device = args.device

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
    print(gen.model)
    print(gen.model.learned_graph.graph.shape)

    num_params = sum(p.numel() for p in gen.model.encoder.parameters() if p.requires_grad)
    print(f'Number of parameters in encoder: {num_params}')

    start_time = time.time()
    train_loader = DataLoader(X_train, batch_size=gen.batch_size, shuffle=True)
    gen.fit(train_loader, f'{ckpt_dir}/model.pt')
    end_time = time.time()

    print(f'Training time: {end_time - start_time}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GOGGLE')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'