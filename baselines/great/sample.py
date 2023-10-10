import torch
import pandas as pd

import os

import argparse
import json

from baselines.great.models.great import GReaT
from baselines.great.models.great_utils import _array_to_dataframe


def main(args):

    dataname = args.dataname
 
    dataset_path = f'data/{dataname}/train.csv'
    info_path = f'data/{dataname}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)
    train_df = pd.read_csv(dataset_path)


    curr_dir = os.path.dirname(os.path.abspath(__file__))

    great = GReaT("distilgpt2",                         
              epochs=200,                             
              save_steps=2000,                     
              logging_steps=50,                    
              experiment_dir="ckpt/adult",
              batch_size=24,
              #lr_scheduler_type="constant",        # Specify the learning rate scheduler 
              #learning_rate=5e-5                   # Set the inital learning rate
             )
    
    model_save_path = f'{curr_dir}/ckpt/{dataname}/model.pt'
    great.model.load_state_dict(torch.load(model_save_path))

    great.load_finetuned_model(f"{curr_dir}/ckpt/{dataname}/model.pt")

    df = _array_to_dataframe(train_df, columns=None)
    great._update_column_information(df)
    great._update_conditional_information(df, conditional_col=None)

    
    n_samples = info['train_num']

    samples = great.sample(n_samples, k=100, device=args.device)
    samples.head()
    save_path = args.save_path
    samples.to_csv(save_path, index = False)


    print('Saving sampled data to {}'.format(save_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GReaT')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--bs', type=int, default=16, help='(Maximum) batch size')
    args = parser.parse_args()