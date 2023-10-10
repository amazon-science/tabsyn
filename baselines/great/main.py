import pandas as pd

import os
import argparse

from baselines.great.models.great import GReaT

def main(args):

    dataname = args.dataname
    batch_size = args.bs
    dataset_path = f'data/{dataname}/train.csv'
    train_df = pd.read_csv(dataset_path)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    great = GReaT("distilgpt2",                         
              epochs=100,                             
              save_steps=2000,                     
              logging_steps=50,                    
              experiment_dir=f"{curr_dir}/ckpt/{dataname}",
              batch_size=batch_size,
              #lr_scheduler_type="constant",        # Specify the learning rate scheduler 
              #learning_rate=5e-5                   # Set the inital learning rate
             )
    
    trainer = great.fit(train_df)
    great.save(ckpt_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GReaT')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--bs', type=int, default=16, help='(Maximum) batch size')
    args = parser.parse_args()