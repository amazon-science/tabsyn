import os
import json
import argparse

import torch
import numpy as np
import pandas as pd

import src
from tabsyn.vae.model import Encoder_model


'''
Based on the generated parent data generate the conditional embeddings for the children (VAE embeddings)
'''

D_TOKEN = 4
TOKEN_BIAS = True

N_HEAD = 1
FACTOR = 32
NUM_LAYERS = 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parent='store'
dataname = 'sales'
data_path_train = f'synthetic/{parent}/tabsyn.csv'
data_path_test = f'synthetic/{parent}/tabsyn_test.csv'
info_path = f'data/{parent}/info.json'

with open(info_path, 'r') as f:
    info = json.load(f)

df_train = pd.read_csv(data_path_train)
df_test = pd.read_csv(data_path_test)
column_names = df_train.columns
num_col_idx = info['num_col_idx']
cat_col_idx = info['cat_col_idx']
num_columns = [column_names[i] for i in num_col_idx]
cat_columns = [column_names[i] for i in cat_col_idx]

# replace nan categorical values with a string 'nan
df_train_cat = df_train[cat_columns]
df_test_cat = df_test[cat_columns]

df_train_cat.fillna('nan', inplace=True)
df_test_cat.fillna('nan', inplace=True)

X_num_train = df_train[num_columns].to_numpy().astype(np.float32)
X_cat_train = df_train_cat.to_numpy()
X_num_test = df_test[num_columns].to_numpy().astype(np.float32)
X_cat_test = df_test_cat.to_numpy()


X_cat = {'train': X_cat_train, 'test': X_cat_test}
X_num = {'train': X_num_train, 'test': X_num_test}
y = np.zeros(X_num_train.shape[0]),

T_dict = {}
T_dict['normalization'] = "quantile"
T_dict['num_nan_policy'] = 'mean'
T_dict['cat_nan_policy'] =  None
T_dict['cat_min_frequency'] = None
T_dict['cat_encoding'] = None
T_dict['y_policy'] = "default"

T = src.Transformations(**T_dict)

D = src.Dataset(
    X_num,
    X_cat,
    {'train': y, 'test': y},
    y_info={},
    task_type=src.TaskType(info['task_type']),
    n_classes=info.get('n_classes')
)
dataset = src.transform_dataset(D, T, None)

X_train_num, X_train_cat = dataset.X_num['train'], dataset.X_cat['train']
X_test_num, X_test_cat = dataset.X_num['test'], dataset.X_cat['test']

categories = src.get_categories(X_cat_train)
d_numerical = X_train_num.shape[1]

X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
X_train_cat, X_test_cat =  torch.tensor(X_train_cat), torch.tensor(X_test_cat)

X_train_num = X_train_num.to(device)
X_train_cat = X_train_cat.to(device)
X_test_num = X_test_num.to(device)
X_test_cat = X_test_cat.to(device)

# Load model
pre_encoder = Encoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)
pre_encoder.load_state_dict(torch.load(f'tabsyn/vae/ckpt/{parent}/encoder.pt'))

with torch.no_grad():
    train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()
    test_z = pre_encoder(X_test_num, X_test_cat).detach().cpu().numpy()

# load parent foreign keys
fks_train = df_train[info['id_col_name']].to_numpy()
fks_test = df_test[info['id_col_name']].to_numpy()
# repeat each row 10 times
train_z = np.repeat(train_z, 48, axis=0)
test_z = np.repeat(test_z, 48, axis=0)
fks_train = np.repeat(fks_train, 48, axis=0)
fks_test = np.repeat(fks_test, 48, axis=0)

# save test embeddings
np.save(f'tabsyn/ckpt/{dataname}/train_cond_z.npy', train_z)
np.save(f'tabsyn/ckpt/{dataname}/test_cond_z.npy', test_z)
np.save(f'tabsyn/ckpt/{dataname}/train_cond_fks.npy', fks_train)
np.save(f'tabsyn/ckpt/{dataname}/test_cond_fks.npy', fks_test)