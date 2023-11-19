import os
import json
import argparse

import torch
import numpy as np

from tabsyn.vae.model import Model_VAE, Encoder_model, Decoder_model
from utils_train import preprocess, TabularDataset

D_TOKEN = 4
TOKEN_BIAS = True

N_HEAD = 1
FACTOR = 32
NUM_LAYERS = 2

def main(args):
    '''
    Save VAE latents for conditional generation with diffusion
    '''

    dataname = args.dataname
    data_dir = f'data/{dataname}'

    device =  args.device

    info_path = f'data/{dataname}/info.json'
    ckpt_dir = f'tabsyn/vae/ckpt/{dataname}' 

    with open(info_path, 'r') as f:
        info = json.load(f)

    X_num, X_cat, categories, d_numerical = preprocess(data_dir, task_type = info['task_type'], concat=False)

    X_train_num, _ = X_num
    X_train_cat, _ = X_cat

    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat

    X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
    X_train_cat, X_test_cat =  torch.tensor(X_train_cat), torch.tensor(X_test_cat)

    X_test_num = X_test_num.float().to(device)
    X_test_cat = X_test_cat.to(device)
    X_train_num = X_train_num.to(device)
    X_train_cat = X_train_cat.to(device)

    pre_encoder = Encoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)


    pre_encoder.eval()
    
    # Saving latent embeddings
    with torch.no_grad():
        # load pretrained model
        if os.path.exists(f'tabsyn/vae/ckpt/{dataname}/encoder.pt'):
            pre_encoder.load_state_dict(torch.load(f'tabsyn/vae/ckpt/{dataname}/encoder.pt'))
        else:
            model = Model_VAE(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR, bias = True)
            model.load_state_dict(torch.load(f'tabsyn/vae/ckpt/{dataname}/model.pt'))
            pre_encoder.load_weights(model)
            # save the decoder and encoder if encoder.pt does not exist
            pre_decoder = Decoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR)
            pre_decoder.load_weights(model)
            torch.save(pre_encoder.state_dict(), f'tabsyn/vae/ckpt/{dataname}/encoder.pt')
            torch.save(pre_decoder.state_dict(), f'tabsyn/vae/ckpt/{dataname}/decoder.pt')

        

        print('Successfully loaded the model!')

        train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()
        test_z = pre_encoder(X_test_num, X_test_cat).detach().cpu().numpy()


        # save test embeddings
        np.save(f'{ckpt_dir}/test_z.npy', test_z)
        np.save(f'{ckpt_dir}/train_z.npy', train_z)
        print('Successfully save pretrained embeddings in disk!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Variational Autoencoder')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'

    main(args)