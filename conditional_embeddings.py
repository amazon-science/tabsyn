import json

import numpy as np

INFO_PATH = 'data/Info'

def get_info(name):
    with open(f'{INFO_PATH}/{name}.json', 'r') as f:
        return json.load(f)
    

def load_parent_embeddings(info):
    train_ids = np.load(f'data/{info["parent_name"]}/ids_train.npy').tolist()
    test_ids = np.load(f'data/{info["parent_name"]}/ids_test.npy').tolist()

    train_embeddings = np.load(f'tabsyn/vae/ckpt/{info["parent_name"]}/train_z.npy')
    test_embeddings = np.load(f'tabsyn/vae/ckpt/{info["parent_name"]}/test_z.npy')

    parent_ids = train_ids + test_ids
    parent_embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)
    return parent_ids, parent_embeddings


def save_cond_embeddings(name, info, mode='train'):
    fks = np.load(f'data/{name}/fks_{mode}.npy')
    
    parent_ids, parent_embeddings = load_parent_embeddings(info)
    cond_embeddings = np.zeros((len(fks), parent_embeddings.shape[1], parent_embeddings.shape[2]))
    for i, id in enumerate(fks):
        parent_id = parent_ids.index(id)
        cond_embeddings[i] = parent_embeddings[parent_id]
    
    np.save(f'data/{name}/cond_{mode}_z.npy', cond_embeddings)


def prepare_embeddings(name):
    info = get_info(name)
    save_cond_embeddings(name, info, 'train')
    save_cond_embeddings(name, info, 'test')
     

if __name__ == "__main__":

    # load the conditional embeddings for tables which have parents (currently just copy parent VAE embeddings)
    for name in ['sales']:
        prepare_embeddings(name)