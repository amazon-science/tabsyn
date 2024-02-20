from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

import numpy as np
import src
import os
import json
import argparse
import warnings

from tabsyn.latent_utils import recover_data
from utils_train import concat_y_to_X

warnings.filterwarnings("ignore")

def make_dataset(
    data_path: str,
    T: src.Transformations,
    task_type,
    change_val: bool,
    concat = True,
):

    # classification
    if task_type == 'binclass' or task_type == 'multiclass':
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy'))  else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

        for split in ['train', 'test']:
            X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                if concat:
                    X_cat_t = concat_y_to_X(X_cat_t, y_t)
                X_cat[split] = X_cat_t  
            if y is not None:
                y[split] = y_t
    else:
        # regression
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

        for split in ['train', 'test']:
            X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)

            if X_num is not None:
                if concat:
                    X_num_t = concat_y_to_X(X_num_t, y_t)
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            if y is not None:
                y[split] = y_t

    info = src.load_json(os.path.join(data_path, 'info.json'))
    

    D = src.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=src.TaskType(info['task_type']),
        n_classes=info.get('n_classes')
    )
    

    if change_val:
        D = src.change_val(D)

    # def categorical_to_idx(feature):
    #     unique_categories = np.unique(feature)
    #     idx_mapping = {category: index for index, category in enumerate(unique_categories)}
    #     idx_feature = np.array([idx_mapping[category] for category in feature])
    #     return idx_feature

    # for split in ['train', 'val', 'test']:
    # D.y[split] = categorical_to_idx(D.y[split].squeeze(1))

    D =  src.transform_dataset(D, T, None)
    
    return D

def main(args):
    dataname = args.dataname
    dataset_path = f'data/{dataname}'

    with open(f'{dataset_path}/info.json', 'r') as f:
        info = json.load(f)

    task_type = info['task_type']
    cat_encoding = args.cat_encoding
    concat = True if task_type == 'regression' else False

    T_dict = {}

    T_dict['normalization'] = "quantile"
    T_dict['num_nan_policy'] = 'mean'
    T_dict['cat_nan_policy'] =  None
    T_dict['cat_min_frequency'] = None
    T_dict['cat_encoding'] = cat_encoding
    T_dict['y_policy'] = "default"

    T = src.Transformations(**T_dict)

    dataset = make_dataset(
        data_path = dataset_path,
        T = T,
        task_type = task_type,
        change_val = False,
        concat = concat
    )

    # conditional sampling for classification dataset

    # if task_type == 'binclass':
    #     X_train = dataset.X_num['train']
    #     y_train = dataset.y['train'].squeeze(1)
    
    #     class_0_idx = (y_train == y_train[0]).nonzero()[0]
    #     class_1_idx = (y_train != y_train[0]).nonzero()[0]

    #     X_train_0 = X_train[class_0_idx]
    #     X_train_1 = X_train[class_1_idx]

    #     X_new = np.random.randn(X_train_0.shape[0] * 2, X_train_0.shape[1])
    #     y = np.zeros((X_train_0.shape[0]))
    #     y_new = np.ones((X_train_0.shape[0] * 2))

    #     X_src = np.concatenate((X_train_0, X_new), axis = 0)
    #     y_src = np.concatenate((y, y_new), axis = 0)

    #     sm = SMOTE()
    #     X_res, y_res = sm.fit_resample(X_src, y_src)
    #     print('Original dataset shape %s' % Counter(y_src))
    #     print('Resampled dataset shape %s' % Counter(y_res))
    #     syn_X0 = X_res[-X_train_0.shape[0]:]
    #     '''

    #     '''

    #     X_new = np.random.randn(X_train_1.shape[0] * 2, X_train_0.shape[1])
    #     y = np.zeros((X_train_1.shape[0]))
    #     y_new = np.ones((X_train_1.shape[0] * 2))

    #     X_src = np.concatenate((X_train_1, X_new), axis = 0)
    #     y_src = np.concatenate((y, y_new), axis = 0)

    #     sm = SMOTE()
    #     X_res, y_res = sm.fit_resample(X_src, y_src)
    #     print('Original dataset shape %s' % Counter(y_src))
    #     print('Resampled dataset shape %s' % Counter(y_res))
    #     syn_X1 = X_res[-X_train_1.shape[0]:]

    #     num_inverse = dataset.num_transform.inverse_transform
    #     cat_inverse = dataset.cat_transform.inverse_transform

    #     num_num_col = len(info['num_col_idx'])

    #     syn_X0_num = syn_X0[:,:num_num_col]
    #     syn_X1_num = syn_X1[:,:num_num_col]
    #     syn_X0_cat = syn_X0[:,num_num_col:]
    #     syn_X1_cat = syn_X1[:,num_num_col:]

    #     recover_X0_num = num_inverse(syn_X0_num)
    #     recover_X0_cat = cat_inverse(syn_X0_cat)
    #     recover_X1_num = num_inverse(syn_X1_num)
    #     recover_X1_cat = cat_inverse(syn_X1_cat)

    #     label_X0 = y_train[class_0_idx]
    #     label_X1 = y_train[class_1_idx]

    #     syn_num = np.concatenate((recover_X0_num, recover_X1_num), axis = 0)
    #     syn_cat = np.concatenate((recover_X0_cat, recover_X1_cat), axis = 0)
    #     syn_y = np.concatenate((label_X0, label_X1), axis = 0)
    #     syn_y = syn_y[:, np.newaxis]

        
    #     syn_df = recover_data(syn_num, syn_cat, syn_y, info)

    #     idx_name_mapping = info['idx_name_mapping']
    #     idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    #     syn_df.rename(columns = idx_name_mapping, inplace=True)
    #     save_path = f'synthetic/{dataname}/smote.csv'
    #     syn_df.to_csv(save_path, index = False)




    # unconditional sampling
    
    if task_type == 'binclass':
        X_train = dataset.X_num['train']
        y_train = dataset.y['train'].squeeze(1)
    
        class_0_idx = (y_train == y_train[0]).nonzero()[0]
        class_1_idx = (y_train != y_train[0]).nonzero()[0]

        X_train_0 = X_train[class_0_idx]
        X_train_1 = X_train[class_1_idx]

        X_new = np.random.randn(X_train_0.shape[0] * 2, X_train_0.shape[1])
        y = np.zeros((X_train_0.shape[0]))
        y_new = np.ones((X_train_0.shape[0] * 2))

        X_src = np.concatenate((X_train_0, X_new), axis = 0)
        y_src = np.concatenate((y, y_new), axis = 0)

        sm = SMOTE()
        X_res, y_res = sm.fit_resample(X_src, y_src)
        print('Original dataset shape %s' % Counter(y_src))
        print('Resampled dataset shape %s' % Counter(y_res))
        syn_X0 = X_res[-X_train_0.shape[0]:]
        '''

        '''

        X_new = np.random.randn(X_train_1.shape[0] * 2, X_train_0.shape[1])
        y = np.zeros((X_train_1.shape[0]))
        y_new = np.ones((X_train_1.shape[0] * 2))

        X_src = np.concatenate((X_train_1, X_new), axis = 0)
        y_src = np.concatenate((y, y_new), axis = 0)

        sm = SMOTE()
        X_res, y_res = sm.fit_resample(X_src, y_src)
        print('Original dataset shape %s' % Counter(y_src))
        print('Resampled dataset shape %s' % Counter(y_res))
        syn_X1 = X_res[-X_train_1.shape[0]:]

        num_inverse = dataset.num_transform.inverse_transform
        cat_inverse = dataset.cat_transform.inverse_transform

        num_num_col = len(info['num_col_idx'])

        syn_X0_num = syn_X0[:,:num_num_col]
        syn_X1_num = syn_X1[:,:num_num_col]
        syn_X0_cat = syn_X0[:,num_num_col:]
        syn_X1_cat = syn_X1[:,num_num_col:]

        recover_X0_num = num_inverse(syn_X0_num)
        recover_X0_cat = cat_inverse(syn_X0_cat)
        recover_X1_num = num_inverse(syn_X1_num)
        recover_X1_cat = cat_inverse(syn_X1_cat)

        label_X0 = y_train[class_0_idx]
        label_X1 = y_train[class_1_idx]

        syn_num = np.concatenate((recover_X0_num, recover_X1_num), axis = 0)
        syn_cat = np.concatenate((recover_X0_cat, recover_X1_cat), axis = 0)
        syn_y = np.concatenate((label_X0, label_X1), axis = 0)
        syn_y = syn_y[:, np.newaxis]

        
        syn_df = recover_data(syn_num, syn_cat, syn_y, info)

        idx_name_mapping = info['idx_name_mapping']
        idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

        syn_df.rename(columns = idx_name_mapping, inplace=True)
        save_path = f'synthetic/{dataname}/smote.csv'
        syn_df.to_csv(save_path, index = False)

    elif task_type == 'regression':
        
        X_train = dataset.X_num['train']
        X_new = np.random.randn(X_train.shape[0] * 2, X_train.shape[1])
        y = np.zeros((X_train.shape[0]))
        y_new = np.ones((X_train.shape[0] * 2))

        X_src = np.concatenate((X_train, X_new), axis = 0)
        y_src = np.concatenate((y, y_new), axis = 0)

        sm = SMOTE()
        X_res, y_res = sm.fit_resample(X_src, y_src)
        print('Original dataset shape %s' % Counter(y_src))
        print('Resampled dataset shape %s' % Counter(y_res))
        syn_X = X_res[-X_train.shape[0]:]

        num_inverse = dataset.num_transform.inverse_transform
        cat_inverse = dataset.cat_transform.inverse_transform

        num_num_col = len(info['num_col_idx'])


        num_col_idx = info['num_col_idx']
        cat_col_idx = info['cat_col_idx']
        target_col_idx = info['target_col_idx']

        num_num_col = len(num_col_idx)
        num_target_col = len(target_col_idx)


        syn_num = syn_X[:,:num_num_col + num_target_col]
        syn_cat = syn_X[:,num_num_col + num_target_col:]

        recover_num = num_inverse(syn_num)
        recover_cat = cat_inverse(syn_cat)

        recover_y = recover_num[:, :num_target_col]
        recover_num = recover_num[:, num_target_col:]

        syn_df = recover_data(recover_num, recover_cat, recover_y, info)

        idx_name_mapping = info['idx_name_mapping']
        idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

        syn_df.rename(columns = idx_name_mapping, inplace=True)
        save_path = f'synthetic/{dataname}/smote.csv'
        syn_df.to_csv(save_path, index = False)

    print('Saving sampled data to {}'.format(save_path))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMOTE')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--cat_encoding', type=str, default='one-hot', help='Encoding method for categorical features.')
    args = parser.parse_args()

    main(args)