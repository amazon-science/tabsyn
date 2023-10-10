import pandas as pd

import json

def add_space_before_string(s):
    for _ in range(len(s)):
        s = s.strip(' ')

    return ' ' + s

def post_process_adult(syn_path):
    dataname = 'adult'

    syn_path = f'synthetic/{dataname}/great_{i}.csv'

    data_dir = f'data/{dataname}'
    info_path = f'{data_dir}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    cat_col_idx = info['cat_col_idx']

    syn_data = pd.read_csv(syn_path)
    columns = syn_data.columns

    for i, name in enumerate(columns):
        if i in cat_col_idx:
            syn_data[name] = syn_data[name].apply(add_space_before_string)
    
    syn_data.to_csv(syn_path, index=False)


