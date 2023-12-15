import numpy as np
import pandas as pd
import os
import sys
import json

TYPE_TRANSFORM ={
    'float', np.float32,
    'str', str,
    'int', int
}

INFO_PATH = 'data/Info'

def preprocess_beijing():
    with open(f'{INFO_PATH}/beijing.json', 'r') as f:
        info = json.load(f)
    
    data_path = info['raw_data_path']

    data_df = pd.read_csv(data_path)
    columns = data_df.columns

    data_df = data_df[columns[1:]]


    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(info['data_path'], index = False)

def preprocess_news():
    with open(f'{INFO_PATH}/news.json', 'r') as f:
        info = json.load(f)

    data_path = info['raw_data_path']
    data_df = pd.read_csv(data_path)
    data_df = data_df.drop('url', axis=1)

    columns = np.array(data_df.columns.tolist())

    cat_columns1 = columns[list(range(12,18))]
    cat_columns2 = columns[list(range(30,38))]

    cat_col1 = data_df[cat_columns1].astype(int).to_numpy().argmax(axis = 1)
    cat_col2 = data_df[cat_columns2].astype(int).to_numpy().argmax(axis = 1)

    data_df = data_df.drop(cat_columns2, axis=1)
    data_df = data_df.drop(cat_columns1, axis=1)

    data_df['data_channel'] = cat_col1
    data_df['weekday'] = cat_col2
    
    data_save_path = 'data/news/news.csv'
    data_df.to_csv(f'{data_save_path}', index = False)

    columns = np.array(data_df.columns.tolist())
    num_columns = columns[list(range(45))]
    cat_columns = ['data_channel', 'weekday']
    target_columns = columns[[45]]

    info['num_col_idx'] = list(range(45))
    info['cat_col_idx'] = [46, 47]
    info['target_col_idx'] = [45]
    info['data_path'] = data_save_path

    with open(f'{INFO_PATH}/{name}.json', 'w') as file:
        json.dump(info, file, indent=4)

def preprocess_store():
    with open(f'{INFO_PATH}/store.json', 'r') as f:
        info = json.load(f)

    id_col_name = info['id_col_name']
    data_path = info['raw_data_path']
    cat_col_idx = info['cat_col_idx']
    data_df = pd.read_csv(data_path)

    # replace nans in categorical columns with '?'
    for col_idx in cat_col_idx:
        data_df.iloc[:, col_idx] = data_df.iloc[:, col_idx].fillna('?')
        
    
    ids_test = data_df.pop(id_col_name).to_numpy()
    np.save(f'data/store/ids_test.npy', ids_test)

    data_df.to_csv(info['test_path'], index = False)

def preprocess_molecule():
    with open(f'{INFO_PATH}/molecule.json', 'r') as f:
        info = json.load(f)
    
    id_col_name = info['id_col_name']
    data_path = info['raw_data_path']
    cat_col_idx = info['cat_col_idx']
    train_data_path = info['data_path']
    data_df = pd.read_csv(data_path)

    # replace nans in categorical columns with '?'
    for col_idx in cat_col_idx:
        data_df.iloc[:, col_idx] = data_df.iloc[:, col_idx].fillna('?')

    # TODO: should not be done here but manually:
    train_data = pd.read_csv(train_data_path)
    train_data.to_csv(train_data_path, index = False)

    ids_test = data_df.pop(id_col_name).to_numpy()
    np.save(f'data/molecule/ids_test.npy', ids_test)

    data_df.to_csv(f'data/molecule/test.csv', index = False)

def preprocess_atom():
    with open(f'{INFO_PATH}/atom.json', 'r') as f:
        info = json.load(f)
    
    id_col_name = info['id_col_name']
    fk_col_name = info['fk_col_name']
    data_path = info['raw_data_path']
    cat_col_idx = info['cat_col_idx']
    train_data_path = info['data_path']
    data_df = pd.read_csv(data_path)

    # replace nans in categorical columns with '?'
    for col_idx in cat_col_idx:
        data_df.iloc[:, col_idx] = data_df.iloc[:, col_idx].fillna('?')

    train_data = pd.read_csv(train_data_path)

    train_data.to_csv(train_data_path, index = False)

    ids_test = data_df.pop(id_col_name).to_numpy()
    fks_test = data_df.pop(fk_col_name[0]).to_numpy()
    np.save(f'data/atom/ids_test.npy', ids_test)
    np.save(f'data/atom/fks_test.npy', fks_test)

    data_df.to_csv(f'data/atom/test.csv', index = False)

def preprocess_bond(reorder=False):
    with open(f'{INFO_PATH}/bond.json', 'r') as f:
        info = json.load(f)
    
    id_col_name = info['id_col_name']
    fk_col_name = info['fk_col_name']
    data_path = info['raw_data_path']
    cat_col_idx = info['cat_col_idx']
    train_data_path = info['data_path']

    data_df_unordered = pd.read_csv(data_path)

    if reorder:
        # TODO: couple with info files
        # reorder dataset to put id as the last column
        data_df = reorder_columns(data_df_unordered, info["id_col_idx"])
    else:
        data_df = data_df_unordered

    # replace nans in categorical columns with '?'
    for col_idx in cat_col_idx:
        data_df.iloc[:, col_idx] = data_df.iloc[:, col_idx].fillna('?')

    train_data = pd.read_csv(train_data_path)

    train_data.to_csv(train_data_path, index = False)

    ids_test = data_df.pop(id_col_name).to_numpy()
    fks_test = np.array([data_df.pop(x).to_numpy() for x in fk_col_name])

    np.save(f'data/{info["name"]}/ids_test.npy', ids_test)
    np.save(f'data/{info["name"]}/fks_test.npy', fks_test)

    data_df.to_csv(info['test_path'], index = False)

def reorder_columns(df, index_id):
    cols = list(df.columns.values)
    index_col = cols.pop(index_id)
    cols = cols.append(index_col)
    
    return df.reindex(columns=cols)

def preprocess_test():
    with open(f'{INFO_PATH}/test.json', 'r') as f:
        info = json.load(f)
    
    id_col_name = info['id_col_name']
    fk_col_name = info['fk_col_name']
    data_path = info['raw_data_path']
    cat_col_idx = info['cat_col_idx']
    train_data_path = info['data_path']
    data_df = pd.read_csv(data_path)

    # replace nans in categorical columns with '?'
    for col_idx in cat_col_idx:
        data_df.iloc[:, col_idx] = data_df.iloc[:, col_idx].fillna('?')

    # convert Date column to int
    # data_df['Date'] = (pd.to_datetime(data_df['Date']).astype('int64') / 100000000000).astype('int64')
    data_df['Date'] = pd.to_numeric(pd.to_datetime(data_df['Date']))
    train_data = pd.read_csv(train_data_path)
    if train_data['Date'].dtype == 'object':
        #train_data['Date'] = (pd.to_datetime(train_data['Date']).astype('int64') /  100000000000).astype('int64')
        train_data['Date'] =  pd.to_numeric(pd.to_datetime(train_data['Date']))
    train_data.to_csv(train_data_path, index = False)

    ids_test = data_df.pop(id_col_name).to_numpy()
    fks_test = data_df.pop(fk_col_name[0]).to_numpy()
    np.save(f'data/test/ids_test.npy', ids_test)
    np.save(f'data/test/fks_test.npy', fks_test)

    data_df.to_csv(f'data/test/test.csv', index = False)


def get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names = None):
    
    if not column_names:
        column_names = np.array(data_df.columns.tolist())
    

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):

        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1


    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k
        
    idx_name_mapping = {}
    
    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def train_val_test_split(data_df, cat_columns, num_train = 0, num_test = 0):
    total_num = data_df.shape[0]
    idx = np.arange(total_num)


    seed = 1234

    while True:
        np.random.seed(seed)
        np.random.shuffle(idx)

        train_idx = idx[:num_train]
        test_idx = idx[-num_test:]


        train_df = data_df.loc[train_idx]
        test_df = data_df.loc[test_idx]



        flag = 0
        for i in cat_columns:
            if len(set(train_df[i])) != len(set(data_df[i])):
                flag = 1
                break

        if flag == 0:
            break
        else:
            seed += 1
        
    return train_df, test_df, seed    

def reindex(cat, num):
    # make lists
    list_cat = np.array([[i, "cat"] for i in cat])
    list_num = np.array([[i, "num"] for i in num])
    if list_cat.size == 0:
        list_ = list_num
    elif list_num.size == 0:
        list_ = list_cat
    else:
        list_ = np.vstack((list_cat, list_num))
    # order by first element
    list_ = sorted(list_, key=lambda x: x[0])
    # collapse
    arr = np.array(list_)
    sorted_indices = np.argsort(arr[:, 0])
    transformed_arr = np.argsort(sorted_indices)
    arr[:, 0] = transformed_arr
    # make new cat and num
    cat_indices = np.where(arr[:, 1] == 'cat')[0]
    num_indices = np.where(arr[:, 1] == 'num')[0]
    # cast to int
    cat_indices = [int(x) for x in cat_indices]
    num_indices = [int(x) for x in num_indices]

    return cat_indices, num_indices

def remove_ids(info, column_names):
    idxs = []
    if info.get("id_col_idx") != None:
        idxs.append(info["id_col_idx"])
        if info.get("fk_col_idx") != None:
            for i in info["fk_col_idx"]:
                idxs.append(i)
    elif info.get("fk_col_idx") != None:
        for i in info["fk_col_idx"]:
            idxs.append(i)
    idxs.sort(reverse=True)
    for i in idxs:
        column_names.pop(i)

    return column_names

def process_data(name):

    if name == 'news':
        preprocess_news()
    elif name == 'beijing':
        preprocess_beijing()
    elif name == 'store':
        preprocess_store()
    elif name == 'test':
        preprocess_test()
    elif name == 'bond':
        preprocess_bond()
    elif name == 'atom':
        preprocess_atom()
    elif name == 'molecule':
        preprocess_molecule()

    with open(f'{INFO_PATH}/{name}.json', 'r') as f:
        info = json.load(f)

    is_cond = info.get('is_cond', False)

    info['cat_col_idx'], info['num_col_idx'] = reindex(info['cat_col_idx'], info['num_col_idx'])

    data_path = info['data_path']
    if info['file_type'] == 'csv':
        data_df = pd.read_csv(data_path, header = info['header'])
        # if info.get("id_col_name"):
        #     data_df = data_df.drop(info['id_col_name'], axis=1)

    elif info['file_type'] == 'xls':
        data_df = pd.read_excel(data_path, sheet_name='Data', header=1)
        # if info.get("id_col_name"):
        #     data_df = data_df.drop(info['id_col_name'], axis=1)

    num_data = data_df.shape[0]

    column_names_ = info['column_names'] if info['column_names'] else data_df.columns.tolist()
    column_names = column_names_
    # remove id and foreign key column names
    column_names = remove_ids(info, column_names)
 
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']
    id_col_name = info.get('id_col_name', None)
    save_dir = f'data/{name}'

    if is_cond:
        fk_col_name = info['fk_col_name']
        fk_train = np.stack([data_df.pop(col).to_numpy() for col in fk_col_name], axis=-1)
             
        np.save(f'{save_dir}/fks_train.npy', fk_train)

    if id_col_name is not None:
        ids_train = data_df.pop(id_col_name).to_numpy()
        np.save(f'{save_dir}/ids_train.npy', ids_train)

    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names)

    # replace nans in categorical columns with '?'
    for col_idx in cat_col_idx:
        data_df.iloc[:, col_idx] = data_df.iloc[:, col_idx].fillna('?')

    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]
    target_columns = [column_names[i] for i in target_col_idx]

    if info['test_path']:

        test_df = pd.read_csv(info['test_path'], header = info['header'])
        
            
        train_df = data_df

    else:  
        # Train/ Test Split, 90% Training, 10% Testing (Validation set will be selected from Training set)

        num_train = int(num_data*0.9)
        num_test = num_data - num_train

        train_df, test_df, seed = train_val_test_split(data_df, cat_columns, num_train, num_test)
    

    train_df.columns = range(len(train_df.columns))
    test_df.columns = range(len(test_df.columns))

    print(name, train_df.shape, test_df.shape, data_df.shape)

    col_info = {}

    for col_idx in num_col_idx:
        col_info[col_idx] = {}
        col_info[col_idx]['type'] = 'numerical'
        col_info[col_idx]['subtype'] = info['column_info'][idx_name_mapping[col_idx]]
        col_info[col_idx]['max'] = float(train_df[col_idx].max())
        col_info[col_idx]['min'] = float(train_df[col_idx].min())
     
    for col_idx in cat_col_idx:
        col_info[col_idx] = {}
        col_info[col_idx]['type'] = 'categorical'
        col_info[col_idx]['subtype'] = info['column_info'][idx_name_mapping[col_idx]]
        col_info[col_idx]['categorizes'] = list(set(train_df[col_idx]))    

    for col_idx in target_col_idx:
        if info['task_type'] == 'regression':
            col_info[col_idx] = {}
            col_info[col_idx]['type'] = 'numerical'
            col_info[col_idx]['subtype'] = info['column_info'][idx_name_mapping[col_idx]]
            col_info[col_idx]['max'] = float(train_df[col_idx].max())
            col_info[col_idx]['min'] = float(train_df[col_idx].min())
        else:
            col_info[col_idx] = {}
            col_info[col_idx]['type'] = 'categorical'
            col_info[col_idx]['subtype'] = info['column_info'][idx_name_mapping[col_idx]]
            col_info[col_idx]['categorizes'] = list(set(train_df[col_idx])) 


    info['column_info'] = col_info

    train_df.rename(columns = idx_name_mapping, inplace=True)
    test_df.rename(columns = idx_name_mapping, inplace=True)

    for col in num_columns:
        train_df.loc[train_df[col] == '?', col] = np.nan
    for col in cat_columns:
        train_df[col] = train_df[col].astype(str)
        train_df.loc[train_df[col] == '?', col] = 'nan'
    for col in num_columns:
        test_df.loc[test_df[col] == '?', col] = np.nan
    for col in cat_columns:
        test_df[col] = test_df[col].astype(str)
        test_df.loc[test_df[col] == '?', col] = 'nan'


    
    X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
    X_cat_train = train_df[cat_columns].to_numpy()
    if target_columns:
        y_train = train_df[target_columns].to_numpy()
    else:
        y_train = np.zeros((train_df.shape[0], 1))


    X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
    X_cat_test = test_df[cat_columns].to_numpy()
    if target_columns:
        y_test = test_df[target_columns].to_numpy()
    else:
        y_test = np.zeros((test_df.shape[0], 1))

 
    np.save(f'{save_dir}/X_num_train.npy', X_num_train)
    np.save(f'{save_dir}/X_cat_train.npy', X_cat_train)
    np.save(f'{save_dir}/y_train.npy', y_train)

    np.save(f'{save_dir}/X_num_test.npy', X_num_test)
    np.save(f'{save_dir}/X_cat_test.npy', X_cat_test)
    np.save(f'{save_dir}/y_test.npy', y_test)

    train_df[num_columns] = train_df[num_columns].astype(np.float32)
    test_df[num_columns] = test_df[num_columns].astype(np.float32)


    train_df.to_csv(f'{save_dir}/train.csv', index = False)
    test_df.to_csv(f'{save_dir}/test.csv', index = False)

    if not os.path.exists(f'synthetic/{name}'):
        os.makedirs(f'synthetic/{name}')
    
    train_df.to_csv(f'synthetic/{name}/real.csv', index = False)
    test_df.to_csv(f'synthetic/{name}/test.csv', index = False)

    print('Numerical', X_num_train.shape)
    print('Categorical', X_cat_train.shape)

    info['column_names'] = column_names_
    info['train_num'] = train_df.shape[0]
    info['test_num'] = test_df.shape[0]

    info['idx_mapping'] = idx_mapping
    info['inverse_idx_mapping'] = inverse_idx_mapping
    info['idx_name_mapping'] = idx_name_mapping 

    metadata = {'columns': {}}
    task_type = info['task_type']
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    for i in num_col_idx:
        metadata['columns'][i] = {}
        metadata['columns'][i]['sdtype'] = 'numerical'
        metadata['columns'][i]['computer_representation'] = 'Float'

    for i in cat_col_idx:
        metadata['columns'][i] = {}
        metadata['columns'][i]['sdtype'] = 'categorical'


    if task_type == 'regression':
        
        for i in target_col_idx:
            metadata['columns'][i] = {}
            metadata['columns'][i]['sdtype'] = 'numerical'
            metadata['columns'][i]['computer_representation'] = 'Float'

    else:
        for i in target_col_idx:
            metadata['columns'][i] = {}
            metadata['columns'][i]['sdtype'] = 'categorical'

    info['metadata'] = metadata

    with open(f'{save_dir}/info.json', 'w') as file:
        json.dump(info, file, indent=4)

    print(f'Processing and Saving {name} Successfully!')

    print(name)
    print('Total', info['train_num'] + info['test_num'])
    print('Train', info['train_num'])
    print('Test', info['test_num'])
    if info['task_type'] == 'regression':
        num = len(info['num_col_idx'] + info['target_col_idx'])
        cat = len(info['cat_col_idx'])
    else:
        cat = len(info['cat_col_idx'] + info['target_col_idx'])
        num = len(info['num_col_idx'])
    print('Num', num)
    print('Cat', cat)



if __name__ == "__main__":

    for name in ['molecule', 'atom', 'bond', 'store', 'test']:
        process_data(name)
