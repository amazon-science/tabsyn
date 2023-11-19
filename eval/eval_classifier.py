import os
import pandas as pd
import numpy as np
import pickle
import argparse

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

from xgboost import XGBClassifier

from sdmetrics.utils import HyperTransformer





class CustomHyperTransformer(HyperTransformer):
    def fit(self, data):
        """Fit the HyperTransformer to the given data.

        Args:
            data (pandas.DataFrame):
                The data to transform.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        for field in data:
            kind = data[field].dropna().infer_objects().dtype.kind
            self.column_kind[field] = kind

            if kind == 'i' or kind == 'f':
                # Numerical column.
                self.column_transforms[field] = {'mean': data[field].mean()}
            elif kind == 'b':
                # Boolean column.
                numeric = pd.to_numeric(data[field], errors='coerce').astype(float)
                self.column_transforms[field] = {'mode': numeric.mode().iloc[0]}
            elif kind == 'O':
                # Categorical column.
                col_data = pd.DataFrame({'field': data[field]})
                enc = OneHotEncoder(handle_unknown='ignore')
                enc.fit(col_data)
                self.column_transforms[field] = {'one_hot_encoder': enc}
            elif kind == 'M':
                # Datetime column.
                nulls = data[field].isna()
                integers = pd.to_numeric(
                    data[field], errors='coerce').to_numpy().astype(np.float64)
                integers[nulls] = np.nan
                self.column_transforms[field] = {'mean': pd.Series(integers).mean()}

    def transform(self, data):
        """Transform the given data based on the data type of each column.

        Args:
            data (pandas.DataFrame):
                The data to transform.

        Returns:
            pandas.DataFrame:
                The transformed data.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        for field in data:
            transform_info = self.column_transforms[field]

            kind = self.column_kind[field]
            if kind == 'i' or kind == 'f':
                # Numerical column.
                data[field] = data[field].fillna(transform_info['mean'])
            elif kind == 'b':
                # Boolean column.
                data[field] = pd.to_numeric(data[field], errors='coerce').astype(float)
                data[field] = data[field].fillna(transform_info['mode'])
            elif kind == 'O':
                # Categorical column.
                col_data = pd.DataFrame({'field': data[field].astype("object")})
                out = transform_info['one_hot_encoder'].transform(col_data).toarray()
                transformed = pd.DataFrame(
                    out, columns=[f'{field}_{i}' for i in range(np.shape(out)[1])])
                data = data.drop(columns=[field])
                data = pd.concat([data, transformed.set_index(data.index)], axis=1)
            elif kind == 'M':
                # Datetime column.
                nulls = data[field].isna()
                integers = pd.to_numeric(
                    data[field], errors='coerce').to_numpy().astype(np.float64)
                integers[nulls] = np.nan
                data[field] = pd.Series(integers)
                data[field] = data[field].fillna(transform_info['mean'])

        return data
    
def discriminative_detection(original_test, synthetic_test, original_train, 
                             synthetic_train, clf=LogisticRegression(solver='lbfgs', max_iter=100), 
                             max_items = 100000, save_path = None, **kwargs):
    # save_path = kwargs.get('save_path', None)
    # metadata = kwargs.get('metadata', None)

    transformed_original_train = original_train.copy()
    transformed_synthetic_train = synthetic_train.copy()
    transformed_original_test = original_test.copy()
    transformed_synthetic_test = synthetic_test.copy()

    # # save test ids and drop the primary key
    # original_ids = transformed_original_test.get(metadata['primary_key'])
    # synthetic_ids =  transformed_synthetic_test.get(metadata['primary_key'])
    # if metadata is not None and 'primary_key' in metadata:
    #     if metadata['primary_key'] in transformed_original_train:
    #         transformed_original_train = transformed_original_train.drop(metadata['primary_key'], axis=1)
    #     if metadata['primary_key'] in transformed_synthetic_train:
    #         transformed_synthetic_train = transformed_synthetic_train.drop(metadata['primary_key'], axis=1)
    #     if metadata['primary_key'] in transformed_synthetic_test:
    #         transformed_synthetic_test = transformed_synthetic_test.drop(metadata['primary_key'], axis=1)
    #     if metadata['primary_key'] in transformed_original_test:
    #         transformed_original_test = transformed_original_test.drop(metadata['primary_key'], axis=1)
    
    # # resample original train and synthetic train to max size
    # if len(original_train) > max_items:
    #     original_train = original_train.sample(max_items)
    # if len(synthetic_train) > max_items:
    #     synthetic_train = synthetic_train.sample(max_items)

    # resample original test and synthetic test to same size
    n = min(len(transformed_original_test), len(transformed_synthetic_test))
    mask_original = np.zeros(len(transformed_original_test), dtype=bool)
    mask_original[:n] = True
    mask_original = np.random.permutation(mask_original)
    mask_synthetic = np.zeros(len(transformed_synthetic_test), dtype=bool)
    mask_synthetic[:n] = True
    mask_synthetic = np.random.permutation(mask_synthetic)

    # apply the mask
    transformed_original_test = transformed_original_test[mask_original]
    transformed_synthetic_test = transformed_synthetic_test[mask_synthetic]
    # # ids
    # original_ids = original_ids[mask_original]
    # synthetic_ids = synthetic_ids[mask_synthetic]

    ht = CustomHyperTransformer()
    transformed_original_train = ht.fit_transform(transformed_original_train)
    columns = transformed_original_train.columns.to_list()
    transformed_original_train = transformed_original_train.to_numpy()
    transformed_original_test = ht.transform(transformed_original_test).to_numpy()
    transformed_synthetic_train = ht.transform(transformed_synthetic_train).to_numpy()
    transformed_synthetic_test = ht.transform(transformed_synthetic_test).to_numpy()

    X_train = np.concatenate([transformed_original_train, transformed_synthetic_train])
    X_test = np.concatenate([transformed_original_test, transformed_synthetic_test])
    # synthetic labels are 1 as this is what we are interested in (for precision and recall)
    y_train = np.hstack([
        np.zeros(transformed_original_train.shape[0]),
        np.ones(transformed_synthetic_train.shape[0])
    ])
    y_test = np.hstack([
        np.zeros(transformed_original_test.shape[0]),
        np.ones(transformed_synthetic_test.shape[0])
    ])

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', clf)
    ])

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    y_pred = probs.argmax(axis=1)
    # if save_path is not None:
    #     # save probabilities
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     df = pd.DataFrame({
    #         'id': np.hstack([original_ids, synthetic_ids]),
    #         'prob_is_fake': probs[:, 1],
    #         'y_pred': y_pred,
    #         'y_true': y_test
    #     })
    #     df.to_csv(save_path, index=False)

    #     # save model binary
    #     model_path = save_path.replace('metrics_report', 'models/evaluation').replace('.csv', '.pkl')
    #     # remove '/probabilities' from path
    #     model_path = model_path.replace('/probabilities', '')
    #     os.makedirs(os.path.dirname(model_path), exist_ok=True)
    #     model.feature_names = columns
    #     with open(model_path, 'wb') as f:
    #         pickle.dump(model, f)
    # return per-sample loss values for 0-1 and log loss
    return {
        'zero_one': (y_test == y_pred).astype(int).tolist(), 
        'log_loss': log_loss(y_test, probs),
        'accuracy': accuracy_score(y_test, y_pred)
    }

def get_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Which model to use for detection.')

    # Add an argument for the model
    parser.add_argument('--model', type=str, choices=['LogisticRegression', 'xgboost'], default=None,
                        help='Specify the classification model (RandomForest or LogisticRegression)')
    # dataset to evaluate
    parser.add_argument('--dataname', type=str, default=None,
                        help='Specify the dataset to evaluate.')

    # Parse the command-line arguments
    args = parser.parse_args()

    return args

def read_data(dataset):
    
    original_train = pd.read_csv(f"synthetic/{dataset}/real.csv")
    synthetic_train = pd.read_csv(f"synthetic/{dataset}/tabsyn.csv")
    original_test = pd.read_csv(f"synthetic/{dataset}/test.csv")
    synthetic_test = pd.read_csv(f"synthetic/{dataset}/tabsyn_test.csv")

    return original_train, synthetic_train, original_test, synthetic_test

if __name__ == "__main__":
    args = get_args()

    original_train, synthetic_train, original_test, synthetic_test = read_data(args.dataname)
    
    if args.model == 'LogisticRegression':
        results = discriminative_detection(original_test, synthetic_test, original_train, 
                             synthetic_train, clf = LogisticRegression(solver='lbfgs', max_iter=100))
    elif args.model == 'xgboost':
        results = discriminative_detection(original_test, synthetic_test, original_train, 
                             synthetic_train, clf = XGBClassifier())
    
    print(results)