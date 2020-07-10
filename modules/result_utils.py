import pickle

import pandas as pd


method_columns = ['model_class', 'config', 'loss_function', 'q_dist', 'sample_from_q',
                  'detach', 'add_noise', 'noise_type', 'warm_up', 'is_loaded', 'method_name']
hparam_columns = ['grad_l1_penalty', 'grad_weight_decay',
                  'lamb', 'loss_function_param', 'noise_std', 'lr', 'weight_decay']
data_columns = ['dataset', 'label_noise_level', 'label_noise_type', 'num_train_examples',
                'remove_prob', 'transform_function', 'data_augmentation']
ignore_columns = ['device', 'batch_size', 'epochs', 'stopping_param', 'save_iter', 'vis_iter',
                  'clean_validation', 'pretrained_arg', 'load_from']
not_listed_columns = ['seed', 'log_dir']


method_order = {
    'CE': 0,
    'CE-noisy-grad-Gaussian': 0.1,
    'CE-noisy-grad-Laplace': 0.2,
    'MAE': 1,
    'FW': 2,
    'DMI': 3,
    'Penalize': 3.5,
    'Predict-Gaussian': 4,
    'Predict-Gaussian-sample': 4.1,
    'Predict-Laplace': 5,
    'Predict-Laplace-sample': 5.1,
    'Predict-Gaussian-loaded': 6,
    'Predict-Laplace-loaded': 7
}


def load_result_tables(list_of_datasets):
    """ Loads results datasets from stored .pkl files. """
    datasets = []
    df = None
    for dataset_path in list_of_datasets:
        with open(dataset_path, 'rb') as f:
            df = pickle.load(f)
        datasets.append(df)
    df = df.drop(labels=ignore_columns, axis=1)  # drop columns that do not matter
    df = pd.concat(datasets, sort=False).reset_index(drop=True)
    df['num_train_examples'].fillna('N/A', inplace=True)
    df['transform_function'].fillna('N/A', inplace=True)
    df['detach'].fillna(1.0, inplace=True)
    df['load_from'].fillna('N/A', inplace=True)
    df['is_loaded'] = (df.load_from != 'N/A')
    df['pretrained_arg'].fillna('N/A', inplace=True)
    df['lr'].fillna('1e-3', inplace=True)

    if 'warm_up' in df.columns:
        df['warm_up'].fillna(0, inplace=True)
    else:
        df['warm_up'] = 0

    if 'weight_decay' is df.columns:
        df['weight_decay'].fillna(0.0, inplace=True)
    else:
        df['weight_decay'] = 0.0

    df['method_name'] = 'unknown'
    return df


def infer_method_name(row):
    if row.model_class == 'StandardClassifier':
        if row.loss_function == 'dmi':
            return 'DMI'
        if row.loss_function == 'fw':
            return 'FW'
        if row.loss_function == 'mae':
            return 'MAE'
        assert row.loss_function == 'ce'
        if row.add_noise == 1.0:
            return 'CE-noisy-grad-{}'.format(row.noise_type)
        return 'CE'
    if row.model_class == 'PredictGradOutput':
        ret = 'Predict'
        ret += f"-{row.q_dist}"
        if row.sample_from_q:
            ret += '-sample'
        if row.loss_function != 'ce':
            ret += f"-{row.loss_function}"
        if row.detach == 0.0:
            ret += '-nodetach'
        if row.is_loaded:
            ret += '-loaded'
        if row.warm_up != 0:
            ret += f"-warm_up{row['warm_up']}"
        return ret
    if row.model_class == 'PenalizeLastLayerFixedForm':
        return 'Penalize'
    return 'unknown'


def fill_short_names(df):
    for idx, row in df.iterrows():
        df.at[idx, 'method_name'] = infer_method_name(row)
    return df


def get_agg_results(df):
    """ Takes a dataframe containing all results and computes aggregate results. """
    grouped = df.groupby(method_columns + hparam_columns + data_columns)
    total_size = 0
    for key, item in grouped:
        group = grouped.get_group(key)
        assert len(group) <= 5  # less than 5 seeds always
        assert len(set(group['seed'])) == len(group)  # all seeds are distinct

        if item.dataset.iloc[0] == 'mnist' and item.label_noise_type.iloc[0] == 'error':
            if item.sample_from_q.iloc[0] == True:
                assert len(group) == 3
            elif item.model_class.iloc[0] == 'PenalizeLastLayerFixedForm':
                assert len(group) == 3
            else:
                assert len(group) == 5

        total_size += len(group)
    assert total_size == len(df)

    agg_results = grouped.agg({'test_accuracy': ['mean', 'std'], 'val_accuracy': ['mean', 'std']})
    agg_results = agg_results.reset_index()
    agg_results.columns = ['_'.join(tup).rstrip('_') for tup in agg_results.columns.values]

    return agg_results


def do_model_selection_by_val_score(df):
    """ Takes aggregate results and selects best model by val_accuracy_mean. """

    def select(group):
        idx = group['val_accuracy_mean'].idxmax()
        return group.loc[idx]

    grouped = df.groupby(method_columns + data_columns)
    best_results = grouped.apply(select)
    best_results = best_results.reset_index(drop=True)

    return best_results
