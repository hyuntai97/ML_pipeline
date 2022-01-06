import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from optuna import Trial
from modules.metric import *
from modules.utils import *

def objective(trial: Trial, params, evals_result, path, **dataset):
    params['lambda_l1'] = trial.suggest_loguniform('lambda_l1',1e-8, 1e-1)
    params['lambda_l2'] = trial.suggest_loguniform('lambda_l2',1e-8, 1e-1)
    params['num_leaves'] = trial.suggest_int('num_leaves', 30, 200)
    
    train_data = lgb.Dataset(
    dataset['X_train'],
    label=dataset['y_train'],
    categorical_feature=dataset['CAT_FEATURES'],
    free_raw_data=False)
    
    valid_data = lgb.Dataset(
    dataset['X_valid'],
    dataset['y_valid'],
    categorical_feature=dataset['CAT_FEATURES'],
    free_raw_data=False)
    
    model = lgb.train(params, 
                     train_set = train_data,
                     valid_sets = [train_data, valid_data],
                     valid_names = ['train', 'valid'],
                     categorical_feature = dataset['CAT_FEATURES'],
                     evals_result = evals_result, 
                     verbose_eval=100)
    
    make_single_directory(f'{path}trials')
    save_model(model, f'{path}trials/', f'model_{trial.number}')
    save_object(params, f'{path}trials/', f'params_{trial.number}')
    
    df_loss = pd.DataFrame({
        key: evals_result[key][params['metric']]
        for key in evals_result.keys()
    })
    df_loss.to_csv(f'{path}trials/loss_{trial.number}.csv', index=False)
    
    trial.set_user_attr(key='model', value=model)
    trial.set_user_attr(key='loss', value=df_loss)
    
    p_valid = model.predict(valid_data.get_data(),
                           num_iteration=model.best_iteration)
    y_valid = valid_data.get_label()
    
    score = auroc(y_valid, p_valid)
    
    return score


def callback_study(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key='best_model',value=trial.user_attrs['model'])
        study.set_user_attr(key='best_loss',value=trial.user_attrs['loss'])
        