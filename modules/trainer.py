import yaml
from glob import glob
import pickle 
import joblib
import os, sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from shutil import copyfile

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt 
import seaborn as sns

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from modules.tuner import *

def ordinal_encoding(df_input, encoding_lst, save_dir):
    df = df_input.copy()
    oe = OrdinalEncoder()
    oe.fit(df[encoding_lst])
    np.save(f'{save_dir}encoding.npy', oe)
    print(f'Encoding file saved to: {save_dir}')
    return oe

def train_model(config, params, evals_result, path, **dataset):
    if config['TRAIN']['optuna']['use']:
        N_TRIALS = config['TRAIN']['optuna']['trials']
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, params, evals_result=evals_result,
                                              path=path, **dataset), n_trials=N_TRIALS, callbacks=[callback_study])
        params.update(study.best_trial.params)
        model = study.user_attrs['best_model']
        df_loss = study.user_attrs['best_loss']
        
    else:
        train_data = lgb.Dataset(dataset['X_train'],
                                label=dataset['y_train'],
                                categorical_feature=dataset['CAT_FEATURES'],
                                free_raw_data=False)
        valid_data = lgb.Dataset(dataset['X_valid'],
                                dataset['y_valid'],
                                categorical_feature=dataset['CAT_FEATURES'],
                                free_raw_data=False)
        model = lgb.train(params
                         ,train_set=train_data
                         ,valid_sets=[train_data, valid_data]
                         ,valid_names = ['train', 'valid']
                         ,evals_result=evals_result
                        #  ,num_boost_round=5000
                         ,early_stopping_rounds=1000
                         ,verbose_eval=10
                         )
        df_loss = pd.DataFrame({key: evals_result[key][params['metric']] for key in evals_result.keys()})
        
    return params, model, df_loss

def save_loss(df_loss, path):
    sns.lineplot(data=df_loss, x=df_loss.index, y='train', label='train', color='#5392cd')
    sns.lineplot(data=df_loss, x=df_loss.index, y='valid', label='valid', color='#dd8452')
    plt.title('loss', fontsize=30)
    make_single_directory(f'{path}')
    plt.savefig(f'{path}/loss.png', dpi=300)
    plt.clf()
    df_loss.to_csv(f'{path}/loss.csv', index=False)