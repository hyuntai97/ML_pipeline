import yaml
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

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
import lightgbm as lgb
from modules.metric import *
from modules.utils import *


if __name__ == '__main__':
    PRJ_DIR = './'
    sys.path.append(PRJ_DIR)
    from modules.metric import *
    from modules.tuner import *
    from modules.trainer import train_model

    CONFIG_PATH = f'{PRJ_DIR}config/train_pipeline_config.yml'
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    SEED = config['TRAIN']['seed']
    DATA_DIR = f'{PRJ_DIR}preprocessed/'
    DATA_SN = config['DATA']['data_SN']
    print(f'Data SN: {DATA_SN}')

    SN = generate_serial_number()
    MODELNM = config['MODEL']['model_nm'] + SN
    print(f'Model Name: {MODELNM}')

    TRAIN_DIR = f'{PRJ_DIR}results/train/{MODELNM}/'
    os.makedirs(TRAIN_DIR, exist_ok=True)

    copyfile(CONFIG_PATH, f'{TRAIN_DIR}/train_config_copy.yml')

    Xtrain_file = 'X_train'
    ytrain_file = 'y_train'
    X_train = pd.read_csv(f'{DATA_DIR}{DATA_SN}/{Xtrain_file}.csv')
    y_train = pd.read_csv(f'{DATA_DIR}{DATA_SN}/{ytrain_file}.csv')
    print('Train Data Shape: ', X_train.shape, y_train.shape)

    Xvalid_file = 'X_valid'
    yvalid_file = 'y_valid'  
    X_valid = pd.read_csv(f'{DATA_DIR}{DATA_SN}/{Xvalid_file}.csv')
    y_valid = pd.read_csv(f'{DATA_DIR}{DATA_SN}/{yvalid_file}.csv')
    print('Valid Data Shape: ', X_valid.shape, y_valid.shape)

    use_cols = config['DATA']['use_features']
    X_train = X_train[use_cols]
    X_valid = X_valid[use_cols] 

    cat_lst = config['DATA']['cat_features']
    for c in cat_lst:
        X_train[c] = X_train[c].astype('category')
        X_valid[c] = X_valid[c].astype('category') 

    params = get_parameters(config) 

    eval_result = {}
    # fix_seeds(seed=SEED)

    dataset = {'X_train':X_train
              ,'y_train':y_train
              ,'X_valid':X_valid
              ,'y_valid':y_valid
              ,'CAT_FEATURES':cat_lst}    

    params, model, df_loss = train_model(config, params, eval_result, TRAIN_DIR, **dataset)
    print('Final Parameters: ')
    for k, v in params.items():
        print(f'\t{k}:{v}')    

    save_object(params, str(TRAIN_DIR), 'LGBM_params')
    save_model(model, f'{TRAIN_DIR}', MODELNM)
    save_loss(df_loss, f'{TRAIN_DIR}', config)    

    print(f'**** Feature Importance ****')
    
    print('\t1. Feature Importance: SPLIT')
    ax = lgb.plot_importance(model, max_num_features=20, importance_type='split')
    ax.set(title=f'Feature Importance (split)',
          xlabel='Feature Importance',
          ylabel='Features')
    ax.figure.savefig(f'{TRAIN_DIR}feature_importance_split.png', dpi=100, bbox_inches='tight')
    plt.clf()
    
    print('\t2. Feature Importance: GAIN')
    ax = lgb.plot_importance(model, max_num_features=20, importance_type='gain')
    ax.set(title=f'Feature Importance (gain)',
          xlabel='Feature Importance',
          ylabel='Features')
    ax.figure.savefig(f'{TRAIN_DIR}feature_importance_gain.png', dpi=100, bbox_inches='tight')
    plt.clf()

    print('complete')




    
