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
import pickle 
import joblib
import string
import random
import glob
from sklearn.metrics import precision_recall_curve

def get_parameters(config):
    params = {}
    for k, v in config['TRAIN']['parameters'].items():
        params[k] = v
    params['random_seed'] = config['TRAIN']['seed']
    
    return params

def save_loss(df_loss, path, config):
    metric_dic = {}
    metric_dic['best_epoch'] = df_loss[df_loss['valid']==df_loss['valid'].max()].index[0]
    metric_dic['learning_rate'] = config['TRAIN']['parameters']['learning_rate']
    metric_df = pd.DataFrame.from_dict(metric_dic, orient='index').T
    metric_df.to_csv(f'{path}/metric_df_train.csv', index=False)
    
    sns.lineplot(data=df_loss, x=df_loss.index, y='train', label='train', color='#5392cd')
    sns.lineplot(data=df_loss, x=df_loss.index, y='valid', label='valid', color='#dd8452')
    plt.title('loss', fontsize=30)
    make_single_directory(f'{path}')
    plt.savefig(f'{path}/loss.png', dpi=100, bbox_inches='tight')
    plt.clf()
    df_loss.to_csv(f'{path}/loss.csv', index=False)

def make_single_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def save_model(model, path, model_nm):
    with open(f'{path}/{model_nm}', 'wb') as fw:
        pickle.dump(model, fw)

def save_object(params, path, param_nm):
    with open(f'{path}/{param_nm}', 'wb') as fw:
        pickle.dump(params, fw)

def load_object(path, model_nm):
    with open(f'{path}{model_nm}', 'rb') as f: 
        model = pickle.load(f)
    return  model

def generate_serial_number():
    string_pool = string.ascii_lowercase + string.digits
    result = ''
    for i in range(7):
        result += random.choice(string_pool)
    return result
    
def save_image(filepath, imagename):
    filename = f'{filepath}{imagename}.png'
    print('Saving data to: ', filename)
    
    files_present = glob.glob(filename)
    if not files_present:
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        print('Export complete')
    else:
        print('WARNINGS: This image file already exists')
        
def precision_recall_curve_plot(y_test, pred_proba):
    precision, recall, threshold = precision_recall_curve(y_test, pred_proba)
    
    plt.figure(figsize=(8,6))
    threshold_boundary = threshold.shape[0]
    plt.plot(threshold, precision[0: threshold_boundary], linestyle='--',label='precision')
    plt.plot(threshold, recall[0: threshold_boundary], label='recall')
    
    stard, end = plt.xlim()
    plt.xticks(np.round(np.arange(stard, end, 0.1), 2))
    
    plt.xlabel('Threshold value')
    plt.ylabel('Precision and Recall value')
    plt.legend()
    plt.grid()