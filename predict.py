import yaml 
import os, sys
import glob
from shutil import copyfile
import json 

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, roc_auc_score, log_loss, precision_recall_curve, auc, precision_score, recall_score
from sklearn.preprocessing import Binarizer

import matplotlib as mpl 
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    PRJ_DIR = './'
    sys.path.append(PRJ_DIR)    

    from modules.utils import *
    from modules.metric import *    

    CONFIG_PATH = f'{PRJ_DIR}config/predict_pipeline_config.yml'
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)   

    SEED = config['seed']
    DATA_DIR = f'{PRJ_DIR}preprocessed/'
    TRAIN_SN = config['TRAIN']['train_SN']
    print(f'Train SN: {TRAIN_SN}')

    MODEL_DIR = f'{PRJ_DIR}/results/train/{TRAIN_SN}/'

    SN = generate_serial_number()
    PREDICT_DR = f'{PRJ_DIR}/results/predict/{TRAIN_SN}_{SN}/'
    os.makedirs(PREDICT_DR, exist_ok=True)   

    copyfile(CONFIG_PATH, f'{PREDICT_DR}/pred_config_copy.yml')

    TRAIN_CONFIG_PATH = f'{PRJ_DIR}config/train_pipeline_config.yml'
    with open(TRAIN_CONFIG_PATH, 'r') as f:
        tr_config = yaml.load(f, Loader=yaml.FullLoader)
    
    use_col = tr_config['DATA']['use_features']
    DATA_SN = tr_config['DATA']['data_SN'] 
    MODELNM = tr_config['MODEL']['model_nm']

    Xtest_file = 'X_test'
    Xtrain_file = 'X_train'
    ytrain_file = 'y_train'
    Xvalid_file = 'X_valid'
    yvalid_file = 'y_valid' 

    X_test = pd.read_csv(f'{DATA_DIR}{DATA_SN}/{Xtest_file}.csv')
    X_test = X_test[use_col]
    
    X_train = pd.read_csv(f'{DATA_DIR}{DATA_SN}/{Xtrain_file}.csv')
    X_train = X_train[use_col]
    y_train = pd.read_csv(f'{DATA_DIR}{DATA_SN}/{ytrain_file}.csv')
    
    X_valid = pd.read_csv(f'{DATA_DIR}{DATA_SN}/{Xvalid_file}.csv')
    X_valid = X_valid[use_col]
    y_valid = pd.read_csv(f'{DATA_DIR}{DATA_SN}/{yvalid_file}.csv')

    cat_lst = tr_config['DATA']['cat_features']
    for c in cat_lst:
        X_train[c] = X_train[c].astype('category')
        X_valid[c] = X_valid[c].astype('category')
        X_test[c] = X_test[c].astype('category')    

    model = load_object(MODEL_DIR, TRAIN_SN)

    THRESHOLD = config['PREDICT']['threshold']
    b = Binarizer(threshold=THRESHOLD)  

    y_pred_proba = model.predict(X_valid)
    y_pred = b.fit_transform(y_pred_proba.reshape(-1, 1))
    accuracy = accuracy_score(y_valid, y_pred)
    roc = auroc(np.array(y_valid), y_pred_proba)

    cm = confusion_matrix(y_valid, y_pred)
    cm_matrix = pd.DataFrame(data=cm, columns=['Predict Negatives: 0', 'Predict Positives: 1']
                             ,index=['Actual Negatives:0', 'Actual Positives:1'])
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    # mpl.rcParams['font.family'] = 'KBFG Text'
    save_image(PREDICT_DR, 'cm_'+MODELNM)
    
    print(classification_report(y_valid, y_pred))

    prec = precision_score(y_valid, y_pred)
    rec = recall_score(y_valid, y_pred)
    f1score = f1_score(y_valid, y_pred)
    precision_recall_curve_plot(y_valid, y_pred_proba)
    save_image(PREDICT_DR, 'prc_'+MODELNM)   

    metric_dic = {}
    metric_dic['data_serial_num'] = tr_config['DATA']['data_SN']
    metric_dic['train_serial_num'] = config['TRAIN']['train_SN']    

    metric_dic['test_accuracy'] = accuracy
    metric_dic['test_f1_score'] = f1score
    metric_dic['test_precision'] = prec
    metric_dic['test_auroc'] = roc   

    metric_df = pd.DataFrame.from_dict(metric_dic, orient='index').T
    metric_df_train = pd.read_csv(f'{MODEL_DIR}/metric_df_train.csv')
    metric_df = pd.concat([metric_df, metric_df_train], axis=1)
    metric_df.to_csv(f'{PREDICT_DR}/metric_df.csv', index=False)

    # =========make submission===========
    submission = pd.read_csv('./dataset/gender_submission.csv')
    submission['예측'] = model.predict(X_test)
    submission.to_csv(f'{PREDICT_DR}/submission.csv', index=False)


    print('====complete====')
