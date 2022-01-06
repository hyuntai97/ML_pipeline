# Classic,data manipulation and linear algebra
import pandas as pd
import numpy as np
import yaml
import os, sys

# Data processing, metrics and modeling
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Stats
import scipy.stats as ss
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

#ignore warning messages 
import warnings
warnings.filterwarnings('ignore') 

if __name__ == '__main__':
    PRJ_DIR = './'
    sys.path.append(PRJ_DIR)
    from modules.utils import *

    DATA_DIR = f'{PRJ_DIR}dataset/'
    DATA_SN = generate_serial_number()
    PREPROCESSED_DATA_DIR = f'{PRJ_DIR}preprocessed/'
    os.makedirs(f'{PREPROCESSED_DATA_DIR}{DATA_SN}', exist_ok=True)
    print(f'Data Directory: {DATA_DIR}')

    train = pd.read_csv(f'{DATA_DIR}train.csv')
    test = pd.read_csv(f'{DATA_DIR}test.csv')

    # Adding a column in each dataset before merging
    train['Type'] = 'train'
    test['Type'] = 'test'

    # Merging train and test
    data = train.append(test) # The entire data: train + test.  

    # How many rows and columns in dataset
    print(f'total data shape : {data.shape}')

    # ========================FE======================== 
    print('feature engineering ...')

    # Creating variable Title
    data['Title'] = data['Name']
    # Cleaning name and extracting Title
    for name_string in data['Name']:
        data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=True)
    
    # Replacing rare titles 
    mapping = {'Mlle': 'Miss', 
            'Ms': 'Miss', 
            'Mme': 'Mrs',
            'Major': 'Other', 
            'Col': 'Other', 
            'Dr' : 'Other', 
            'Rev' : 'Other',
            'Capt': 'Other', 
            'Jonkheer': 'Royal',
            'Sir': 'Royal', 
            'Lady': 'Royal', 
            'Don': 'Royal',
            'Countess': 'Royal', 
            'Dona': 'Royal'}
    data.replace({'Title': mapping}, inplace=True)
    titles = ['Miss', 'Mr', 'Mrs', 'Royal', 'Other', 'Master']  

    # Replacing missing age by median/title 
    for title in titles:
        age_to_impute = data.groupby('Title')['Age'].median()[titles.index(title)]
        data.loc[(data['Age'].isnull()) & (data['Title'] == title), 'Age'] = age_to_impute     

    # Creating new feature : family size
    data['Family_Size'] = data['Parch'] + data['SibSp'] + 1
    data.loc[:,'FsizeD']='Alone'
    data.loc[(data['Family_Size']>1),'FsizeD']='Small'
    data.loc[(data['Family_Size']>4),'FsizeD']='Big'

    # fill NA of Fare by Fare's median
    fa = data[data["Pclass"]==3]
    data['Fare'].fillna(fa['Fare'].median(), inplace = True) 

    # Creating new feature : Child 
    data.loc[:,'Child']=1
    data.loc[(data['Age']>=18),'Child']=0

    # fill NA of Embarked by Embarked's median
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

    # drop Cabin column (unuse column)
    data.drop('Cabin', axis=1, inplace=True)

    # columns classify
    target_col = ["Survived"]
    cat_cols = ['Pclass', 'Sex', 'Embarked', 'Title', 'FsizeD', 'Child']
    num_cols = ['Age', 'SibSp', 'Parch', 'Fare', 'Family_Size']   

    #Label encoding Binary columns
    le = LabelEncoder()
    for i in cat_cols :
        data[i] = le.fit_transform(data[i])

    #Scaling Numerical columns
    std = StandardScaler()
    scaled = std.fit_transform(data[num_cols])
    scaled = pd.DataFrame(scaled,columns=num_cols)   

    #dropping original values merging scaled values for numerical columns
    data.reset_index(drop=True, inplace=True)
    df_data_og = data.copy()
    data = data.drop(columns = num_cols,axis = 1)
    data = data.merge(scaled,left_index=True,right_index=True,how = "left")

    # train / valid / test split 
    train = data[data['Type']=='train']
    test = data[data['Type']=='test']

    train, valid = train_test_split(train, test_size=0.3, random_state=42)

    X_train = train.drop('Survived', axis=1)
    y_train = train['Survived']

    X_valid = valid.drop('Survived', axis=1)
    y_valid = valid['Survived']

    X_test = test.drop('Survived', axis=1)

    X_train.to_csv(f'{PREPROCESSED_DATA_DIR}{DATA_SN}/X_train.csv', index=False)
    y_train.to_csv(f'{PREPROCESSED_DATA_DIR}{DATA_SN}/y_train.csv', index=False)
    X_valid.to_csv(f'{PREPROCESSED_DATA_DIR}{DATA_SN}/X_valid.csv', index=False)
    y_valid.to_csv(f'{PREPROCESSED_DATA_DIR}{DATA_SN}/y_valid.csv', index=False)
    X_test.to_csv(f'{PREPROCESSED_DATA_DIR}{DATA_SN}/X_test.csv', index=False)


    print('preprocessing done !')