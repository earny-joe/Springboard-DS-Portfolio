import matplotlib.pyplot as plt
import numpy as np

def eda_path(cwd):
    '''create function that defines project path and data path'''
    # project path
    capstone_path = cwd
    
    # data path
    data_path = capstone_path + '/data/CheXpert-v1.0-small/'
    
    # import Path to utlize PosixPath file system
    from pathlib import Path
    
    path = Path(capstone_path)
    
    return path

def load_data(path):
    '''create function that loads train and valid CSV files'''
    
    # create train_df and valid_df
    import pandas as pd
    train_df = pd.read_csv(path/'data/CheXpert-v1.0-small/train.csv')
    valid_df = pd.read_csv(path/'data/CheXpert-v1.0-small/valid.csv')
    
    #return both dfs for future use
    return train_df, valid_df

def print_info(list_df):
    for idx, df in enumerate(list_df):
        print(idx)
        print(df.info())
        print('-' * 50)
        
def fill_pathologies_nan(train_df):
    '''fill-in null values in train_df'''
    # gather list of pathologies
    pathology_list = list(train_df.columns[5:])
    
    # fill na's with 0, then convert to int
    train_df[pathology_list] = train_df[pathology_list].fillna(0).astype(int)
    
    # print number of null values
    print(train_df.isnull().sum() / len(train_df))
    
    return train_df

def patient_sex_bar_graph(list_df):
    '''returns bar graph that shows how many patients were male v. female'''
    for df in list_df:
        plt.figure(figsize=(8,6))
        df['Sex'].value_counts(normalize=True).plot(kind='bar', color='steelblue', edgecolor='black')
        plt.title('Patient Sex')
        plt.xticks(rotation=45);
        
def patient_age_hist(list_df):
    '''print histogram of patients ages in both train and valid df'''
    for df in list_df:
        plt.figure()
        df['Age'].plot(kind='hist', color='steelblue', edgecolor='black', bins=25, figsize=(10,6))
        plt.axvline(x=np.mean(df['Age']))
        plt.title('Patient Age')

        
def analyze_pathologies(train_df):
    '''function that returns analysis of assorted pathologies for train df'''
    # gather list of pathologies
    pathology_list = list(train_df.columns[5:])
    pathology_list.sort()
    
    # loop through pathologies and return their respective value counts
    for pathology in pathology_list:
        print(pathology)
        print('-' * 30)
        print(train_df[pathology].value_counts(normalize=True))
        print('')
        
def uncertainty_dict(pathology_list, train_df):
    '''function that creates a dictionary of pathologies and the number of their associated uncertainty labels'''
    uncertainty_dict = {}
    for pathology in pathology_list:
        uncertainty_value = len(train_df[train_df[pathology] == -1])
        total_uncertainty = round(uncertainty_value / len(train_df) * 100, 2)
        uncertainty_dict[pathology] = uncertainty_value
        print('{}: {} or {}% of observations in that column.'.format(pathology, uncertainty_value, total_uncertainty))
        
    return uncertainty_dict

def negative_dict(pathology_list, train_df):
    '''function that creates a dictionary of pathologies and the number of their associated uncertainty labels'''
    negative_dict = {}
    for pathology in pathology_list:
        negative_value = len(train_df[train_df[pathology] == 0])
        total_negative = round(negative_value / len(train_df) * 100, 2)
        negative_dict[pathology] = negative_value
        print('{}: {} or {}% of observations in that column.'.format(pathology, negative_value, total_negative))
        
    return negative_dict

def positive_dict(pathology_list, train_df):
    '''function that creates a dictionary of pathologies and the number of their associated uncertainty labels'''
    positive_dict = {}
    for pathology in pathology_list:
        positive_value = len(train_df[train_df[pathology] == 1])
        total_positive = round(positive_value / len(train_df) * 100, 2)
        positive_dict[pathology] = positive_value
        print('{}: {} or {}% of observations in that column.'.format(pathology, positive_value, total_positive))
        
    return positive_dict
    
def plt_dict(dictionary, label):
    '''function that plots inputed dictionary'''
    plt.figure(figsize = (14,6))
    plt.bar(range(len(dictionary)), sorted(dictionary.values(), reverse=True), color='steelblue', edgecolor='black')
    plt.xticks(range(len(dictionary)), sorted(dictionary, key=dictionary.get, reverse=True), rotation=90)
    plt.title('Number of {} Labels for Each Pathology'.format(label))
    

    
    
    
    
    
    
    
    
    
        
