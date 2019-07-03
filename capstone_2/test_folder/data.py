# functions that alter data into format that is usable for deep learning
import pandas as pd
import numpy as np
import random
import os
import torch

def data_processing(path, pathology, seed, sample_pct):
    # read in csv files for training and validation sets
    train_df = pd.read_csv(path/'CheXpert-v1.0-small/train.csv')
    valid_df = pd.read_csv(path/'CheXpert-v1.0-small/valid.csv')
    print(train_df.shape)
    print(valid_df.shape)
    print('-' * 30)
    
    # extract patient id and add to columns
    train_df['Patient_id'] = train_df.Path.str.split('/', 3, True)[2]
    valid_df['Patient_id'] = valid_df.Path.str.split('/', 3, True)[2]
    print('Extracted Patient ID from Path column and created new column named Patient_id')
    print('-' * 30)
    
    # create function to seed data (allows us to more easily reproduce sample data set
    def seed_data(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    print('Created function named seed_data to set seed for sample data')
    print('-' * 30)
    
    # seed data
    seed_data(seed)
    print('Seeded data.')
    print('-' * 30)
    
    # function to gather sample of original CheXpert data
    def sample_df(sample_perc = sample_pct):
        # unique observations from patient column
        unique_patients = train_df['Patient_id'].unique() 
        # random mask the same length as # of unqiue patients
        mask = np.random.rand(len(unique_patients)) <= sample_perc 
        # patients sampled according to mask
        sample_patients = unique_patients[mask] 
        # create new sample df
        sample_df = train_df[train_df['Patient_id'].isin(sample_patients)]
        return sample_df
    
    # create sample dataframe
    sample_df = sample_df()
    print('Created sample dataframe with input seed and {}% of original data'.format(sample_pct))
    print('-' * 30)
    
    # create new dataframe for specific competition pathology (i.e. one of the five competition pathologies from CheXpert paper)
    pathology = pathology
    train_sample_df = sample_df[['Path', pathology]].fillna(0).reset_index(drop=True)
    valid_pathology_df = valid_df[['Path', pathology]].fillna(0).reset_index(drop=True)
    print('Created training and validation dataframe with expressed competition pathology, replacing NaN\'s with 0 and resetting the index')
    print('-' * 30)
    
    #output shape of both train and validation 
    print('')
    print(pathology)
    print('-' * 30)
    print('Training shape')
    print(train_sample_df.shape)
    print('')
    print('Validation shape')
    print(valid_pathology_df.shape)
    
    return train_sample_df, valid_pathology_df

def uzero(train_df, valid_df, pathology):
    '''function that converts uncertain labels (-1) to to Negative (0) labels, then converts pathology column to integer type'''
    # replace -1 (uncertain) labels in train_df with 0 (negative) label
    train_df = train_df.replace(-1, 0)
    # convert pathology column to integer type
    train_df[pathology] = train_df[pathology].astype(int)
    valid_df[pathology] = valid_df[pathology].astype(int)
    
    return train_df, valid_df

def uone(train_df, valid_df, pathology):
    '''function that converts uncertain labels (-1) to to Positive (1) labels, then converts pathology column to integer type'''
    # replace -1 (uncertain) labels in train_df with 0 (negative) label
    train_df = train_df.replace(-1, 1)
    # convert pathology column to integer type
    train_df[pathology] = train_df[pathology].astype(int)
    valid_df[pathology] = valid_df[pathology].astype(int)
    
    return train_df, valid_df
    

def uignore(train_df, valid_df, pathology):
    '''function that drops uncertain labels (-1) from dataset then converts remaining labels to integer types'''
    # select all observations that aren't -1 then reset the index
    train_df = train_df[train_df[pathology] != -1].reset_index(drop=True)
    # convert pathology columns to type integer
    train_df[pathology] = train_df[pathology].astype(int)
    valid_df[pathology] = valid_df[pathology].astype(int)
    
    return train_df, valid_df
    