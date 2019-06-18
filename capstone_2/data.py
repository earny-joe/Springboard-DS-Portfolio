# functions that alter data into format that is usable for deep learning
import pandas as pd
pd.options.display.max_columns = None
import numpy as np
import random
import os
import torch

def data_processing(path, task, seed, sample_pct):
    # read in csv files for training and validation sets
    train_df = pd.read_csv(path/'CheXpert-v1.0-small/train.csv')
    valid_df = pd.read_csv(path/'CheXpert-v1.0-small/valid.csv')
    print(train_df.shape)
    print(valid_df.shape)
    
    # extract patient id and add to columns
    train_df['Patient_id'] = train_df.Path.str.split('/', 3, True)[2]
    valid_df['Patient_id'] = valid_df.Path.str.split('/', 3, True)[2]
    print('Extracted Patient ID from Path column and created new column named Patient_id')
    
    # create function to seed data (allows us to more easily reproduce sample data set
    def seed_data(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    print('Created function named seed_data to set seed for sample data')
    
    # seed data
    seed_data(seed)
    print('Seeded data.')
    
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
    print(sample_df.head())
    
    # create new dataframe for specific competition task (i.e. one of the five competition pathologies from CheXpert paper)
    task = task
    train_sample_df = sample_df[['Path', task]].fillna(0).reset_index(drop=True)
    valid_task_df = valid_df[['Path', task]].fillna(0).reset_index(drop=True)
    
    return train_sample_df, valid_task_df
    
    #output shape of both train and validation 
    print(task)
    print('Training shape')
    print(train_sample_df.shape)
    print('')
    print('Validation shape')
    print(valid_task_df.shape)