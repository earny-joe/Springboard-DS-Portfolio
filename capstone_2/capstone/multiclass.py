# fastai libraries
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
from fastai.callbacks import EarlyStoppingCallback

# functions that alter data into format that is usable for deep learning
import pandas as pd
import numpy as np
import random
import os
import torch

def dataset_setup(train_df, valid_df, seed, sample_perc = 0.05):
    '''function that prepares dataset for training'''
    # create lists of u-one and u-zero features (according to performance in paper)
    u_one_features = ['Atelectasis', 'Edema', 'Pleural Effusion']
    u_zero_features = ['Cardiomegaly', 'Consolidation']
    # create valid column indicating if observation is part of training or validation set
    train_df['valid'] = False
    valid_df['valid'] = True
    # create patient and study columns
    train_df['patient'] = train_df['Path'].str.split('/', 3, True)[2]
    train_df['study'] = train_df['Path'].str.split('/', 4, True)[3]
    # do the same for the validation set
    valid_df['patient'] = valid_df['Path'].str.split('/', 3, True)[2]
    valid_df['study'] = valid_df['Path'].str.split('/', 4, True)[3]
    # combine train_df and valid_df
    full_df = pd.concat([train_df, valid_df])
    
    def feature_string(row):
        '''function that determines what pathologies are present for each observation'''
        feature_list = []
        for feature in u_one_features:
            if row[feature] in [-1,1]:
                feature_list.append(feature)
        
        for feature in u_zero_features:
            if row[feature] == 1:
                feature_list.append(feature)
        
        return ';'.join(feature_list)
    
    # apply feature_string function to full_df to extract pathologies for each observation
    full_df['feature_string'] = full_df.apply(feature_string, axis=1).fillna('')
    
    # the following functions seed the data and then gather a sample of CheXpert data set (allows for faster iteration)
    def seed_everything(seed):
        '''function that seeds data according to user input'''
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        
    def sample_chexpert(sample_perc):
        '''function that gathers a sample data set according to seed'''
        train_only_df = full_df[~full_df['valid']]
        valid_only_df = full_df[full_df['valid']]
        unique_patients = train_only_df['patient'].unique()
        mask = np.random.rand(len(unique_patients)) <= sample_perc
        sample_patients = unique_patients[mask]
        
        # gather sample_df
        sample_df = train_only_df[train_df['patient'].isin(sample_patients)]
        full_sample_df = pd.concat([sample_df, valid_only_df])
        
        return full_sample_df
    
def get_src(df = full_df):
    return (ImageList.from_df(df, data_path, 'Path').split_from_df('valid')
            .label_from_df('feature_string',label_delim=';'))
    
def get_data(size, src, bs=16):
    return (src.transform(get_transforms(do_flip=False), size=size, padding_mode='zeros')
                .databunch(bs=bs).normalize(imagenet_stats))