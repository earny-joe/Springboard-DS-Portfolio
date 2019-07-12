# functions that alter data into format that is usable for deep learning
import pandas as pd
import numpy as np
import random
import os
import torch

# import fastai libraries
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
from fastai.callbacks import EarlyStoppingCallback, SaveModelCallback

def load_data(path, seed, percent=0.05):
    # read in csv files for training and validation sets
    train_df = pd.read_csv(path/'CheXpert-v1.0-small/train.csv')
    valid_df = pd.read_csv(path/'CheXpert-v1.0-small/valid.csv')
    # add valid column to indicate if observations are part of validation set
    train_df['valid'] = False
    valid_df['valid'] = True
    print('Added valid column')
    print('-' * 30)
    # extract patient id and add to columns
    train_df['Patient_id'] = train_df.Path.str.split('/', 3, True)[2]
    valid_df['Patient_id'] = valid_df.Path.str.split('/', 3, True)[2]
    print('Extracted Patient ID from Path column and created new column named Patient_id')
    print('-' * 30)
    # concat train_df and valid_df together
    full_df = pd.concat([train_df, valid_df])
    print('Concatenated train_df and valid_df, resulting shape = {}'.format(full_df.shape))
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
    
    def sample_df(sample_perc = percent):
        '''function to gather sample of original CheXpert data'''
        train_only_df = full_df[~full_df.valid]
        valid_only_df = full_df[full_df.valid]
        unique_patients = train_only_df.Patient_id.unique()
        mask = np.random.rand(len(unique_patients)) <= sample_perc
        sample_patients = unique_patients[mask]
        sample_df = train_only_df[train_df.Patient_id.isin(sample_patients)]
        sample_df = pd.concat([sample_df,valid_only_df])
        return sample_df
    
    # create sample dataframe
    sample_df = sample_df(sample_perc = percent)
    print('Created sample dataframe with input seed and {}% of original data'.format(percent * 100))
    print('Resulting sample df has shape of {}'.format(sample_df.shape))
    print('-' * 30)
    
    return sample_df

def dl_dataframe_setup(sample_df, pathology):
    # create new dataframe for specific competition pathology (i.e. one of the five competition pathologies from CheXpert paper)
    pathology = pathology
    pathology_df = sample_df[['Path', 'valid', pathology]].fillna(0).reset_index(drop=True)
    print('Created data set with expressed competition pathology, replacing NaN\'s with 0 and resetting the index')
    print('-' * 30)
    
    #output basic information of dataset
    print('')
    print(pathology)
    print('-' * 30)
    print('Shape of sample_df: {}'.format(pathology_df.shape))  
    
    return pathology_df

def uzero(df, pathology):
    '''function that converts uncertain labels (-1) to to Negative (0) labels, then converts pathology column to integer type'''
    # replace -1 (uncertain) labels in train_df with 0 (negative) label
    df = df.replace(-1, 0)
    print('Replaced -1 observations with 0.')
    print('-' * 30)
    # convert pathology column to integer type
    df[pathology] = df[pathology].astype(int)
    print('Converted pathology column into integer type.')
    print('-' * 30)
    
    return df

def uignore(df, pathology):
    '''function that drops uncertain labels (-1) from dataset then converts remaining labels to integer types'''
    # select all observations that aren't -1 then reset the index
    df = df[df[pathology] != -1].reset_index(drop=True)
    print('Dropped -1 observations.')
    print('-' * 30)
    # convert pathology column to integer type
    df[pathology] = df[pathology].astype(int)
    print('Converted pathology column into integer type.')
    print('-' * 30)
    print('Shape of new dataframe: {}'.format(df.shape))
    
    return df


def get_src(df, path, feature_col):
    '''function to convert dataframe to fast.ai ImageList'''
    src = (ImageList.from_df(df=df, path=path, folder='.', suffix='').split_from_df('valid').label_from_df(feature_col))
    
    return src

def get_data(size, src, small_batch = False, default_trans = False):
    '''function to return data ready for model training'''
    # determine batch_size based on GPU memory
    free = gpu_mem_get_free_no_cache()
    # the max size of bs depends on the available GPU RAM
    if small_batch == True:
        bs = 2
    else:
        if free > 8200: 
            bs=32
        else:
            bs=16
    print(f"using bs={bs}, have {free}MB of GPU RAM free.")
    print('-' * 30)
    if default_trans == True:
        data = (src.transform(get_transforms(do_flip=False), size=size, padding_mode='zeros')
                .databunch(bs=bs).normalize(imagenet_stats))
    else:
        data = (src.transform(get_transforms(do_flip=False,
                                             max_rotate=None, 
                                             max_zoom=0., 
                                             max_lighting=0.3, 
                                             max_warp=0,
                                             p_affine=0.5, 
                                             p_lighting=0.75, 
                                             xtra_tfms=[]), 
                              size=size, padding_mode='zeros').databunch(bs=bs).normalize(imagenet_stats))
    print('Data ready.')
    
    return data

