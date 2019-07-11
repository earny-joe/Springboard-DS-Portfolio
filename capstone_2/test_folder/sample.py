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

def prep_data(path):
    '''function that loads in data and does necessary modifications for DL training'''
    # read in csv files for training and validation sets
    train_df = pd.read_csv(path/'CheXpert-v1.0-small/train.csv')
    valid_df = pd.read_csv(path/'CheXpert-v1.0-small/valid.csv')
    print('Read in training and validation data sets')
    print('-' * 30)
    # add valid column to indicate if it is part of validation training set
    train_df['valid'] = False
    valid_df['valid'] = True
    print('Added column to both data sets that indicates if observation is part of validation set or not.')
    print('-' * 30)
    # extract patient id and study number into new columns
    train_df['patient'] = train_df['Path'].str.split('/',3,True)[2]
    train_df['study'] = train_df['Path'].str.split('/',4,True)[3]
    valid_df['patient'] = valid_df['Path'].str.split('/',3,True)[2]
    valid_df['study'] = valid_df['Path'].str.split('/',4,True)[3]
    print('Extracted patient ID and study number for each observation and assigned into new columns.')
    print('-' * 30)
    # create list of pathologies
    pathology_list = list(train_df.columns[5:19])
    # fill in NA's within pathology columns
    train_df[pathology_list] = train_df[pathology_list].fillna(0)
    print('Filled in NaNs within pathology columns with 0 (see Stanford ML Github for further information on dealing with NaNs).')
    print('-' * 30)
    # convert pathologies to integer types
    train_df[pathology_list] = train_df[pathology_list].astype(int)
    valid_df[pathology_list] = valid_df[pathology_list].astype(int)
    print('Converted pathology columns to integer type.')
    print('-' * 30)
    
    # get number of labels that are 0, 1, -1 in Cardiomegaly column
    zeroes, ones, neg_ones = train_df['Cardiomegaly'].value_counts()
    print('Pre-replacement Label Distribution: Label 0 = {}, Label 1 = {}, Label -1 = {}'.format(zeroes, ones, neg_ones))
    print('-' * 30)
    # replace -1 (uncertain) labels in train_df with 0 (negative) label
    train_df['Cardiomegaly'] = train_df['Cardiomegaly'].replace(-1, 0)
    post_zero, post_one = train_df['Cardiomegaly'].value_counts()
    # true or false statement to assert replacement worked
    true_false = ((zeroes + neg_ones) == post_zero)
    print('Replaced uncertain labels in Cardiomegaly column with 0 (i.e. negative)')
    print('Post-replacement Label Distribution: Label 0 = {}, Label 1 = {}'.format(post_zero, post_one))
    print('Does number of post-replacement 0 labels equal the sum of pre-replacement -1s and 0s? {}'.format(true_false))
    
    print('Returned training and validation data sets as pandas dataframes.')
    return train_df, valid_df

def set_seed(seed):
    '''Sets ranomization seed for environment'''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def undersample_and_prep(train_df, valid_df):
    '''function that undersamples pathologies majority class'''
    count_class_0, count_class_1 = train_df['Cardiomegaly'].value_counts()
    # Divide by class
    df_class_0 = train_df[train_df['Cardiomegaly'] == 0]
    df_class_1 = train_df[train_df['Cardiomegaly'] == 1]
    print('Created two new data sets, one with positive observations and the other with the negatives.')
    print('The shape of dataframe containing 0 (negative) labels: {}'.format(df_class_0.shape))
    print('The shape of dataframe containing 1 (positive) labels: {}'.format(df_class_1.shape))
    print('-' * 30)
    # undersample class 0 according to count of class 1
    df_class_0_under = df_class_0.sample(count_class_1)
    df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)
    print('Random under-sampling: \n{}'.format(df_test_under['Cardiomegaly'].value_counts()))
    print('-' * 30)
    # reshuffle all rows in new dataframe and reset index
    df_test_under_shuffled = df_test_under.sample(frac=1).reset_index(drop=True)
    print('Reshuffled rows in new undersampled dataframe.')
    print('-' * 30)
    # concat undersampled train_df and valid_df together
    full_df = pd.concat([df_test_under_shuffled, valid_df])
    print('Concatenated undersampled training data set with validation data.')
    print('Returning full data set.')
    return full_df

def oversample_and_prep(train_df, valid_df, frac = 0.5):
    '''function that oversamples pathologies majority class'''
    count_class_0, count_class_1 = train_df['Cardiomegaly'].value_counts()
    # Divide by class
    df_class_0 = train_df[train_df['Cardiomegaly'] == 0]
    df_class_1 = train_df[train_df['Cardiomegaly'] == 1]
    print('Created two new data sets, one with positive observations and the other with the negatives.')
    print('The shape of dataframe containing 0 (negative) labels: {}'.format(df_class_0.shape))
    print('The shape of dataframe containing 1 (positive) labels: {}'.format(df_class_1.shape))
    print('-' * 30)
    # oversample class 1 according to count of class 0
    df_class_1_over = df_class_1.sample(count_class_0, replace=True)
    df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)
    print('Random over-sampling: \n{}'.format(df_test_over['Cardiomegaly'].value_counts()))
    print('-' * 30)
    # randomly sample according to inputted % from df_test_over to new dataframe and reset index (df_test_over is VERY LARGE)
    df_test_over_sample = df_test_over.sample(frac=frac).reset_index(drop=True)
    sample_class_0, sample_class_1 = df_test_over_sample['Cardiomegaly'].value_counts()
    print('Sampled from over-sampled dataframe.')
    print('Value Counts for Cardiomegaly column in new dataframe: \n{}'.format(df_test_over_sample['Cardiomegaly'].value_counts()))
    print('-' * 30)
    # concat undersampled train_df and valid_df together
    full_df = pd.concat([df_test_over_sample, valid_df])
    print('Concatenated oversampled training data set with validation data.')
    print('Returning full data set.')
    
    return full_df
    

def uignore(df, pathology):
    '''function that drops uncertain labels (-1) according to specific pathology from dataset then converts remaining labels to integer types'''
    # select all observations that aren't -1 then reset the index
    df = df[df[pathology] != -1].reset_index(drop=True)
    print('Dropped -1 observations according to {} pathology.'.format(pathology))
    print('-' * 30)
    # convert pathology column to integer type
    df[pathology_list] = df[pathology_list].astype(int)
    print('Converted pathology column into integer type.')
    print('-' * 30)
    print('Shape of new dataframe: {}'.format(df.shape))
    return df


    