import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path):
    '''function that loads the data'''
    file = open(path + '/data/ultimate_data_challenge.json', 'r')
    df = pd.DataFrame(json.load(file))
    file.close()
    
    return df

def multi_hist_plot(df, cols):
    '''plots multiple histograms of columns passed in by user via cols'''
    for col in cols:
        plt.figure(figsize=(10,6))
        df[col].plot(kind='hist', color='steelblue', edgecolor='black')
        plt.axvline(x = np.mean(df[col]))
        plt.title(str(col).upper().replace('_', ' '))
        print('The mean for the {} column is equal to {}'.format(col, np.mean(df[col])))
        print('-' * 30)
        
def value_counts_plot(df, col, title):
    '''function that plots a bar plot of the value counts from a particular column'''
    plt.figure(figsize=(10,6))
    df[col].value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title(title)
    plt.xticks(rotation=45);
    
def boxplot(df, cols):
    '''constructs multiple boxplots based on user input'''
    for col in cols:
        plt.figure(figsize=(10,6))
        sns.boxplot(x=col, data=df, color='steelblue')
        plt.xlabel(str(col).replace('_', ' '))
        plt.title(str(col).upper().replace('_', ' '))