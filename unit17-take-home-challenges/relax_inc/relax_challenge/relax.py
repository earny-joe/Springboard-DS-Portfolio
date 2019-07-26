import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from collections import Counter

def unique_users(engagement_df, users_df):
    '''function that returns the number of unique users in each df and shows the difference between the 2'''
    # Are there the same number of unique users in each data set?
    print('There are {} users who have ever used the application.'\
          .format(len(list(engagement_df['user_id'].unique()))))
    print('There are {} signed up for the application.'\
          .format(len(list(users_df['object_id'].unique()))))
    print('That is a difference of {} users between those who have signed up for the app and those who have used it.'\
          .format(int(len(list(users_df['object_id'].unique()))) 
                  - int(len(list(engagement_df['user_id'].unique())))))
    
def last_sess_creation_time(users_df):
    '''checks for users who do not have a last_session_creation_time'''
    no_last_session = len(users_df[users_df['last_session_creation_time'].isnull()])
    print('There are {} users who have never used the application.'.format(no_last_session))
    
                          