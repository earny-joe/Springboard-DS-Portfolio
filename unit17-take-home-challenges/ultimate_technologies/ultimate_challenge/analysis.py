import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def load_data(path):
    import pandas as pd
    # get file path
    file = path + '/data/logins.json'
    # create logins_df using pandas.read_json
    logins_df = pd.read_json(file)
    logins_df.set_index('login_time', inplace=True)
    logins_df['count'] = 1
    
    return logins_df

def data_prep(df):
    '''function that preps data into suitable format'''
    # Aggregate login counts based on 15-minute time intervals
    resample_df = df.resample('15T').sum()
    # add time column
    resample_df['time'] = pd.to_datetime(resample_df.index)
    # fill missing values with 0
    resample_df = resample_df.fillna(0)
    
    return resample_df


def extract_data_info(df):
    '''Extracts time information from logins_df'''
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    df['week'] = df['time'].dt.week
    df['weekday'] = df['time'].dt.weekday
    
    return df

def plot_month_day_hour(logins_df):
    '''plots bar graphs of the number of observations for each month/day/hour'''
    for col in list(logins_df.columns[2:]):
        plt.figure(figsize=(10,6))
        sns.countplot(x=col, data=logins_df)
        plt.title(str(col).upper())
        
def groupby_plot(df, timeframe, weekday=False):
    '''function that will group by a time period, return the count by that period, and plot the results'''
    if weekday == True:
        nweek = 16
        plt.figure(figsize=(12,8))
        ax = (df.groupby('weekday')['count'].sum()/nweek).plot(kind = 'bar', color='steelblue', edgecolor='black')
        plt.title("Average Login per Weekday")
        plt.xlabel("Weekday")
        plt.ylabel("Number of logins")
        plt.xticks(rotation=45)
        ax.set_xticklabels([calendar.day_name[d] for d in range(7)])
    else:
        print('Grouping by ' + str(timeframe).capitalize())
        print(df.groupby(timeframe)['count'].aggregate(np.sum))
        print('-' * 30)
        plt.figure(figsize=(12,8))
        df.groupby(timeframe)['count'].sum().plot(kind = 'bar', color='steelblue', edgecolor='black')
        plt.title('Logins by ' + str(timeframe).capitalize())
        plt.xlabel(str(timeframe).capitalize())
        plt.ylabel('Number of Logins')
        plt.xticks(rotation = 0)
        
def resample_df(df, timeframe):
    '''resamples dataframe according to specified time frame'''
    resample_df = df.resample(timeframe).sum()
    # add data and weekday columns
    return resample_df

def boxplot_graph(df, x, y, timeframe):
    '''constructs boxplot given user input'''
    plt.figure(figsize=(12,8))
    ax = sns.boxplot(x=x, y=y, data=df)
    plt.title("Login Number of " + str(timeframe).capitalize())
    plt.xlabel(str(timeframe).capitalize())
    plt.ylabel("Number of logins")
    plt.xticks(rotation=45)
    ax.set_xticklabels([calendar.day_name[d] for d in range(7)])
           
    
def resample_15_min(logins_df):
    '''function that takes logins_df and returns observations resampled into 15 min segments'''
    # create new df from logins_df
    interval_df = logins_df.set_index('login_time')
    # to represent a login
    interval_df['count'] = 1
    # can drop the other observations for the interval_df
    interval_df.drop(['month', 'day', 'hour', 'minute'], axis=1, inplace=True)
    # resample with 15 min intervals
    interval_df = interval_df.resample('15T').sum()
    interval_df['month'] = interval_df['login_time'].dt.month
    
    return interval_df

def plot_series(interval_df):
    plt.figure(figsize=(16,6))
    plt.plot(interval_df, linewidth=1.5)
    plt.xticks(rotation=45)
    plt.title('Logins')
    
def month_dfs(interval_df):
    '''seperate interval_df into months: jan, feb, march, apr'''
    jan_df = interval_df['1970-01-01':'1970-01-31'].reset_index(drop=False)
    feb_df = interval_df['1970-02-01':'1970-02-28'].reset_index(drop=False)
    march_df = interval_df['1970-03-01':'1970-03-31'].reset_index(drop=False)
    apr_df = interval_df['1970-04-01':'1970-04-13'].reset_index(drop=False)
    
    return jan_df, feb_df, march_df, apr_df

def month_daysofweek(df):
    '''using datetimes isoweekday(), label observations to their respective day of the week'''
    df['day_of_week'] = df['login_time'].map(lambda x: x.isoweekday())
    return df

def bar_plot_graph(df, col):
    plt.figure(figsize=(8,6))
    df[col].value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title(str(col).replace('_', ' ').upper())
    plt.xticks(rotation=0)
    



    
    
    
    