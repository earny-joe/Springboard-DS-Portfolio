import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def load_data(path):
    import pandas as pd
    # get file path
    file = path + '/data/logins.json'
    # create logins_df using pandas.read_json
    logins_df = pd.read_json(file)
    
    return logins_df

def extract_data_info(logins_df):
    '''Extracts time information from logins_df'''
    logins_df['month'] = logins_df['login_time'].dt.month
    logins_df['day'] = logins_df['login_time'].dt.day
    logins_df['hour'] = logins_df['login_time'].dt.hour
    logins_df['minute'] = logins_df['login_time'].dt.minute
    
    return logins_df

def plot_month_day_hour(logins_df):
    '''plots bar graphs of the number of observations for each month/day/hour'''
    for col in list(logins_df.columns[1:4]):
        plt.figure(figsize=(8,6))
        sns.countplot(x=col, data=logins_df)
    
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
    
    return interval_df

def plot_series(interval_df):
    plt.figure(figsize=(14,6))
    plt.plot(interval_df)
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
    df[col].value_counts().plot(kind='bar', figsize=(8,6))
    plt.title(col)
    
    
    
    