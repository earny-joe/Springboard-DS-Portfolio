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
        
def barplot(df, cols):
    '''creates multiple barplots based on df and columns input by user'''
    for col in cols:
        plt.figure(figsize=(10,6))
        sns.barplot(x = list(df[col].value_counts().index), 
                    y = df[col].value_counts(), edgecolor='black')
        plt.xlabel(str(col).replace('_', ' '))
        plt.ylabel('Count')
        plt.title(str(col).upper().replace('_', ' '))
        
def trip_date_review(df, last_trip_date):
    '''Returns observations on last_trip_date column'''
    nl = '\n'
    print('The first trip date was on {}.'.format(df[last_trip_date].min()))
    print('-' * 40)
    print('The most recent trip date was on {}.'.format(df[last_trip_date].max()))
    print('-' * 40)
    print('Here are the 5 most recent trips and their date times:')
    recent_trips = sorted(df[last_trip_date], reverse=True)[:5]
    print(recent_trips)
    print('-' * 40)
    print('Below are the 5 most recent unique trips and their date times:')
    unique_trips = sorted(df[last_trip_date].unique(), reverse=True)[:5]
    print(unique_trips)
    
def users_last30(df, col, date):
    '''outputs percentage of users that have taken a trip since date inputted'''
    users_last30 = len(df[df[col] >= date])
    print('{}% of users have taken a trip since {}.'.format(round((users_last30 / len(df)) * 100, 2), date))
    
def target_variable(df):
    '''function creates new target variable column called retained to show if that user had used the service in last 30 days'''
    df_copy = df.copy()
    df_copy['retained_user'] = -1
    print('Created retained_user column.')
    print('-' * 40)
    df_copy.loc[df_copy['last_trip_date'] >= '2014-06-01', 'retained_user'] = 1
    df_copy.loc[df_copy['last_trip_date'] < '2014-06-01', 'retained_user'] = 0
    print('Assigned a value of 1 if user had used service since 2014-06-01 and 0 if they had not.')
    print('-' * 40)
    # drop datetime columns 
    df_copy.drop(['last_trip_date', 'signup_date'], axis = 1, inplace=True)
    print('Dropped date time columns: last_trip_date and signup_date.')
    print('-' * 40)
    # encode categorical features
    df_copy = pd.get_dummies(df_copy, columns=['city', 'phone', 'ultimate_black_user'])
    print('One-hot encoded categorical columns for Random Forest.')
    return df_copy

def plot_roc_curve(fprs, tprs):
    """Plot the Receiver Operating Characteristic from a list
    of true positive rates and false positive rates."""
    
    # Initialize useful lists + the plot axes.
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(14,10))
    
    # Plot ROC for each K-Fold + compute AUC scores.
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
    # Plot the luck line.
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)
    
    # Plot the mean ROC.
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
    # Plot the standard deviation around the mean ROC.
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    # Fine tune and show the plot.
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    plt.show()
    return (f, ax)

def compute_roc_auc(index):
    y_predict = clf.predict_proba(X.iloc[index])[:,1]
    fpr, tpr, thresholds = roc_curve(y.iloc[index], y_predict)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score