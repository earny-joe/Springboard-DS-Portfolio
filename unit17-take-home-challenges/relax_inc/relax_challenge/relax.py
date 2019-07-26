import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from collections import Counter
from datetime import timedelta, datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE

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
    
def apply_adopt(engagement_df, users_df):
    '''determines if a user has been active within any 7-day period'''
    def adopted_user(x):    
        "takes a users input and returns whether or not they have been active within any 7-day period"
        #select out rows of this user
        df_temp = engagement_df.loc[engagement_df['user_id'] == x] 
        #resample to show if active in a day. .mean() is just of 1
        df_temp = df_temp.resample('D').mean().dropna() 
        adopted = 0
        #loop over active days till the second to last day
        for i in range(len(df_temp)-2): 
            # difference between every 1st and 3rd day
            if df_temp.index[i + 2] - df_temp.index[i] <= timedelta(days=7):
                adopted = 1
                break
            else:
                adopted = 0
        return adopted
    
    # apply to users_df to label adopted users
    users_df['adopted_user'] = users_df['object_id'].apply(adopted_user)
    print('Created adopted_user column and applied adopted_user function.')
    print('-' * 30)
    sum_adopt = sum(users_df['adopted_user'])
    percentage = (sum(users_df['adopted_user'])/len(users_df['adopted_user'])) * 100
    print('There are {} adopted users or {}% of total users.'.format(sum_adopt, percentage))
    
    return engagement_df, users_df

def users_cleanup(df):
    df = df.copy()
    '''function that takes user data set and cleans/preps it for ML'''
    df['creation_time'] = pd.to_datetime(df['creation_time'])
    # convert last_session_creation_time column to datetime
    print('Converted creation_time columns to datetime.')
    print('-' * 30)
    # gather information on email providers
    df['email_provider'] = [x.split('@')[1] for x in df['email']]
    # top 5 email provders
    email_providers = df['email_provider'].value_counts().index[:6]
    # if not in top 5 label as other
    df['email_provider'] = [x if x in email_providers else 'other' for x in df['email_provider']]
    print('Created column that extracts email provider; if not in top five, case receives other label.')
    print('-' * 30)
    # convert necessary columns to categorical types
    categoricals = ['creation_source', 'opted_in_to_mailing_list', 'enabled_for_marketing_drip','adopted_user']
    df[categoricals] = df[categoricals].astype('category')
    print('Converted {} columns to categorical type.'.format(['creation_source', 'opted_in_to_mailing_list', 'enabled_for_marketing_drip','adopted_user']))
    print('-' * 30)
    # fill in columns with missing values
    df[['last_session_creation_time', 'invited_by_user_id']] = df[['last_session_creation_time', 'invited_by_user_id']].fillna(0)
    print('Filled in missing values.')
    print('-' * 30)
    # created dummy columns (i.e. one-hot encoded) the creation source column
    df = pd.get_dummies(df, columns=['creation_source', 'email_provider'])
    print('Created dummy columns (i.e. one-hot encoded) for creation source and email provider columns.')
    
    return df

def ml_baseline(X, y, random_state, title):
    '''function that splits data, fits baseline classifier, cross-validates and then prints out performance metrics'''
    # Split the data into 70% train and 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    # initiate classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    # fit random forest classifier
    classifier.fit(X_train, y_train)
    # cross validate and record performance metrics: roc_auc, precision and recall
    classifier_scores = cross_validate(classifier, X_train, y_train, cv=5,
                                   scoring=['roc_auc', 'precision', 'recall'],
                                   n_jobs=-1)
    print('Mean ROC-AUC score of test set for {}: {:.3f}'.format(title, classifier_scores['test_roc_auc'].mean()))
    print('Mean Precision score of test set for {}: {:.3f}'.format(title, classifier_scores['test_precision'].mean()))
    print('Mean Recall score of test set for {}: {:.3f}'.format(title, classifier_scores['test_recall'].mean()))
    # create series that stores feature importance from classifier model
    feature_imp = pd.Series(classifier.feature_importances_, index=X.columns).sort_values(ascending=False)
    # plot important features
    plt.figure(figsize=(12,6))
    sns.barplot(x=feature_imp[:6], y=feature_imp.index[:6], edgecolor='black')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    # plot AUROC curve
    # predict probabilities
    probs = classifier.predict_proba(X_test)
    # keep probabilities for positive outcome only
    probs = probs[:, 1]
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    #calculate AUC
    auc = roc_auc_score(y_test, probs)
    plt.figure(figsize=(12,8))
    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    plt.title('ROC Curve for {} Model'.format(title))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the plot
    plt.show()
    
    return classifier, classifier_scores

def ml_smote(X, y, random_state, title):
    '''function that utilizes SMOTE to oversample data set, then splits it, fits random forest classifier, cross-validates
    and then prints out metrics
    '''
    # initiate smote object
    smote = SMOTE(ratio='minority')
    # fit and resample the data set using SMOTE
    X_sm, y_sm = smote.fit_sample(X, y)
    # Split the data into 70% train and 30% test
    X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.3, random_state=random_state)
    # initiate classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    # fit random forest classifier to model
    classifier.fit(X_train, y_train)
    # cross validate and record performance metrics
    classifier_scores = cross_validate(classifier, X_train, y_train, cv=5,
                                   scoring=['roc_auc', 'precision', 'recall'],
                                   n_jobs=-1)
    print('Mean ROC-AUC score of test set for {}: {:.3f}'.format(title, classifier_scores['test_roc_auc'].mean()))
    print('Mean Precision score of test set for {}: {:.3f}'.format(title, classifier_scores['test_precision'].mean()))
    print('Mean Recall score of test set for {}: {:.3f}'.format(title, classifier_scores['test_recall'].mean()))
   # create series that stores feature importance from classifier model
    feature_imp = pd.Series(classifier.feature_importances_, index=X.columns).sort_values(ascending=False)
    # plot important features
    plt.figure(figsize=(12,6))
    sns.barplot(x=feature_imp[:6], y=feature_imp.index[:6], edgecolor='black')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    # plot AUROC curve
    # predict probabilities
    probs = classifier.predict_proba(X_test)
    # keep probabilities for positive outcome only
    probs = probs[:, 1]
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    #calculate AUC
    auc = roc_auc_score(y_test, probs)
    plt.figure(figsize=(12,8))
    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    plt.title('AUROC Curve for {} Model'.format(title))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the plot
    plt.show()
    
    return classifier, classifier_scores
    


    

    
    
                          