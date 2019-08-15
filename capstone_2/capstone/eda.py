import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE


def eda_path(cwd):
    '''create function that defines project path and data path'''
    # project path
    capstone_path = cwd
    
    # data path
    data_path = capstone_path + '/data/CheXpert-v1.0-small/'
    
    # import Path to utlize PosixPath file system
    from pathlib import Path
    
    path = Path(capstone_path)
    
    return path

def load_data(path):
    '''create function that loads train and valid CSV files'''
    
    # create train_df and valid_df
    import pandas as pd
    train_df = pd.read_csv(path/'data/CheXpert-v1.0-small/train.csv')
    valid_df = pd.read_csv(path/'data/CheXpert-v1.0-small/valid.csv')
    
    #return both dfs for future use
    return train_df, valid_df

def print_info(list_df):
    for idx, df in enumerate(list_df):
        print(idx)
        print(df.info())
        print('-' * 50)
        
def fill_pathologies_nan(train_df):
    '''fill-in null values in train_df'''
    # gather list of pathologies
    pathology_list = list(train_df.columns[5:])
    
    # fill na's with 0, then convert to int
    train_df[pathology_list] = train_df[pathology_list].fillna(0).astype(int)
    
    # print number of null values
    print(train_df.isnull().sum() / len(train_df))
    
    return train_df

def patient_sex_bar_graph(list_df):
    '''returns bar graph that shows how many patients were male v. female'''
    for df in list_df:
        plt.figure(figsize=(8,6))
        df['Sex'].value_counts(normalize=True).plot(kind='bar', color='steelblue', edgecolor='black')
        plt.title('Patient Sex')
        plt.xticks(rotation=45);
        
def patient_age_hist(list_df):
    '''print histogram of patients ages in both train and valid df'''
    for df in list_df:
        plt.figure()
        df['Age'].plot(kind='hist', color='steelblue', edgecolor='black', bins=25, figsize=(10,6))
        plt.axvline(x=np.mean(df['Age']))
        plt.title('Patient Age')

        
def analyze_pathologies(train_df):
    '''function that returns analysis of assorted pathologies for train df'''
    # gather list of pathologies
    pathology_list = list(train_df.columns[5:])
    pathology_list.sort()
    
    # loop through pathologies and return their respective value counts
    for pathology in pathology_list:
        print(pathology)
        print('-' * 30)
        print(train_df[pathology].value_counts(normalize=True))
        print('')
        
def uncertainty_dict(pathology_list, train_df):
    '''function that creates a dictionary of pathologies and the number of their associated uncertainty labels'''
    uncertainty_dict = {}
    for pathology in pathology_list:
        uncertainty_value = len(train_df[train_df[pathology] == -1])
        total_uncertainty = round(uncertainty_value / len(train_df) * 100, 2)
        uncertainty_dict[pathology] = uncertainty_value
        print('{}: {} or {}% of observations in that column.'.format(pathology, uncertainty_value, total_uncertainty))
        
    return uncertainty_dict

def negative_dict(pathology_list, train_df):
    '''function that creates a dictionary of pathologies and the number of their associated uncertainty labels'''
    negative_dict = {}
    for pathology in pathology_list:
        negative_value = len(train_df[train_df[pathology] == 0])
        total_negative = round(negative_value / len(train_df) * 100, 2)
        negative_dict[pathology] = negative_value
        print('{}: {} or {}% of observations in that column.'.format(pathology, negative_value, total_negative))
        
    return negative_dict

def positive_dict(pathology_list, train_df):
    '''function that creates a dictionary of pathologies and the number of their associated uncertainty labels'''
    positive_dict = {}
    for pathology in pathology_list:
        positive_value = len(train_df[train_df[pathology] == 1])
        total_positive = round(positive_value / len(train_df) * 100, 2)
        positive_dict[pathology] = positive_value
        print('{}: {} or {}% of observations in that column.'.format(pathology, positive_value, total_positive))
        
    return positive_dict
    
def plt_dict(dictionary, label):
    '''function that plots inputed dictionary'''
    plt.figure(figsize = (14,6))
    plt.bar(range(len(dictionary)), sorted(dictionary.values(), reverse=True), color='steelblue', edgecolor='black')
    plt.xticks(range(len(dictionary)), sorted(dictionary, key=dictionary.get, reverse=True), rotation=90)
    plt.title('Number of {} Labels for Each Pathology'.format(label))

def uncertain_features(row, pathologies):
    '''function that detects uncertain observations in each row and then outputs string of uncertain labels'''
    feature_list = []
    for feature in pathologies:
        if row[feature] == -1:
            feature_list.append(feature)
            
    return ';'.join(feature_list)

def uncertain_col(df, pathologies):
    '''applies uncertain_features function to dataframe input by user, and creates a new column'''
    df['uncertain_features'] = df.apply(uncertain_features, axis = 1)
    # replace '' with None to represent no uncertain features found
    df['uncertain_features'] = df['uncertain_features'].replace('', 'None')
    
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


def ml_sample(df, random_state, title):
    '''function that resamples with data splits it, fits random forest classifier, cross-validates and then prints out metrics'''
    # Class count
    count_class_0, count_class_1 = df['Cardiomegaly'].value_counts()
    # Divide by class
    df_class_0 = df[df['Cardiomegaly'] == 0]
    df_class_1 = df[df['Cardiomegaly'] == 1]
    # create over samples data set 
    df_class_1_over = df_class_1.sample(count_class_0, replace=True)
    df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)
    print('Random over-sampling:')
    print(df_test_over['Cardiomegaly'].value_counts())
    print('-' * 40)
    # plot new value counts for over-sampled data set
    df_test_over['Cardiomegaly'].value_counts().plot(kind='bar', title='Count (target)', figsize=(12,6), color='steelblue', edgecolor='black')
    # create dataframe of features and the target variable Cardiomegaly
    X_sample = df_test_over.drop('Cardiomegaly', axis=1)
    y_sample = df_test_over['Cardiomegaly']
    # Split the data into 70% train and 30% test
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=1)
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
    feature_imp = pd.Series(classifier.feature_importances_, index=X_sample.columns).sort_values(ascending=False)
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
    plt.title('ROC Curve for {} Model'.format(title))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the plot
    plt.show()
    
    return classifier, classifier_scores
    
def ml_smote_adaboost(X, y, random_state, title):
    '''function that utilizes SMOTE to oversample data set, then splits it, fits AdaBoost classifier, cross-validates
    and then prints out metrics
    '''
    # initiate smote object
    smote = SMOTE(ratio='minority')
    # fit and resample the data set using SMOTE
    X_sm, y_sm = smote.fit_sample(X, y)
    # Split the data into 70% train and 30% test
    X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.3, random_state=random_state)
    # initiate classifier
    classifier = AdaBoostClassifier(n_estimators=100, random_state=random_state)
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
    plt.title('ROC Curve for {} Model'.format(title))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the plot
    plt.show()
    
    return classifier, classifier_scores

def ml_smote_gradient(X, y, random_state, title):
    '''function that utilizes SMOTE to oversample data set, then splits it, fits GradientBoostingClassifier, cross-validates
    and then prints out metrics
    '''
    # initiate smote object
    smote = SMOTE(ratio='minority')
    # fit and resample the data set using SMOTE
    X_sm, y_sm = smote.fit_sample(X, y)
    # Split the data into 70% train and 30% test
    X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.3, random_state=random_state)
    # initiate classifier
    classifier = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
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
    plt.title('ROC Curve for {} Model'.format(title))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the plot
    plt.show()
    
    return classifier, classifier_scores
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
        
