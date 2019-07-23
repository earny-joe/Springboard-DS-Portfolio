import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from time import time

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

def ml_baseline(X, y, random_state, title):
    '''function that splits data, fits baseline classifier, cross-validates and then prints out performance metrics'''
    # Split the data into 70% train and 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    # initiate classifier
    classifier = RandomForestClassifier(n_estimators = 100, random_state=random_state, n_jobs=-1)
    # fit random forest classifier
    classifier.fit(X_train, y_train)
    # cross validate and record performance metrics: roc_auc, precision and recall
    classifier_scores = cross_validate(classifier, X_train, y_train, cv=5,
                                   scoring=['roc_auc', 'precision', 'recall', 'accuracy'],
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

def report(results, n_top=3):
    '''#Utility function to report best scores'''
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_AUC'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_AUC'][candidate],
                  results['std_test_AUC'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
def gridsearch_params():
    '''function that returns dictionary of assorted hyperparameter values to test in GridSearchCV'''
    # number of trees in the forest
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]
    # number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # max number of levels in trees
    max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
    # minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # min number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # method of selecting samples for training each tree
    bootstrap = [True, False]
    # create random grid
    random_grid = {'n_estimators': n_estimators, #number of trees in the forest
               'max_features': max_features, # number of features to consider when looking for the best split
               'max_depth': max_depth, # max depth of the tree
               'min_samples_split': min_samples_split, # min number of samples required to split an internal node
               'min_samples_leaf': min_samples_leaf, # # min number of samples required to be at a leaf node
               'bootstrap': bootstrap
              }
    
    return random_grid

def randomizedsearchCV(estimator, X, y, n_iter_search, param_distributions, random_state):
    '''function that runs randomized search of input param_distributions'''
    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_AUC'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_AUC'][candidate],
                    results['std_test_AUC'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    # number of iterations 
    n_iter_search = n_iter_search
    # metric we're going to use for scoring
    scoring = {'AUC': 'roc_auc'}
    # run randomized search
    random_search = RandomizedSearchCV(estimator = estimator, param_distributions=param_distributions, 
                               n_iter=n_iter_search, scoring=scoring, cv=5, refit='AUC', verbose=1, random_state=random_state, n_jobs=-1)
    start = time()
    random_search.fit(X, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
    n_estimators = random_search.best_params_['n_estimators']
    min_samples_split = random_search.best_params_['min_samples_split']
    min_samples_leaf = random_search.best_params_['min_samples_leaf']
    max_features = random_search.best_params_['max_features']
    max_depth = random_search.best_params_['max_depth']
    bootstrap = random_search.best_params_['bootstrap']
    # print report of best cv results
    report(random_search.cv_results_)
                       
    return n_estimators, min_samples_split, min_samples_leaf, max_features, max_depth, bootstrap


def ml_bestparams(X, y, random_state, n_estimators, min_samples_split, min_samples_leaf, max_features, max_depth, bootstrap, title):
    '''function that splits data, fits baseline classifier, cross-validates and then prints out performance metrics'''
    # Split the data into 70% train and 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    # initiate classifier
    classifier = RandomForestClassifier(n_estimators = n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                        max_features=max_features, max_depth=max_depth, random_state=random_state, n_jobs=-1)
    # fit random forest classifier
    classifier.fit(X_train, y_train)
    # cross validate and record performance metrics: roc_auc, precision and recall
    classifier_scores = cross_validate(classifier, X_train, y_train, cv=5,
                                   scoring=['roc_auc', 'precision', 'recall', 'accuracy'],
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