
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
#from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
get_ipython().magic(u'matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12,4


# In[2]:

train = pd.read_csv('train.csv')
train.info()


# In[3]:

test = pd.read_csv('test.csv')
test.info()


# In[4]:

dtype_df_train = train.dtypes.reset_index()
dtype_df_train.columns = ["Count", "Column Type"]
dtype_df_train


# In[5]:

missing_df_train = train.isnull().sum(axis=0).reset_index()
missing_df_train.columns = ['column_name', 'missing_count']
missing_df_train['missing_ratio'] = missing_df_train['missing_count'] / train.shape[0]
missing_df_train.loc[missing_df_train['missing_count']>0]


# In[6]:

medianCityCode = train['citycode'].median(axis=0)
train['citycode'].fillna(medianCityCode, inplace=True)
medianGenderCode = train['GenderCode'].median(axis=0)
train['GenderCode'].fillna(medianGenderCode, inplace=True)
medianCustomerAge = train['CustomerAge'].median(axis=0)
train['CustomerAge'].fillna(medianCustomerAge, inplace=True)


# In[7]:

missing_df_train = train.isnull().sum(axis=0).reset_index()
missing_df_train.columns = ['column_name', 'missing_count']
missing_df_train['missing_ratio'] = missing_df_train['missing_count'] / train.shape[0]
missing_df_train.loc[missing_df_train['missing_count']>0]


# In[8]:

dtype_df_test = test.dtypes.reset_index()
dtype_df_test.columns = ["Count", "Column Type"]
dtype_df_test


# In[9]:

dtype_df_train.groupby("Column Type").aggregate('count').reset_index()


# In[10]:

dtype_df_test.groupby("Column Type").aggregate('count').reset_index()


# In[11]:

for name in test.columns:
    if name not in train.columns:
        print(name)


# In[12]:

test = test.drop(['Inbound_Count_-12+Month', 'Inbound_Duration_-12+Month',
                  'Inbound_Count_Ever','Inbound_Duration_Ever'], axis=1)


# In[13]:

dtype_df_test = test.dtypes.reset_index()
dtype_df_test.columns = ["Count", "Column Type"]
dtype_df_test
dtype_df_test.groupby("Column Type").aggregate('count').reset_index()


# In[14]:

missing_df_test = test.isnull().sum(axis=0).reset_index()
missing_df_test.columns = ['column_name', 'missing_count']
missing_df_test['missing_ratio'] = missing_df_test['missing_count'] / test.shape[0]
missing_df_test.loc[missing_df_test['missing_count']>0]


# In[15]:

medianCityCode = test['citycode'].median(axis=0)
test['citycode'].fillna(medianCityCode, inplace=True)
medianGenderCode = test['GenderCode'].median(axis=0)
test['GenderCode'].fillna(medianGenderCode, inplace=True)
medianCustomerAge = test['CustomerAge'].median(axis=0)
test['CustomerAge'].fillna(medianCustomerAge, inplace=True)


# In[16]:

missing_df_test = test.isnull().sum(axis=0).reset_index()
missing_df_test.columns = ['column_name', 'missing_count']
missing_df_test['missing_ratio'] = missing_df_test['missing_count'] / test.shape[0]
missing_df_test.loc[missing_df_test['missing_count']>0]


# In[17]:

y_train = train['Payment1st3Months_F']
x_train = train.drop(['Payment1st3Months_F'], axis=1)
y_test = test['Payment1st3Months_F']
x_test = test.drop(['Payment1st3Months_F'], axis=1)


# In[18]:

def modelfit(alg, y_train, x_train, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm
    alg.fit(x_train, y_train)
    #Predict training set
    dtrain_predictions = alg.predict(x_train)
    dtrain_predprob = alg.predict_proba(x_train)[:,1]
    #Perform Cross-Validation
    if performCV:
        cv_score = cross_val_score(alg, x_train, y_train, cv=cv_folds, scoring='roc_auc', n_jobs=3)
    #Print Model Report
    print "\nModel Report"
    print "Accuracy: %.4g" %accuracy_score(y_train.values, dtrain_predictions)
    print "AUC Score (Train): %f" %roc_auc_score(y_train, dtrain_predprob)
    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g"         % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, x_train.columns).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')


# In[19]:

#Choose all predictors except target & IDcols
gbm0 = GradientBoostingClassifier(random_state=10)
modelfit(gbm0, y_train, x_train)


# In[20]:

#Choose all predictors except target & IDcols
start = time.time()
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator                        = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,
                                                   max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
param_grid = param_test1, scoring='roc_auc',n_jobs=6,iid=False, cv=5)
gsearch1.fit(x_train, y_train)
end = time.time()
print(end - start)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# In[21]:

start = time.time()
param_test2 = {'max_depth':range(5,16,2)}
gsearch2 = GridSearchCV(estimator                         = GradientBoostingClassifier(learning_rate=0.1, n_estimators=50, max_features='sqrt',
                                                     subsample=0.8, random_state=10), 
param_grid = param_test2, scoring='roc_auc',n_jobs=6,iid=False, cv=5)
gsearch2.fit(x_train,y_train)
end = time.time()
print(end - start)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_


# In[22]:

start = time.time()
param_test3 = {'min_samples_split':range(200,1001,200)}
gsearch3 = GridSearchCV(estimator                         = GradientBoostingClassifier(learning_rate=0.1, n_estimators=50, max_features='sqrt',
                                                     subsample=0.8, random_state=10), 
param_grid = param_test3, scoring='roc_auc',n_jobs=6,iid=False, cv=5)
gsearch3.fit(x_train,y_train)
end = time.time()
print(end - start)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_


# In[23]:

start = time.time()
param_test4 = {'min_samples_leaf':range(30,71,10)}
gsearch4 = GridSearchCV(estimator                         = GradientBoostingClassifier(learning_rate=0.1, n_estimators=50,max_depth=9,
                                                     max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test4, scoring='roc_auc',n_jobs=6,iid=False, cv=5)
gsearch4.fit(x_train,y_train)
end = time.time()
print(end - start)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_


# In[24]:

start = time.time()
param_test5 = {'max_features':range(7,20,2)}
gsearch5 = GridSearchCV(estimator                         = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9, 
                                                     min_samples_split=1200, min_samples_leaf=60, subsample=0.8, 
                                                     random_state=10),
param_grid = param_test4, scoring='roc_auc',n_jobs=6,iid=False, cv=5)
gsearch5.fit(x_train,y_train)
end = time.time()
print(end - start)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_


# In[25]:

#Choose all predictors except target & IDcols
gbm5 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=50, min_samples_split=1000,min_samples_leaf=70,
                                                   max_depth=9,max_features='sqrt',subsample=0.8,random_state=10)
modelfit(gbm0, y_train, x_train)


# In[26]:

gbm5.fit(x_test, y_test)
y_pred_gbm = gbm5.predict(x_test)
y_pred_proba_gbm = gbm5.predict_proba(x_test)[:,1]
print('Accuracy for Gradient Boosting:')
print(accuracy_score(y_test, y_pred_gbm))
print('Confusion Matrix for Gradient Boosting')
print(confusion_matrix(y_test, y_pred_gbm))
print('F1 Score for Gradient Boosting')
print(f1_score(y_test, y_pred_gbm))
print('AUC Score for Gradient Boosting')
print(roc_auc_score(y_test, y_pred_proba_gbm))

