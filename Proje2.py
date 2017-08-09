
# coding: utf-8

# In[32]:

import time
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as snsgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
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

dtype_df_test = test.dtypes.reset_index()
dtype_df_test.columns = ["Count", "Column Type"]
dtype_df_test


# In[6]:

dtype_df_train.groupby("Column Type").aggregate('count').reset_index()


# In[7]:

dtype_df_test.groupby("Column Type").aggregate('count').reset_index()


# In[8]:

for name in test.columns:
    if name not in train.columns:
        print(name)


# In[9]:

test = test.drop(['Inbound_Count_-12+Month', 'Inbound_Duration_-12+Month',
                  'Inbound_Count_Ever','Inbound_Duration_Ever'], axis=1)


# In[10]:

dtype_df_test = test.dtypes.reset_index()
dtype_df_test.columns = ["Count", "Column Type"]
dtype_df_test
dtype_df_test.groupby("Column Type").aggregate('count').reset_index()


# In[11]:

missing_df_train = train.isnull().sum(axis=0).reset_index()
missing_df_train.columns = ['column_name', 'missing_count']
missing_df_train['missing_ratio'] = missing_df_train['missing_count'] / train.shape[0]
missing_df_train.loc[missing_df_train['missing_count']>0]


# In[12]:

medianCityCode = train['citycode'].median(axis=0)
train['citycode'].fillna(medianCityCode, inplace=True)
medianGenderCode = train['GenderCode'].median(axis=0)
train['GenderCode'].fillna(medianGenderCode, inplace=True)
medianCustomerAge = train['CustomerAge'].median(axis=0)
train['CustomerAge'].fillna(medianCustomerAge, inplace=True)


# In[13]:

missing_df_train = train.isnull().sum(axis=0).reset_index()
missing_df_train.columns = ['column_name', 'missing_count']
missing_df_train['missing_ratio'] = missing_df_train['missing_count'] / train.shape[0]
missing_df_train.loc[missing_df_train['missing_count']>0]


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


# In[18]:

y_test = test['Payment1st3Months_F']
x_test = test.drop(['Payment1st3Months_F'], axis=1)


# In[19]:

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


# In[20]:

#Choose all predictors except target & IDcols
dt0 = DecisionTreeClassifier()
modelfit(dt0, y_train, x_train)


# In[21]:

y_pred_dt0 = dt0.predict(x_test)
y_pred_proba_dt0 = dt0.predict_proba(x_test)[:,1]
print('Accuracy for Decision Tree Classifier:')
print(accuracy_score(y_test, y_pred_dt0))
print('Confusion Matrix for Decision Tree Classifier')
print(confusion_matrix(y_test, y_pred_dt0))
print('F1 Score for Decision Tree Classifier')
print(f1_score(y_test, y_pred_dt0))
print('AUC Score for Decision Tree Classifier')
print(roc_auc_score(y_test, y_pred_proba_dt0))


# In[22]:

rf0 = RandomForestClassifier()
modelfit(rf0, y_train, x_train)


# In[26]:

y_pred_rf0 = rf0.predict(x_test)
y_pred_proba_rf0 = rf0.predict_proba(x_test)[:,1]
print('Accuracy for Random Forest Classifier:')
print(accuracy_score(y_test, y_pred_rf0))
print('Confusion Matrix for Random Forest Classifier')
print(confusion_matrix(y_test, y_pred_rf0))
print('F1 Score for Random Forest Classifier')
print(f1_score(y_test, y_pred_rf0))
print('AUC Score for Random Forest Classifier')
print(roc_auc_score(y_test, y_pred_proba_rf0))


# In[25]:

gbm0 = GradientBoostingClassifier()
modelfit(gbm0, y_train, x_train)


# In[27]:

y_pred_gbm0 = gbm0.predict(x_test)
y_pred_proba_gbm0 = gbm0.predict_proba(x_test)[:,1]
print('Accuracy for Gradient Boosting Classifier:')
print(accuracy_score(y_test, y_pred_gbm0))
print('Confusion Matrix for Gradient Boosting Classifier')
print(confusion_matrix(y_test, y_pred_gbm0))
print('F1 Score for Gradient Boosting Classifier')
print(f1_score(y_test, y_pred_rf0))
print('AUC Score for Gradient Boosting Classifier')
print(roc_auc_score(y_test, y_pred_proba_gbm0))


# In[29]:

abm0 = AdaBoostClassifier()
modelfit(abm0, y_train, x_train)


# In[30]:

y_pred_abm0 = abm0.predict(x_test)
y_pred_proba_abm0 = abm0.predict_proba(x_test)[:,1]
print('Accuracy for Adaboost Classifier:')
print(accuracy_score(y_test, y_pred_abm0))
print('Confusion Matrix for Adaboost Classifier')
print(confusion_matrix(y_test, y_pred_abm0))
print('F1 Score for Adaboost Classifier')
print(f1_score(y_test, y_pred_abm0))
print('AUC Score for Adaboost Classifier')
print(roc_auc_score(y_test, y_pred_proba_abm0))


# In[ ]:

xgb = AdaBoostClassifier()
modelfit(abm0, y_train, x_train)

