
# coding: utf-8

# In[80]:

import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier


# In[36]:

train = pd.read_csv('train.csv')
train.info()


# In[37]:

test = pd.read_csv('test.csv')
test.info()


# In[38]:

dtype_df_train = train.dtypes.reset_index()
dtype_df_train.columns = ["Count", "Column Type"]
dtype_df_train


# In[39]:

dtype_df_test = test.dtypes.reset_index()
dtype_df_test.columns = ["Count", "Column Type"]
dtype_df_test


# In[40]:

dtype_df_train.groupby("Column Type").aggregate('count').reset_index()


# In[41]:

dtype_df_test.groupby("Column Type").aggregate('count').reset_index()


# In[42]:

for name in test.columns:
    if name not in train.columns:
        print(name)


# In[43]:

test = test.drop(['Inbound_Count_-12+Month', 'Inbound_Duration_-12+Month',
                  'Inbound_Count_Ever','Inbound_Duration_Ever'], axis=1)


# In[44]:

dtype_df_test = test.dtypes.reset_index()
dtype_df_test.columns = ["Count", "Column Type"]
dtype_df_test
dtype_df_test.groupby("Column Type").aggregate('count').reset_index()


# In[45]:

missing_df_train = train.isnull().sum(axis=0).reset_index()
missing_df_train.columns = ['column_name', 'missing_count']
missing_df_train['missing_ratio'] = missing_df_train['missing_count'] / train.shape[0]
missing_df_train.loc[missing_df_train['missing_count']>0]


# In[46]:

medianCityCode = train['citycode'].median(axis=0)
train['citycode'].fillna(medianCityCode, inplace=True)
medianGenderCode = train['GenderCode'].median(axis=0)
train['GenderCode'].fillna(medianGenderCode, inplace=True)
medianCustomerAge = train['CustomerAge'].median(axis=0)
train['CustomerAge'].fillna(medianCustomerAge, inplace=True)


# In[47]:

missing_df_train = train.isnull().sum(axis=0).reset_index()
missing_df_train.columns = ['column_name', 'missing_count']
missing_df_train['missing_ratio'] = missing_df_train['missing_count'] / train.shape[0]
missing_df_train.loc[missing_df_train['missing_count']>0]


# In[48]:

missing_df_test = test.isnull().sum(axis=0).reset_index()
missing_df_test.columns = ['column_name', 'missing_count']
missing_df_test['missing_ratio'] = missing_df_test['missing_count'] / test.shape[0]
missing_df_test.loc[missing_df_test['missing_count']>0]


# In[49]:

medianCityCode = test['citycode'].median(axis=0)
test['citycode'].fillna(medianCityCode, inplace=True)
medianGenderCode = test['GenderCode'].median(axis=0)
test['GenderCode'].fillna(medianGenderCode, inplace=True)
medianCustomerAge = test['CustomerAge'].median(axis=0)
test['CustomerAge'].fillna(medianCustomerAge, inplace=True)


# In[50]:

missing_df_test = test.isnull().sum(axis=0).reset_index()
missing_df_test.columns = ['column_name', 'missing_count']
missing_df_test['missing_ratio'] = missing_df_test['missing_count'] / test.shape[0]
missing_df_test.loc[missing_df_test['missing_count']>0]


# In[17]:

for i in range(len(train.columns)):
    plt.figure(figsize=(12,8))
    type(train.iloc[:,i])
    sns.distplot(train.iloc[:,i], bins=50, kde=False)
    plt.show()


# In[51]:

print('Number of labels that are 1: %i' 
      %train['Payment1st3Months_F'].sum(axis=0))
print('Number of all Labels: %i' 
      %train['Payment1st3Months_F'].count())


# In[52]:

print('Number of labels that are 1: %i' 
      %test['Payment1st3Months_F'].sum(axis=0))
print('Number of all Labels: %i' 
      %test['Payment1st3Months_F'].count())


# In[53]:

y_train = train['Payment1st3Months_F']
x_train = train.drop(['Payment1st3Months_F'], axis=1)


# In[54]:

y_test = test['Payment1st3Months_F']
x_test = test.drop(['Payment1st3Months_F'], axis=1)


# In[55]:

lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)
y_pred_proba = lr.predict_proba(x_test)[:,1]
print('Accuracy for Logistic Regression:')
print(accuracy_score(y_test, y_pred_lr))
print('Confusion Matrix for Logistic Regression')
print(confusion_matrix(y_test, y_pred_lr))
print('F1 Score for Logistic Regression')
print(f1_score(y_test, y_pred_lr))
print('AUC Score for Logistic Regression')
print(roc_auc_score(y_test, y_pred_proba))


# In[23]:

rf = RandomForestClassifier(n_estimators = 100)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
y_pred_proba = rf.predict_proba(x_test)[:,1]
print('Accuracy for Random Forest:')
print(accuracy_score(y_test, y_pred_rf))
print('Confusion Matrix for Random Forest')
print(confusion_matrix(y_test, y_pred_rf))
print('F1 Score for Random Forest')
print(f1_score(y_test, y_pred_rf))
print('AUC Score for Random Forest')
print(roc_auc_score(y_test, y_pred_proba))


# In[24]:

gbm = GradientBoostingClassifier()
gbm.fit(x_train, y_train)
y_pred_gbm = gbm.predict(x_test)
y_pred_proba_gbm = gbm.predict_proba(x_test)[:,1]
print('Accuracy for Gradient Boosting:')
print(accuracy_score(y_test, y_pred_gbm))
print('Confusion Matrix for Gradient Boosting')
print(confusion_matrix(y_test, y_pred_gbm))
print('F1 Score for Gradient Boosting')
print(f1_score(y_test, y_pred_gbm))
print('AUC Score for Gradient Boosting')
print(roc_auc_score(y_test, y_pred_proba_gbm))


# In[30]:

abm = AdaBoostClassifier()
abm.fit(x_train, y_train)
y_pred_abm = gbm.predict(x_test)
y_pred_proba_abm = abm.predict_proba(x_test)[:,1]
print('Accuracy for AdaBoost:')
print(accuracy_score(y_test, y_pred_abm))
print('Confusion Matrix for AdaBoost')
print(confusion_matrix(y_test, y_pred_abm))
print('F1 Score for AdaBoost')
print(f1_score(y_test, y_pred_abm))
print('AUC Score for AdaBoost')
print(roc_auc_score(y_test, y_pred_proba_abm))


# In[79]:

xgb = XGBClassifier()
xgb.fit(x_train, y_train)
y_pred_xgb = xgb.predict(x_test)
y_pred_proba_xgb = xgb.predict_proba(x_test)[:,1]
print('Accuracy for XGBoost:')
print(accuracy_score(y_test, y_pred_xgb))
print('Confusion Matrix for XGBoost')
print(confusion_matrix(y_test, y_pred_xgb))
print('F1 Score for XGBoost')
print(f1_score(y_test, y_pred_xgb))
print('AUC Score for XGBoost')
print(roc_auc_score(y_test, y_pred_proba_xgb))


# In[82]:

svm1 = svm.SVC()
svm1.fit(x_train, y_train)
y_pred_svm = svm1.predict(x_test)
y_pred_proba_svm = svm1.predict_proba(x_test)[:,1]
print('Accuracy for Support Vector Machine:')
print(accuracy_score(y_test, y_pred_svm))
print('Confusion Matrix for Support Vector Machine')
print(confusion_matrix(y_test, y_pred_svm))
print('F1 Score for Support Vector Machine')
print(f1_score(y_test, y_pred_svm))
print('AUC Score for Support Vector Machine')
print(roc_auc_score(y_test, y_pred_proba_svm))


# In[26]:

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 
print('Original dataset shape {}'.format(Counter(y_train)))


# In[27]:

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(x_train, y_train)
print('Resampled dataset shape {}'.format(Counter(y_res)))


# In[28]:

lr = LogisticRegression()
lr.fit(X_res, y_res)
y_pred_lr = lr.predict(x_test)
print('Accuracy for Logistic Regression:')
print(accuracy_score(y_test, y_pred_lr))
print('Confusion Matrix for Logistic Regression')
print(confusion_matrix(y_test, y_pred_lr))
print('F1 Score for Logistic Regression')
print(f1_score(y_test, y_pred_lr))


# In[29]:

rf = RandomForestClassifier(n_estimators = 100)
rf.fit(X_res, y_res)
y_pred_rf = rf.predict(x_test)
print('Accuracy for Random Forest:')
print(accuracy_score(y_test, y_pred_rf))
print('Confusion Matrix for Random Forest')
print(confusion_matrix(y_test, y_pred_rf))
print('F1 Score for Random Forest')
print(f1_score(y_test, y_pred_rf))

