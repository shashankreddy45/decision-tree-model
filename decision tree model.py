#!/usr/bin/env python
# coding: utf-8

# In[168]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.tree import plot_tree
from matplotlib import rcParams
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz


# In[169]:


dataset_train= pd.read_csv('CE802_P2_DATA.csv')


# In[170]:


dataset_train


# In[171]:


dataset_train.isnull()


# In[172]:


dataset_train.isnull().sum()


# In[173]:


dataset_train['F21'] = dataset_train['F21'].fillna(dataset_train['F21'].mean())


# In[174]:


dataset_train.head()


# In[175]:


dataset_train.shape


# In[176]:


dataset_train.head(20)


# In[177]:


dataset_train.tail()


# In[178]:


dataset_train.info()


# In[179]:


X = dataset_train.drop(columns = ['Class'])
Y = dataset_train['Class']
X_train,X_valid,Y_train,Y_valid = train_test_split(X,Y,test_size = 0.26)


# In[180]:


Y_train.value_counts(normalize=True)


# In[181]:


Y_valid.value_counts(normalize=True)


# In[182]:


#shape of training set
X_train.shape,Y_train.shape


# In[183]:


#shape of validation set
X_valid.shape,Y_valid.shape


# In[184]:


#creating the decision tree model.
#random state gives the same reult every time we run the model
model = DecisionTreeClassifier(random_state = 10)




# In[185]:


#fitting the model
model.fit(X_train, Y_train)


# In[186]:


#checking the validation score
model.score(X_valid,Y_valid)


# In[187]:


#checking the training score
model.score(X_train,Y_train)


# In[188]:


model.predict(X_valid)


# In[189]:


#model.predict_proba(X_valid)


# In[190]:


#changing the max depth
train_accuracy = []
validation_accuracy = []
for depth in range(1,10):
    model = DecisionTreeClassifier(max_depth = depth,random_state=10,min_samples_split=2,min_samples_leaf=1,max_leaf_nodes=None)
    model.fit(X_train,Y_train)
    train_accuracy.append(model.score(X_train,Y_train))
    validation_accuracy.append(model.score(X_valid,Y_valid))


# In[191]:


fr = pd.DataFrame({'max_depth':range(1,10),'train_acc':train_accuracy,'valid_acc':validation_accuracy})
fr


# In[192]:


plt.figure(figsize = (12,6))
plt.plot(fr['max_depth'], fr['train_acc'], marker = '*')
plt.plot(fr['max_depth'], fr['valid_acc'], marker = '*')
plt.xlabel('depth of tree')
plt.ylabel('performance')
plt.legend()


# In[193]:


#in this case we can see that at max depth of 5 it is producing highest validation accuracy.
model = DecisionTreeClassifier(max_depth = 5,random_state=10)
model.fit(X_train, Y_train)
model.score(X_valid,Y_valid)


# In[194]:


model.score(X_train,Y_train)


# In[195]:


decision_tree = tree.export_graphviz(model,out_file = 'tree.dot',feature_names = X_train.columns,max_depth = 5,filled = True)


# In[196]:


get_ipython().system(' dot -Tpng tree.dot -o tree.png')


# In[220]:


img = plt.imread('tree.png')
plt.figure(figsize=(80,80))
plt.imshow(img)


# In[198]:


model = DecisionTreeClassifier(class_weight = 'balanced',random_state=88)
grid = {
        "max_depth":np.arange(14)+1,
        'criterion':['gini','entropy']
        }
grid_cv_dt = GridSearchCV(estimator=model, param_grid=grid, cv=5,verbose=1)
grid_cv_dt.fit(X_train, Y_train)


# In[199]:


grid_cv_dt.best_params_


# In[200]:


grid_cv_dt.best_estimator_


# In[201]:


def classifcation_report_train_test(y_train, y_train_pred, y_test, y_test_pred):

    print('''
            =========================================
               CLASSIFICATION REPORT FOR TRAIN DATA
            =========================================
            ''')
    print(classification_report(y_train, y_train_pred))

    print('''
            =========================================
               CLASSIFICATION REPORT FOR TEST DATA
            =========================================
            ''')
    print(classification_report(y_test, y_test_pred))


# In[202]:


dt_predict_train=grid_cv_dt.best_estimator_.predict(X_train)
dt_predict_test=grid_cv_dt.best_estimator_.predict(X_valid)
classifcation_report_train_test(Y_train, dt_predict_train, Y_valid, dt_predict_test)


# In[203]:


#in this case we can see that at max depth of 5 it is producing highest validation accuracy.
model = DecisionTreeClassifier(max_depth = 5,random_state=10)
model.fit(X_train, Y_train)
model.score(X_valid,Y_valid)


# In[204]:


model.score(X_train,Y_train)


# In[205]:


decision_tree = tree.export_graphviz(model,out_file = 'tree.dot',feature_names = X_train.columns,max_depth = 5,filled = True)


# In[206]:


get_ipython().system(' dot -Tpng tree.dot -o tree.png')


# In[221]:


img = plt.imread('tree.png')
plt.figure(figsize=(60,60))
plt.imshow(img)


# In[208]:


dataset_test = pd.read_csv('CE802_P2_TEST.CSV')


# In[209]:


dataset_test


# In[210]:


test = dataset_test.drop(columns = ['Class'])
test.head()


# In[211]:


test


# In[212]:


test['F21'] = test['F21'].fillna(test['F21'].mean())


# In[213]:


test_pred = grid_cv_dt.best_estimator_.predict(test)


# In[214]:


test_pred


# In[215]:


test['Class'] = test_pred
test.head()


# In[ ]:




