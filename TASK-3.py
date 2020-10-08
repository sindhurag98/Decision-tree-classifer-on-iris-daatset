#!/usr/bin/env python
# coding: utf-8

# # TASK-3 PREDICTION USING DECISION TREE ALGORITHM

# **- By Sindhura Gundubogula**

# In this task, We create Decision tree Classifer for the well known "IRIS" dataset and visually it graphically. Our goal is if we feed any new data to this classifer it should predict the right class accordingly. 
# 
# Instead of using or creating any new data, I split the dataset into train and test.
#  
#  Dataset is available at URL: https://bit.ly/3kXTdox

# **STEP-1 import all required python libraries**

# In[26]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
sns.set()


# **STEP-2 Load the dataset**

# In[102]:


from sklearn.datasets import load_iris

iris = load_iris()


# In[103]:


iris 


# In[29]:


iris.info()


# Our forture is that the chosen dataset is clean and has no missis values or inconsistent data so we can skip data cleaning.

# **STEP-3 Building the Decision tree classifer**

# **Declare inputs and target**

# In[104]:


x = iris.data
y = iris.target


# In[105]:


x


# In[106]:


y


# **Split TRAIN and TEST data**

# In[239]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 6)


# **Fit a decision tree classifer with Train data**

# In[137]:


traintree=DecisionTreeClassifier()
traintree.fit(x_train,y_train)


# **Visualizing the Decisiontree**

# Install required libraries for better visualization. we used **pydotplus** and **graphviz**

# In[224]:


# Import necessary libraries for graph viz
from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


# In[225]:


# Visualize the graph
dot_data = StringIO()
export_graphviz(traintree, out_file=dot_data, feature_names=iris.feature_names, class_names = iris.target_names,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# **Estimating training prerdiction**

# In[141]:


y_hat = traintree.predict(x_train)#predictions are stored in y^(y_train should match y_hat)


# In[144]:


y_train


# In[145]:


y_hat


# In[153]:


# Comparing Actual vs Predicted
dt_performance = pd.DataFrame({'Actual_train': y_train, 'Predicted_train': y_hat})  
dt_performance


# In[159]:


dt_performance['Actual_train'].value_counts()


# In[160]:


dt_performance['Predicted_train'].value_counts()


# In[190]:


fig, axes = plt.subplots(1, 2,  sharex=True, figsize=(10,5))

sns.countplot(ax=axes[0],x='Actual_train', data=dt_performance)
    
sns.countplot(ax=axes[1],x='Predicted_train', data=dt_performance)


# our training predictions were accurate

# **STEP- 4 TESTING THE ALGORITHM**

# In[177]:


y_hat_test = traintree.predict(x_test)


# In[178]:


y_hat_test


# In[179]:


y_test


# In[180]:


# Comparing Actual vs Predicted
dt_test= pd.DataFrame({'Actual_test': y_test, 'Predicted_test': y_hat_test})  
dt_test


# In[182]:


dt_test['Actual_test'].value_counts()


# In[183]:


dt_test['Predicted_test'].value_counts()


# In[187]:


fig, axes = plt.subplots(1, 2,  sharex=True, figsize=(10,5))

sns.countplot(ax=axes[0],x='Actual_test',data=dt_test)
    
sns.countplot(ax=axes[1],x='Predicted_test', data=dt_test)


# We can see there is a slight difference between actual and predicted values of our test data

# **Plotting confusion matrix and calculating accuracy**

# In[230]:


from sklearn.metrics import confusion_matrix,plot_confusion_matrix, accuracy_score
from sklearn.svm import SVC

cm = confusion_matrix(y_test, y_hat_test)

print(cm)


# In[237]:


sns.heatmap(cm, annot=True)  


# In[238]:


print('PREDICTION ACCURACY : \033[1;32m{}'.format(accuracy_score(y_test, y_hat_test)))


# ** **end of algorithm** **
# 
