#!/usr/bin/env python
# coding: utf-8

# # PREDICTION OF DIABETES 

# ## IMPORTING LIBRARIES

# In[1]:


import numpy as np
import pandas as pd
import sklearn as skl 
import seaborn as sns
import os
import matplotlib.pyplot as plt


# In[2]:


os.getcwd()


# In[3]:


os.chdir("C://Users//lenovo//Downloads")
os.getcwd()


# ## DATA CLEANING IN EXCEL

# ##### The data consisted of various missing values for features Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin and BMI.
# ##### I dropped 15 entries where values were missing for more than 3 features. 
# ##### For the rest of the missing values, I've assumed that they're missing at random.
# ##### Hence, imputation can be used to deal with these missing values. 
# ##### For the rest of the variables, I substituted the missing values with median of the given observations and imported the excel file thereafter. 
# 

# In[100]:


db = pd.read_excel("db.xlsx")


# In[101]:


db


# In[102]:


db.describe()


# ### Countplot for patients who tested negative and positive for the diabetes test 

# In[103]:


sns.countplot(x = db['Outcome'],palette = 'RdBu_r');


# 

# In[104]:


db.groupby('Outcome').size().plot(kind='pie',autopct='%.3f',colors=['skyblue','yellow'])


# ### Piechart displays 65.3% tested negative and 34.7% tested positive for the test

# ### Correlation matrix for the dataset

# In[137]:


sns.heatmap(db.corr(), annot=True,cmap='Blues');


# ### The correlation matrix heatmap displays that Glucose has the highest correlation with the Outcome.               Correlation amongst independent variables isn't high, hence, it's safe to say that multicollinearity is not present.
# 

# ## DEFINING X & Y

# In[140]:


X = db.drop(columns=['Outcome'], axis=1)
Y = db['Outcome']


# ## SCALING AND SPLITTING THE DATASET

# #### Scaling of independent variabels is necessary as the units in which all of these are measured is different. Features are scaled to standard normal using the StandardScaler.
# 

# In[141]:


from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler


# #### Splitting the dataset into training and testing sets using test size proportion as 0.33

# In[116]:


xt, xt2, yt, yt2 = tts(X,Y,test_size=0.33,random_state=101)


# #### Scaling the training and testing X variables

# In[117]:


SS = StandardScaler()
xt = SS.fit_transform(xt)
xt2 = SS.transform(xt2)


# #### The dataset is cleaned, scaled and model is ready to be trained. 

# ### There are various ways to fit a model to the data. I've chosen Random Forest Classifier

# In[118]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report, ConfusionMatrixDisplay


# ## TRAIN THE MODEL

# ### Random forests uses various Decision Trees. Here, I've taken 200 Decision Trees to obtain a Random Forest.

# In[119]:


classf = RandomForestClassifier(n_estimators=200,random_state=101)
classf.fit(xt,yt)


# ## PREDICT THE TEST SET RESULTS

# In[120]:


pred = classf.predict(xt2)
pred


# ### PROBABILITY ATTACHED TO THE PREDICTION

# In[142]:


probs = classf.predict_proba(xt2)[:,1]
probs


# ## CONFUSION MATRIX

# In[122]:


CM = confusion_matrix(yt2,pred,labels=classf.classes_)


# In[148]:


sns.heatmap(CM,annot=True,fmt='0.1f',annot_kws={'size':18})
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual');


# ### Sensitivity = 0.67
# ### Specificity = 0.88

# In[149]:


print(classification_report(yt2,pred))


# ### To check which feature has the most impact on the Outcome

# In[150]:


classf.feature_importances_


# #### Glucose followed by BMI impact the Outcome highly as compared to the other features.

# ## EVALUATING THE MODEL

# In[126]:


classf.score(xt2,yt2)


# #### Model shows an accuracy of 80.32%

# ## PLOTTING ROC CURVE AND AREA UNDER THE CURVE

# In[127]:


from sklearn.metrics import roc_auc_score, roc_curve, auc


# In[128]:


fpr,tpr,threshold = roc_curve(yt2,probs)
roc_auc = auc(fpr,tpr)
print(roc_auc)
plt.plot(fpr,tpr,label='AUC=%0.2f'%roc_auc,color='green')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'b--')
plt.xlim([0,1])
plt.ylim([0,1.1])
plt.xlabel('FalsePositiveRate')
plt.ylabel('TruePositiveRate')
plt.show()


# ## FINDING THRESHOLD USING YOUDEN'S J STATISTIC  

# #### Youden's index is used in conjunction with receiver operating characteristic (ROC) analysis. The index is defined for all points of an ROC curve, and the maximum value of the index may be used as a criterion for selecting the optimum cut-off point for the a diagnostic test. 

# #### Finding values of True positive rate and False Positive rate for which the distance between the two values is maximum to obtain the cut off.

# In[154]:


idx = np.argmax(tpr-fpr)
print(threshold[idx])


# #### Threshold value/ Cut-Off for the test is 0.29
# 
# #### If Outcome > 0.29, Diabetic
# #### If Outcome < 0.29, Non- Diabetic
# 

# In[ ]:




