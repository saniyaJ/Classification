# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 12:22:12 2020

@author: SaniyaJaswani
"""


# =============================================================================
# Logistic Regression
# =============================================================================



# =============================================================================
# Business Case -- 
#The dataset comes from the UCI Machine Learning repository, and it is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. 
#The classification goal is to predict whether the client will subscribe (1/0) to a term deposit (variable y).
# =============================================================================


# =============================================================================
# Setting the Environment
# =============================================================================
import os
import pandas as pd
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)



# # Importing the dataset
os.chdir('C:/Users/SaniyaJaswani/Desktop/dataScience/Python')
data = pd.read_csv('Banking.csv', header=0)
DataCopy = data
data = data.dropna()
print(data.shape)
print(list(data.columns))

# # Exploratory Data Analysis

# 1. Predict variable (desired target)
# y — has the client subscribed a term deposit? (binary: “1”, means “Yes”, “0” means “No”)

#Barplot for the dependent variable
sns.countplot(x='y',data=data, palette='hls')
plt.show()


#Check the missing values
data.isnull().sum()

#Customer job distribution
sns.countplot(y="job", data=data)
plt.show()


#Customer marital status distribution
sns.countplot(x="marital", data=data)
plt.show()


#Barplot for credit in default
sns.countplot(x="default", data=data)
plt.show()

#Barplot for housing loan
sns.countplot(x="housing", data=data)
plt.show()


#Barplot for personal loan
sns.countplot(x="loan", data=data)
plt.show()


#Barplot for previous marketing loan outcome
sns.countplot(x="poutcome", data=data)
plt.show()



# Our prediction will be based on the customer’s job, marital status, whether he(she) has credit in default, 
#whether he(she) has a housing loan, whether he(she) has a personal loan, and the outcome of the previous marketing campaigns. 
#So, we will drop the variables that we do not need.


#Dropping the reduant columns
data.drop(data.columns[[0, 3, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]], axis=1, inplace=True)

#Creating Dummy Variables
dataDummy = pd.get_dummies(data, columns =['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])

#Drop the unknown columns
dataDummy.drop(dataDummy.columns[[12, 16, 18, 21, 24]], axis=1, inplace=True)
dataDummy.columns

#Check the independence between the independent variables
sns.heatmap(dataDummy.corr())
plt.show()


# =============================================================================
# Split the data into training and test sets
# =============================================================================
X = dataDummy.iloc[:,1:]
y = dataDummy.iloc[:,0]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train.shape

# =============================================================================
# Synthetic Minority Oversampling Technique (SMOTE) to solve the problem of Imbalanced Data
# =============================================================================


#Works by creating synthetic samples from the minor class (no-subscription) instead of creating copies.
#Randomly choosing one of the k-nearest-neighbors and using it to create a similar, but randomly tweaked, new observations.


data_new=pd.get_dummies(DataCopy, columns =['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome'])
data_new.columns.values


X = data_new.loc[:, data_new.columns != 'y']
y = data_new.loc[:, data_new.columns == 'y']

from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))

#Now we have a balanced Data
# =============================================================================
# Recursive Feature Elimination for selecting Important Variables
# =============================================================================

data_final_vars=data_new.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)



cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 'default_no', 'default_unknown', 
      'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"] 
X=os_data_X[cols]
y=os_data_y['y']


import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


# =============================================================================
# Fitting the Logistic Model
# =============================================================================

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_pred_train = classifier.predict(X_train)


# =============================================================================
# Evaluating the Logistic Model
# =============================================================================


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# =============================================================================
# Interpretation:Interpretation: Of the entire test set, 88% of the promoted term deposit were the term deposit that the customers liked. Of the entire test set, 90% of the customer’s preferred term deposits that were promoted.
# =============================================================================


##Computing false and true positive rates
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr,_=roc_curve(classifier.predict(X_test),y_test,drop_intermediate=False)

import matplotlib.pyplot as plt
##Adding the ROC
##Random FPR and TPR

##Title and label
plt.plot(fpr, tpr, color='red',
 lw=2, label='ROC curve')


roc_auc_score(classifier.predict(X_test),y_test)
















