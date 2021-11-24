# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 01:31:11 2021

@author: ABHIJOT KAUR
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#use seaborn plotting style defaults
import seaborn as sns; sns.set()
#from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score

from sklearn import svm


#upload files into Colaboratory
#uploaded = files.upload()

#pip install openpyxl

#read cvs file into dfframe
#df = pd.read_excel('Dry_Bean_Dataset.xlsx',engine='openpyxl')
#df = pd.read_csv('iris.csv', index_col=0)
#print(df.head())
#df.columns
df = pd.read_excel('iris.xls')

df.drop('Id',axis=1,inplace=True)
labels,levels=pd.factorize(df['Species'])
df['Species']=labels
df = df.rename(columns={'SepalLengthCm': 'SL','SepalWidthCm': 'SW','PetalLengthCm': 'PL','PetalWidthCm': 'PW'})
print(df.head())
#df.drop(['Species',],axis=1,inplace=True)


#df=df[:10000]

df.isnull().sum()
#df.drop(['address',],axis=1,inplace=True)


from sklearn.decomposition import PCA

standard_deviations = 3


from scipy import stats

def drop_numerical_outliers(df, z_thresh=2):
    # Constrains will contain `True` or `False` depending on if it is a value below the threshold.
    constrains = df.select_dtypes(include=[np.number]) \
        .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh) \
        .all(axis=1)
    # Drop (inplace) values set to be rejected
    df.drop(df.index[~constrains], inplace=True)
    
drop_numerical_outliers(df)    

#df = df[df.apply(lambda x: np.abs(x - x.mean()) / x.std() < standard_deviations)
#   .all(axis=1)]


Y= df['Species']

df.drop(['Species',],axis=1,inplace=True)

for i in range(0, len(df.columns)):
	df.iloc[:,i] = pd.to_numeric(df.iloc[:,i], errors='ignore')
    
df = (df - df.mean())/df.std()

from sklearn.metrics import accuracy_score,classification_report

print('\n')
print('LogisticRegression with StandardScaler prediction results ---->\n')
   

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#discplay coefficients
X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.2)

pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
pipe_prediction = pipe.predict(X_test)


# # But Confusion Matrix and Classification Report give more details about performance
print(confusion_matrix(pipe_prediction, y_test))
print(classification_report(pipe_prediction, y_test))







print('\n')
print('svm prediction results ---->\n')
   
SVC_model = svm.SVC()
SVC_model.fit(X_train, y_train)
SVC_prediction = SVC_model.predict(X_test)
print(accuracy_score(SVC_prediction, y_test))
print(confusion_matrix(SVC_prediction, y_test))


print('\n')
print('KNeighborsClassifier prediction results ---->\n')
   


KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train, y_train)
KNN_prediction = KNN_model.predict(X_test)


print(accuracy_score(KNN_prediction, y_test))
print(classification_report(KNN_prediction, y_test))



# #
# #MM = MultinomialNB()
# #MM.fit(X_train, y_train)
# #MM_pred = MM.predict(X_test)
# #print(accuracy_score(MM_pred, y_test))


# from sklearn.tree import DecisionTreeClassifier

# DT = DecisionTreeClassifier()
# DT.fit(X_train, y_train)
# DT_pred = DT.predict(X_test)
# print(accuracy_score(DT_pred, y_test))

# from sklearn.ensemble import BaggingClassifier
# from sklearn.tree import DecisionTreeClassifier

# bg=BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=20)
# bg.fit(X_train, y_train)
# bg_pred = bg.predict(X_test)
# print(accuracy_score(bg_pred, y_test))

# from sklearn.ensemble import AdaBoostClassifier

# adb = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=10,max_depth=4),n_estimators=10,learning_rate=0.6)
# adb.fit(X_train, y_train)
# adb_pred = adb.predict(X_test)
# print(accuracy_score(adb_pred, y_test))


# from sklearn.ensemble import RandomForestClassifier
# # n_estimators = number of decision trees
# rf = RandomForestClassifier(n_estimators=30, max_depth=9)
# rf.fit(X_train, y_train)
# rf_pred = rf.predict(X_test)
# print(accuracy_score(rf_pred, y_test))

# from sklearn.ensemble import VotingClassifier

# clf1 = LogisticRegression(multi_class='ovr', random_state=1)
# clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
# clf3 = GaussianNB()

# evc=VotingClassifier(estimators=[('lr',clf1),('rf',clf2)],voting='hard')
# evc.fit(X_train, y_train)
# evc_pred = evc.predict(X_test)
# print(accuracy_score(evc_pred, y_test))


# from numpy import mean
# from numpy import std
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score

# cv = KFold(n_splits=10, random_state=1, shuffle=True)
# # create model
# logisticRegr = LogisticRegression()


# clf = svm.SVC(kernel='linear', C=1, random_state=42)
# scores = cross_val_score(clf, X_train, y_train, cv=10)

# print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))



# from sklearn import model_selection
# from sklearn.ensemble import BaggingClassifier
# from sklearn.tree import DecisionTreeClassifier

# seed = 7
# kfold = model_selection.KFold(n_splits=10, random_state=seed)
# cart = DecisionTreeClassifier()
# num_trees = 100

# model = BaggingClassifier(base_estimator=cart,n_estimators=num_trees, random_state=seed)

# results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
# print(results.mean())

# evc_pred = model.predict(X_test)
# print(accuracy_score(evc_pred, y_test))

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5,shuffle=(True))
logisticRegr = LogisticRegression()

print('\n')
print('Logistic Regression Kfolds prediction results ---->\n')
    
for train_ix, test_ix in kfold.split(df):
    #X_train, X_test = df[train_ix], df[test_ix]
    X_train, X_test = df.loc[np.intersect1d(df.index, train_ix)], \
                df.loc[np.intersect1d(df.index, test_ix)]
    y_train, y_test = Y[np.intersect1d(Y.index, train_ix)], \
        Y[np.intersect1d(Y.index, test_ix)]
   
    
    logisticRegr.fit(X_train, y_train)
    logisticRegr_prediction = logisticRegr.predict(X_test)
    
    # Accuracy score is the simplest way to evaluate

    print('Accuracy = ', accuracy_score(y_test,logisticRegr_prediction))
    #print(confusion_matrix(y_test,logisticRegr_prediction))
    #print(classification_report(y_test,logisticRegr_prediction))
    #print('Precision = ',precision_score(y_test,logisticRegr_prediction))
    #print('Recall = ',recall_score(y_test,logisticRegr_prediction))
   