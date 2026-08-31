# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 15:13:43 2026

@author: ASUS
"""

import pandas as  pd
import numpy as np

df=pd.read_csv('combined_data.csv')

df.duplicated().sum()         # zero
df.isna().sum()               #0
colum=df.columns              
#Index(['label', 'text', 'text_length'], dtype='object')
df['label'].value_counts()   
 #   1    43910  ,   0    39538            0-ham     ,1-spam

import matplotlib.pyplot as plt

print(df['label'].value_counts())
df['label'].value_counts().plot(kind='bar')
plt.title('Spam-1 vs Ham-0 Count')
plt.show()

from sklearn.model_selection import train_test_split

X=df['text']
y=df['label']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, 
                                               random_state=42,stratify=y)
#stratify makes sure both train and test get the same ratio of spam-to-ham

#------Convert word to number using Vectorizer--------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer  
# Term Frequency–Inverse Document Frequency score

vect=TfidfVectorizer(stop_words='english',
                           max_features=5000)
X_train_vect=vect.fit_transform(X_train)
X_test_vect=vect.transform(X_test)

#--------------------------------------Logistic-regression----------------------------------------


from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(X_train_vect,y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred=model.predict(X_test_vect)
                                           
accuracy=accuracy_score(y_test, y_pred)
clasfn_report=classification_report(y_test, y_pred)             
conf_matrix=confusion_matrix(y_test, y_pred)          

print (conf_matrix,"\n\n", accuracy,'\n\n', clasfn_report)

#--------------------------------------Naivebayes------------------------

from sklearn.naive_bayes import MultinomialNB

model2=MultinomialNB()
model2.fit(X_train_vect,y_train)
y2_pred=model2.predict(X_test_vect)

accuracy2=accuracy_score(y_test, y2_pred)
clasfn_report2=classification_report(y_test, y2_pred)
cm2=confusion_matrix(y_test, y2_pred)

print (cm2,"\n\n", accuracy2,'\n\n', clasfn_report2)

#--------------------------------------Using pipeline--------------------------


from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

Final_models={
              'logisctic_regression':LogisticRegression(max_iter=1000),
              'naive_bayes': MultinomialNB(),
              'decision_tree': DecisionTreeClassifier(random_state=42),
              'random_forest': RandomForestClassifier(random_state=42,n_estimators=100),
              'knn':KNeighborsClassifier(n_neighbors=5),
              'svc':SVC(kernel='linear',random_state=42)
              }


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, 
                                               random_state=42,stratify=y)
results=[]


for name , model in Final_models.items():
    

    pipe=Pipeline([(
        'tfidf',TfidfVectorizer(stop_words='english',max_features=5000)),
        ('classifier', model)
        ])

    pipe.fit(X_train,y_train)
    y_predict=pipe.predict(X_test)
    
    acc=accuracy_score(y_test, y_predict)
    report_classfn=classification_report(y_test, y_predict)
    conf_matrix=confusion_matrix(y_test, y_predict)
    
    print(f"{name}\nAccuracy: {acc}\n\n{report_classfn}\n\n{conf_matrix}\n{'-'*60}")

    results.append({'Model': name, 'Accuracy': acc})





results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
print(results_df)











