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
                                               random_state=42)


#------Convert word to number using Vectorizer--------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer  
# Term Frequency–Inverse Document Frequency score

vect=TfidfVectorizer(stop_words='english',
                           max_features=5000)
X_train_vect=vect.fit_transform(X_train)
X_test_vect=vect.transform(X_test)


from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(X_train_vect,y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred=model.predict(X_test_vect)
                                           
accuracy=accuracy_score(y_test, y_pred)
clasfn_report=classification_report(y_test, y_pred)             
conf_matrix=confusion_matrix(y_test, y_pred)          



















