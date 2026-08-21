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
colum=df.columns              #Index(['label', 'text', 'text_length'], dtype='object')
df['label'].value_counts()    #   1    43910  ,   0    39538            0-ham     ,1-spam

import matplotlib.pyplot as plt

print(df['label'].value_counts())
df['label'].value_counts().plot(kind='bar')
plt.title('Spam-1 vs Ham-0 Count')
plt.show()
