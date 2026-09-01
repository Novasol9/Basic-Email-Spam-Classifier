# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 18:45:52 2026

@author: ASUS
"""

import joblib

loaded_model = joblib.load('svm_spam_classifier.pkl')

# Try it on a brand new email
new_email =input  ("Please enter your e-mail to check for spam:")
prediction = loaded_model.predict([new_email])
print("\nthis is SPAM" if prediction[0] == 1 else "HAM")

