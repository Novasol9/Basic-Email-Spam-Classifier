\# Basic Email Spam Classifier



A machine learning project that classifies emails as spam or ham (not spam) using text data, comparing multiple classification algorithms to find the best performer.



\## Overview



This project covers the full ML workflow for a text classification problem:

\- Exploratory Data Analysis (EDA) on email text data

\- Text vectorization (TF-IDF)

\- Model training and evaluation across 6 different algorithms

\- Side-by-side comparison using a scikit-learn Pipeline



\## Dataset



The dataset (`combined\_data.csv`) contains labeled emails:



| Label | Meaning | Count |

|-------|---------|-------|

| 0     | Ham (not spam) | 39,538 |

| 1     | Spam | 43,910 |



Columns: `label`, `text`, `text\_length`



> Note: `combined\_data.csv` is not included in this repository (exceeds GitHub's file size limit). See \[Requirements](#requirements) for where to obtain it.



\## Project Structure



```

email-spam-classifier/

├── email\_spam\_classifier.py   # Main script: EDA, vectorization, model training \& comparison

├── README.md

└── .gitignore                 # excludes the large dataset file

```



\## Requirements



```

pandas

numpy

matplotlib

scikit-learn

```



Install with:

```bash

pip install pandas numpy matplotlib scikit-learn

```



Dataset used: \[Email Spam Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset) — download and place `combined\_data.csv` in the project root before running.



\## What the script does



1\. \*\*Load \& clean data\*\* — checks for duplicates and missing values.

2\. \*\*EDA\*\* — visualizes spam vs. ham class balance.

3\. \*\*Train/test split\*\* — 80/20 split, stratified to preserve the spam/ham ratio in both sets.

4\. \*\*Vectorization\*\* — converts raw email text into numeric features using `TfidfVectorizer` (5000 features, English stop words removed).

5\. \*\*Baseline models\*\* — Logistic Regression and Naive Bayes trained individually for a first look.

6\. \*\*Model comparison via Pipeline\*\* — six algorithms (Logistic Regression, Naive Bayes, Decision Tree, Random Forest, KNN, SVM) each wrapped in a `Pipeline` with their own vectorizer, trained and evaluated identically for a fair comparison.



\## Results



| Model                | Accuracy |

|-----------------------|----------|

| SVM (linear kernel)   | 0.9866   |

| Random Forest         | 0.9855   |

| Logistic Regression   | 0.9826   |

| Naive Bayes           | 0.9651   |

| Decision Tree         | 0.9648   |

| KNN                   | 0.8554   |



\*\*SVM performed best\*\*, closely followed by Random Forest and Logistic Regression — all three comfortably above 98% accuracy. \*\*KNN performed noticeably worse\*\*, likely due to the high dimensionality of the TF-IDF feature space, where distance-based methods tend to struggle.



\## Branches



\- `main` — stable pipeline with baseline models

\- `model-comparison` — adds the full 6-model Pipeline comparison



\## Next Steps



\- Hyperparameter tuning (GridSearchCV) on the top-performing models

\- Try additional text features (e.g. n-grams, character counts)

\- Save the best model with `joblib` for reuse on new emails

\- Build a simple script/interface to classify a single new email

