# Titanic Survival Prediction - Assignment 6

This project analyzes the Titanic dataset to predict passenger survival using various machine learning models. The workflow includes data exploration, preprocessing, feature engineering, model training, evaluation, and interpretation.

## Dataset
- **Source:** `titanic.csv`
- **Features:** PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
- **Target:** Survived (0 = did not survive, 1 = survived)
- **Size:** 891 rows, 12 columns

## Problem Statement
Predict whether a passenger survived the Titanic disaster based on their features (binary classification).

## Workflow
1. **Dataset Description & Problem Definition**
   - Overview of features, target, and assumptions (missing values, feature limitations).
2. **Exploratory Data Analysis (EDA)**
   - Descriptive statistics, visualizations, and discussion of data issues (missingness, skewness, outliers).
3. **Preprocessing**
   - Handling missing values, encoding categorical variables, feature engineering, and transformations.
4. **Feature Selection & Data Balancing**
   - Correlation, tree-based importances, mutual information, and class balancing (class_weight, SMOTE).
5. **Modeling & Evaluation**
   - Models: Logistic Regression, Decision Tree, Random Forest, XGBoost
   - Evaluation: Stratified 5-Fold CV, accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and plots.
6. **Model Comparison & Interpretation**
   - Compare models, discuss bias-variance, feature importance, and select the best model.
7. **Manual Hyperparameter Tuning**
   - Stepwise tuning of Random Forest, with results and justification.

## Results
- **Best Model:** Random Forest (mean accuracy > 0.83, macro F1 > 0.82, ROC-AUC > 0.86)
- **XGBoost:** Also strong (accuracy 0.809, macro F1 0.798, ROC-AUC 0.861)
- **Key Features:** Sex, Pclass, Fare, Age, FamilySize, Title

## Requirements
- Python 3.x
- pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, imbalanced-learn

Install requirements (if needed):
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```
