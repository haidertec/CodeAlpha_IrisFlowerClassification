# CodeAlpha - Iris Flower Classification

## 📌 Overview
End-to-end Machine Learning pipeline to classify Iris flowers into three species (Setosa, Versicolor, Virginica) using sepal and petal measurements.

## 🎯 Task
CodeAlpha Data Science Internship - Task 1

## 🛠 Techniques Used
- Feature Engineering (petal/sepal area ratios)
- 8 ML Models: KNN, Decision Tree, Random Forest, Extra Trees, SVM, Gradient Boosting, Logistic Regression, Gaussian NB
- Hyperparameter Tuning (RandomizedSearchCV + Stratified K-Fold)
- Ensemble Learning (Soft-Voting + Stacking)
- Complete Evaluation (ROC-AUC, F1-macro, MCC, LogLoss)

## 📊 Key Results
- Best test accuracy: ~98%
- Top feature: petal_sepal_area_ratio (from permutation importance)
- Stacking ensemble delivered best generalization

## 📈 Visualizations
- Correlation heatmap & feature distributions
- PCA & t-SNE cluster plots
- ROC & Precision-Recall curves
- Confusion matrix (counts + normalized)
- Calibration & learning curves
- Permutation feature importance

## 🧰 Tech Stack
Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Jupyter, Joblib

## 🏆 Author
Haider Tech — CodeAlpha Data Science Intern
