# Iris Classification - Project Report

## 1. Objective
Classify Iris flowers into 3 species using ML.

## 2. Dataset
- 150 samples
- 4 original features
- 3 classes (setosa, versicolor, virginica)

## 3. Methodology
- Feature engineering: 4 → 35 features
- AutoML: 20 models tested
- Genetic Algorithm optimization
- Multi-objective Pareto optimization
- Deep Ensemble (uncertainty)
- Adversarial robustness
- Conformal prediction
- Fairness analysis
- Data drift detection
- SHAP + Permutation + PDP
- Hyperparameter sensitivity
- Model registry

## 4. Results
- Best Model: Decision Tree
- Test Accuracy: 100%
- CV Accuracy: ~95%
- Deep Ensemble: 82.61%

## 5. Key Findings
- pl_sq (petal length squared) most important feature
- No data drift detected
- Fairness concern: demographic parity diff 0.3889

## 6. Files
- Code: iris_ultra_fast.py
- Model: iris_ultra\models\DT_v20260927_220030.pkl
- Plots: 6 PNG files
- Prediction: predict_saved_model.py