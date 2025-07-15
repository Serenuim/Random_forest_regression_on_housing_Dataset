# Random_forest_regression_on_housing_Dataset

## 🧠 Random Forest Regressor Optimization

This project applies a Random Forest Regressor to predict housing prices using the California Housing dataset. The model was optimized using `RandomizedSearchCV` to improve predictive performance and generalization.

---

### ⚙️ Model Comparison: Before vs After Tuning

| Metric         | Before Tuning                | After Tuning                 | Improvement       |
|----------------|------------------------------|------------------------------|--------------------|
| **MSE**        | 2,671,284,726.57             | 2,365,123,679.31             | 🔽 ↓ ~306M         |
| **R² Score**   | 0.7080                       | 0.7415                       | 🔼 ↑ +0.0335        |
| **MBE**        | +36,748.08                   | +33,539.07                   | 🔽 ↓ ~3,209         |

> ✅ After tuning, the model explains **74.1%** of the variance in house prices and reduces both error and bias.

---

### 🧪 Random Search Hyperparameter Tuning

`RandomizedSearchCV` was used to explore different hyperparameter combinations for the Random Forest Regressor. The search was performed across the following space:

```python
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [5, 10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}
