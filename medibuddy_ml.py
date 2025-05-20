# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error

# Load data
data = pd.read_csv('D:/LABMENTIX INTERNSHIP/merged_data.csv')  # Updated path

# Print column names to debug
print("Available columns:", data.columns.tolist())

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# Check if 'sex' column exists
if 'sex' not in data.columns:
    raise KeyError("Column 'sex' not found in dataset. Please check for typos or missing data.")

# Encode categorical variables
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

data['sex'] = le_sex.fit_transform(data['sex'])  # male=1, female=0
data['smoker'] = le_smoker.fit_transform(data['smoker'])  # yes=1, no=0
data['region'] = le_region.fit_transform(data['region'])  # northeast=0, northwest=1, southeast=2, southwest=3

# Features and target
X = data[['sex', 'age', 'bmi', 'children', 'smoker', 'region']]
y = data['charges_in_inr']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initial Random Forest model
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
r2_initial = r2_score(y_test, y_pred)
rmse_initial = np.sqrt(mean_squared_error(y_test, y_pred))

print("Initial Model Performance:")
print(f"R² Score: {r2_initial:.2f}")
print(f"RMSE: {rmse_initial:.2f} INR")

# Optimized hyperparameter tuning with RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 150],  # Reduced search space
    'max_depth': [None, 10],  
    'min_samples_split': [2, 5],  
    'min_samples_leaf': [1, 2]  
}

random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_grid,
    n_iter=5,  # Reduced iterations for efficiency
    cv=2,  # Lowered cross-validation folds
    scoring='r2',
    n_jobs=1  # Limits parallel processing to prevent memory exhaustion
)

random_search.fit(X_train, y_train)

# Best model
best_rf = random_search.best_estimator_
y_pred_best = best_rf.predict(X_test)
r2_best = r2_score(y_test, y_pred_best)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))

print("\nTuned Model Performance:")
print(f"R² Score: {r2_best:.2f}")
print(f"RMSE: {rmse_best:.2f} INR")
print(f"Best Parameters: {random_search.best_params_}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save for Tableau
feature_importance.to_csv('D:/LABMENTIX INTERNSHIP/feature_importance.csv', index=False)

# Save predictions (optional)
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_best})
predictions.to_csv('D:/LABMENTIX INTERNSHIP/predictions.csv', index=False)

print("\nFiles saved: feature_importance.csv, predictions.csv")
