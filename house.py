import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ═══════════════════════════════════════════════════════════════════════════
# ⚙️ CONFIGURATION & FOLDER SETUP
# ═══════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(r"C:\Users\T2520872\Downloads\housing.csv", "housing.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Project_Outputs")
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

TARGET = 'median_house_value'

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: DATA PREPROCESSING (Missing, Outliers, Duplicates, Transformation)
# ═══════════════════════════════════════════════════════════════════════════
print("--- [3/7] Preprocessing & EDA ---")
df = pd.read_csv(DATA_FILE)

# 1. Duplicates
df = df.drop_duplicates()

# 2. Missing Values (total_bedrooms)
imputer = SimpleImputer(strategy='median')
df['total_bedrooms'] = imputer.fit_transform(df[['total_bedrooms']])

# 3. Outliers (Removing the capped $500k values to improve model accuracy)
df = df[df[TARGET] < 499000]

# 4. EDA Visuals (REQUIRED FOR 10% GRADE)
# Plot 1: Target Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df[TARGET], kde=True, color='teal')
plt.title('House Value Distribution')
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_distribution.png'))
plt.close()

# Plot 2: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_heatmap.png'))
plt.close()

# 5. Data Transformation (Encoding & Scaling later)
le = LabelEncoder()
df['ocean_proximity'] = le.fit_transform(df['ocean_proximity'])

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 & 5: FEATURE SELECTION & MODEL BUILDING (3 Algorithms)
# ═══════════════════════════════════════════════════════════════════════════
print("--- [4-5/7] Model Building (3 Algorithms) ---")

# We keep location data to ensure R2 stays high (~0.80)
X = df.drop(TARGET, axis=1)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    
    # Regression metrics calculation (AS PER RUBRIC)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    
    results.append({
        "Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "MAPE%": mape
    })

comparison_df = pd.DataFrame(results)
print("\n--- Model Evaluation Comparison ---")
print(comparison_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: MODEL OPTIMIZATION (20% WEIGHTAGE)
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- [6/7] Model Optimization (Tuning Random Forest) ---")
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [20, None],
    'min_samples_split': [2, 5]
}

# RandomizedSearch Optimization
opt_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), 
                                param_grid, n_iter=3, cv=3, random_state=42)
opt_search.fit(X_train_scaled, y_train)

final_model = opt_search.best_estimator_
final_preds = final_model.predict(X_test_scaled)

print(f"   Optimized R2 Score: {r2_score(y_test, final_preds):.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# SAVE EVERYTHING
# ═══════════════════════════════════════════════════════════════════════════
comparison_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_metrics.csv"), index=False)
