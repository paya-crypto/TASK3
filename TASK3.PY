import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load dataset
df = pd.read_csv("C:/Users/PAYAL MAHARANA/OneDrive/Documents/python/world_bank_data_2025.csv")

# Step 2: Preprocess dataset
# Separate numeric and categorical columns
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Fill missing values
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    if not df[col].mode().empty:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Remove non-numeric columns (categorical) before training
df_numeric = df[num_cols].copy()

# Optional: Check remaining nulls
print("\nRemaining Nulls:")
print(df_numeric.isnull().sum().sum())

# Step 3: Multiple Linear Regression
target = 'GDP (Current USD)'

if target in df_numeric.columns:
    X_multi = df_numeric.drop(columns=[target], errors='ignore')
    y_multi = df_numeric[target]

    # Train-test split
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

    # Fit model
    model_multi = LinearRegression()
    model_multi.fit(X_train_m, y_train_m)

    # Predict
    y_pred_m = model_multi.predict(X_test_m)

    # Evaluation
    print("\n📊 Multiple Linear Regression Results:")
    print(f"MAE: {mean_absolute_error(y_test_m, y_pred_m):.2f}")
    print(f"MSE: {mean_squared_error(y_test_m, y_pred_m):.2f}")
    print(f"R²: {r2_score(y_test_m, y_pred_m):.2f}")

    # Coefficients
    coeff_df = pd.DataFrame({
        'Feature': X_multi.columns,
        'Coefficient': model_multi.coef_
    })
    print("\nModel Coefficients:")
    print(coeff_df)

    # Plot Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test_m, y_pred_m, color='blue', alpha=0.6, edgecolor='k')
    plt.plot([y_test_m.min(), y_test_m.max()], [y_test_m.min(), y_test_m.max()], 'r--')
    plt.xlabel("Actual GDP")
    plt.ylabel("Predicted GDP")
    plt.title("Actual vs Predicted GDP (Multiple Regression)")
    plt.tight_layout()
    plt.show()

else:
    print(f"\n⚠️ Target column '{target}' not found in the dataset.")