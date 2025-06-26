import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 1: Load dataset
df = pd.read_csv("C:/Users/PAYAL MAHARANA/OneDrive/Documents/python/world_bank_data_2025.csv")

# Step 2: Basic Info
print("First 5 rows:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Step 3: Summary Statistics
print("\n📈 Summary Statistics:")
print(df.describe(include='all'))

# Step 4: Handle missing values
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    if not df[col].mode().empty:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Step 5: Encode categorical variables
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Step 6: Outlier Removal (IQR method on selected numerical columns)
selected_cols = num_cols[:5]  # Adjust as needed
for col in selected_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

# Step 7: Feature Scaling
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Step 8: Define features (X) and target (y)
target = 'GDP (Current USD)'  # 🔁 Change this if you want to predict a different column
if target not in df.columns:
    raise ValueError(f"Column '{target}' not found in dataset. Please select a valid target column.")

X = df.drop(columns=[target])
y = df[target]

# Step 9: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 11: Predict and evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# Step 12: Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual GDP")
plt.ylabel("Predicted GDP")
plt.title("Actual vs Predicted GDP")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 13: Interpret Coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\n📈 Top Influencing Features on GDP Prediction:")
print(coefficients.head(10))