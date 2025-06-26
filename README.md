# 🌍 World Bank Economic Data Analysis – Linear Regression

This project explores and applies *simple and multiple linear regression* on world economic indicators to understand and predict GDP (Current USD) using Python and machine learning tools.

---

## 🎯 Objective

- Implement and understand both *simple* and *multiple* linear regression models.
- Analyze the relationship between GDP and other economic indicators.
- Evaluate the effectiveness of various features in predicting a country's GDP.

---

## 🛠 Tools & Libraries

- pandas, numpy – Data manipulation
- matplotlib, seaborn – Visualization
- scikit-learn – Machine learning (Linear Regression, Metrics)

---

## 📚 Dataset

- Source: World Bank economic indicators
- Target Variable: *GDP (Current USD)*
- Features: Multiple numeric indicators (Inflation, Unemployment Rate, Tax Revenue, Public Debt, etc.)

---

## 🔍 Project Workflow

### 1. Data Preprocessing
- Handled missing values using median (numeric) and mode (categorical)
- Removed non-numeric columns (like country names) to ensure compatibility with regression

### 2. Exploratory Data Analysis
- Histograms, Boxplots, Pairplots
- Correlation matrix to assess relationships

### 3. Regression Models

#### 🔹 Simple Linear Regression:
- One feature used: *Unemployment Rate (%)*
- Visualized regression line against GDP

#### 🔹 Multiple Linear Regression:
- All available numeric features used (except GDP itself)
- Model trained and evaluated using:
  - *MAE (Mean Absolute Error)*
  - *MSE (Mean Squared Error)*
  - *R² (R-squared score)*

---

## 📈 Results

### ✔ Simple Linear Regression:
- Gives a visual understanding of how one feature (like Unemployment Rate) affects GDP
- Useful for interpretation, but limited predictive power

### ✔ Multiple Linear Regression:
- More robust, uses all numeric features
- Provides *model coefficients* indicating feature importance
- Produces significantly better prediction metrics

---
