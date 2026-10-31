
# Ex. No. 4a – Machine Learning Model: Linear Regression

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine_Learning-orange?logo=scikitlearn)
![Experiment](https://img.shields.io/badge/Experiment-Completed-success)

## AIM

To predict house prices using regression models and compare the performance of different machine-learning regression models based on RMSE, MAE, and R².

## OBJECTIVES

- Load and inspect the House Price Dataset.
- Perform exploratory data analysis (EDA).
- Examine relationships between house features and prices.
- Detect and treat extreme house-price outliers.
- Split the dataset into training and testing sets.
- Train and evaluate ten regression models.
- Compare models using RMSE, MAE, and R².
- Visualize predictions, residuals, and feature importance.

## SOFTWARE REQUIREMENTS

| Component | Description |
|---|---|
| Programming language | Python 3.x |
| Data analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine learning | Scikit-learn |
| Environment | Google Colab or Jupyter Notebook |
| Dataset | `house_prices_dataset.csv` |

## DATASET DESCRIPTION

The House Price Dataset contains information about residential properties. The objective is to predict the continuous numerical value of a house's price from its characteristics.

![Modern residential house](https://images.squarespace-cdn.com/content/v1/5cc1cfa57d0c917c71c34845/1556647321295-HD0D8JAH17L2Y6JQC5PC/540-Pleasant-New-Exterior-2.jpg)

| Feature | Description |
|---|---|
| `square_feet` | Size of the house in square feet |
| `num_rooms` | Number of rooms |
| `age` | Age of the house in years |
| `distance_to_city(km)` | Distance from the city centre in kilometres |
| `price` | Target variable representing house price |

## PROBLEM STATEMENT

Develop a machine-learning model to predict house prices using house characteristics. Train different regression algorithms, evaluate their prediction performance, and select a better-performing model based on standard regression metrics.

## THEORY

### 1. Machine Learning

Machine Learning is a branch of artificial intelligence that enables computers to learn patterns from existing data and make predictions without being explicitly programmed for every situation.

In supervised learning, a model learns from labeled examples containing input features and known target values.

### 2. Regression

Regression is a supervised machine-learning technique used to predict continuous numerical values.

In this experiment:

- **Independent variables (`X`):** `square_feet`, `num_rooms`, `age`, and `distance_to_city(km)`
- **Dependent variable (`y`):** `price`

### 3. Linear Regression

Linear Regression models the relationship between input features and a continuous target using a linear equation.

For multiple input features:

\[
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_nx_n
\]

where:

- \(y\) is the predicted house price
- \(x_1, x_2, \ldots, x_n\) are input features
- \(\beta_0\) is the intercept
- \(\beta_1, \beta_2, \ldots, \beta_n\) are model coefficients

The model estimates coefficients that minimize the sum of squared prediction errors.

![House price versus size with a fitted regression line](https://datasciencebook.ca/_main_files/figure-html/08-lm-predict-all-1.png)

### 4. Regression Models Used

| Model | Description |
|---|---|
| Linear Regression | Models a linear relationship between features and target |
| Ridge Regression | Uses L2 regularization to control coefficient magnitudes |
| Lasso Regression | Uses L1 regularization and can reduce coefficients to zero |
| ElasticNet Regression | Combines L1 and L2 regularization |
| Polynomial Regression | Adds polynomial features to model nonlinear relationships |
| Decision Tree Regressor | Predicts using feature-based decision rules |
| Random Forest Regressor | Combines predictions from multiple decision trees |
| Gradient Boosting Regressor | Builds sequential models to improve prediction errors |
| Support Vector Regressor (SVR) | Uses support-vector principles for regression |
| K-Nearest Neighbors (KNN) Regressor | Predicts from nearby training observations |

### 5. Exploratory Data Analysis

Exploratory Data Analysis (EDA) is used to understand the dataset before model training.

- **Histograms:** Examine feature distributions.
- **Correlation heatmap:** Visualizes relationships between numerical variables.
- **Scatter plots:** Examine relationships between individual features and house price.
- **Box plots:** Help identify unusually low or high observations.

### 6. Outlier Treatment

Outliers are observations that differ substantially from the majority of the data. Extreme house prices may influence some regression models.

In this experiment, house prices below the 1st percentile or above the 99th percentile are removed.

### 7. Train-Test Split

The dataset is divided into:

- **Training set:** 80% of the data, used to train models.
- **Testing set:** 20% of the data, used to evaluate predictions.

The experiment uses `random_state=42` to make the split reproducible.

### 8. Feature Scaling

`StandardScaler` standardizes features using:

\[
z = \frac{x-\mu}{\sigma}
\]

where \(x\) is the original value, \(\mu\) is the training-set mean, and \(\sigma\) is the training-set standard deviation.

The scaler is fitted on training data and then applied to testing data.

### 9. Evaluation Metrics

#### Root Mean Squared Error (RMSE)

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

RMSE penalizes larger prediction errors more strongly. A lower value indicates better performance.

#### Mean Absolute Error (MAE)

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

MAE represents the average absolute prediction error. A lower value indicates better performance.

#### R² Score

\[
R^2 = 1-\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
{\sum_{i=1}^{n}(y_i-\bar{y})^2}
\]

R² measures how well the model explains variation in the target. A higher value generally indicates better performance.

## WORKFLOW

```text
                ┌──────────────────────────┐
                │          START           │
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │ Import libraries and     │
                │ load house-price dataset │
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │ Inspect and summarize   │
                │ dataset                 │
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │ Perform exploratory     │
                │ data analysis           │
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │ Detect and treat        │
                │ price outliers          │
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │ Separate features (X)   │
                │ and target (y)          │
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │ Split training and      │
                │ testing data            │
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │ Scale input features    │
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │ Train ten regression    │
                │ models                  │
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │ Evaluate using RMSE,    │
                │ MAE, and R²             │
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │ Compare models and      │
                │ visualize results       │
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │           END            │
                └──────────────────────────┘
```

## PROCEDURE

1. Import the required Python libraries.
2. Load the House Price Dataset using Pandas.
3. Display the dataset and inspect its shape, information, and descriptive statistics.
4. Check for missing values.
5. Plot feature distributions using histograms.
6. Generate a correlation heatmap and feature-versus-price scatter plots.
7. Use box plots to inspect possible outliers.
8. Remove house prices below the 1st percentile and above the 99th percentile.
9. Separate the independent features and target variable.
10. Split the dataset into training and testing sets.
11. Apply `StandardScaler` to the input features.
12. Train the baseline and ten regression models.
13. Evaluate each model using RMSE, MAE, and R².
14. Compare model results and visualize predictions, residuals, and feature importance.

## PROGRAM

> **Dataset requirement:** Place `house_prices_dataset.csv` in the specified Google Drive folder or update the file path in the program.

```python
# Ex. No. 4a - MACHINE LEARNING MODEL: LINEAR REGRESSION

# Step 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Step 2: Load the dataset
df = pd.read_csv(
    '/content/drive/MyDrive/Datasets/house_prices_dataset.csv'
)

print("First five records:")
print(df.head())

# Step 3: Dataset overview
print("\nDataset information:")
df.info()

print("\nDataset shape:")
print(df.shape)

print("\nDescriptive statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# Step 4: Exploratory Data Analysis
numeric_features = [
    'square_feet',
    'num_rooms',
    'age',
    'distance_to_city(km)',
    'price'
]

# Feature distributions
for col in numeric_features:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    df[numeric_features].corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Feature versus price scatter plots
input_features = [
    'square_feet',
    'num_rooms',
    'age',
    'distance_to_city(km)'
]

for col in input_features:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df[col], y=df['price'])
    plt.title(f'{col} vs Price')
    plt.tight_layout()
    plt.show()

# Step 5: Outlier detection using box plots
for col in numeric_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()

# Step 6: Outlier treatment
Q1 = df['price'].quantile(0.01)
Q99 = df['price'].quantile(0.99)

df = df[
    (df['price'] >= Q1) &
    (df['price'] <= Q99)
].copy()

print("\nDataset shape after outlier treatment:", df.shape)

# Step 7: Define features and target
X = df[
    [
        'square_feet',
        'num_rooms',
        'age',
        'distance_to_city(km)'
    ]
]

y = df['price']

# Step 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Step 9: Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 10: Baseline model
y_pred_baseline = np.full(
    shape=len(y_test),
    fill_value=y_train.mean()
)

# Step 11: Train regression models
models = {}

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
models["Linear Regression"] = lr.predict(X_test_scaled)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
models["Ridge"] = ridge.predict(X_test_scaled)

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
models["Lasso"] = lasso.predict(X_test_scaled)

# ElasticNet Regression
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train_scaled, y_train)
models["ElasticNet"] = elastic.predict(X_test_scaled)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_lr = LinearRegression()
poly_lr.fit(X_train_poly, y_train)
models["Polynomial Regression"] = poly_lr.predict(X_test_poly)

# Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train_scaled, y_train)
models["Decision Tree"] = dt.predict(X_test_scaled)

# Random Forest Regressor
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train_scaled, y_train)
models["Random Forest"] = rf.predict(X_test_scaled)

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
gbr.fit(X_train_scaled, y_train)
models["Gradient Boosting"] = gbr.predict(X_test_scaled)

# Support Vector Regressor
svr = SVR(
    kernel='rbf',
    C=100,
    gamma=0.1,
    epsilon=0.1
)
svr.fit(X_train_scaled, y_train)
models["SVR"] = svr.predict(X_test_scaled)

# K-Nearest Neighbors Regressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
models["KNN"] = knn.predict(X_test_scaled)

# Step 12: Evaluate models
results = []

# Evaluate baseline
baseline_rmse = np.sqrt(
    mean_squared_error(y_test, y_pred_baseline)
)
baseline_mae = mean_absolute_error(
    y_test, y_pred_baseline
)
baseline_r2 = r2_score(
    y_test, y_pred_baseline
)

results.append([
    "Mean Baseline",
    baseline_rmse,
    baseline_mae,
    baseline_r2
])

# Evaluate regression models
for name, y_pred in models.items():
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append([name, rmse, mae, r2])

results_df = pd.DataFrame(
    results,
    columns=["Model", "RMSE", "MAE", "R2"]
)

results_df = results_df.sort_values(
    by="RMSE"
).reset_index(drop=True)

print("\nModel comparison:")
print(results_df.to_string(index=False))

results_df.to_csv(
    "regression_model_comparison.csv",
    index=False
)

# Step 13: RMSE comparison graph
plt.figure(figsize=(10, 6))
sns.barplot(
    x="RMSE",
    y="Model",
    data=results_df
)
plt.title("RMSE Comparison Across Regression Models")
plt.tight_layout()
plt.show()

# Step 14: Actual versus predicted plots
plt.figure(figsize=(15, 12))

for i, (name, y_pred) in enumerate(models.items()):
    plt.subplot(5, 2, i + 1)

    plt.scatter(y_test, y_pred, alpha=0.5)

    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        'r--'
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(name)

plt.tight_layout()
plt.show()

# Step 15: Residual analysis
for name, y_pred in models.items():
    residuals = y_test - y_pred

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')

    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title(f"{name}: Residual Plot")

    plt.tight_layout()
    plt.show()

# Step 16: Feature importance
feature_names = X.columns

# Random Forest feature importance
plt.figure(figsize=(6, 4))
sns.barplot(
    x=rf.feature_importances_,
    y=feature_names
)
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

# Gradient Boosting feature importance
plt.figure(figsize=(6, 4))
sns.barplot(
    x=gbr.feature_importances_,
    y=feature_names
)
plt.title("Gradient Boosting Feature Importance")
plt.tight_layout()
plt.show()

# Step 17: Display best-performing model
best_model = results_df.iloc[0]

print("\nBest model according to RMSE:")
print("Model:", best_model["Model"])
print("RMSE:", round(best_model["RMSE"], 2))
print("MAE:", round(best_model["MAE"], 2))
print("R2:", round(best_model["R2"], 4))

print("\nExperiment completed successfully.")
```

## MODEL COMPARISON

The models are compared using the following criteria:

| Metric | Preferred value | Interpretation |
|---|---|---|
| RMSE | Lower | Smaller squared prediction errors |
| MAE | Lower | Smaller average absolute errors |
| R² | Higher | More variation in house prices explained |

The program sorts the comparison table by RMSE and saves it as `regression_model_comparison.csv`.

### RMSE Comparison

The generated graph compares the RMSE values of the baseline and regression models. The model with the lowest RMSE has the smallest root mean squared prediction error on the test set.

### Actual vs. Predicted Prices

The scatter plots compare actual house prices with model predictions. Points closer to the diagonal reference line indicate more accurate predictions.

### Residual Analysis

A residual is the difference between the actual and predicted values:

\[
\text{Residual} = \text{Actual Price} - \text{Predicted Price}
\]

Residual plots help identify prediction errors and patterns that may indicate a poor model fit.

### Feature Importance

Random Forest and Gradient Boosting provide feature-importance estimates that help identify which input features contribute most to their predictions.

## RESULT

House-price prediction was performed using Linear Regression and nine additional regression models. The models were evaluated and compared using RMSE, MAE, and R². Exploratory data analysis, outlier treatment, feature scaling, prediction visualization, residual analysis, and feature-importance analysis were also performed.

The best-performing model is identified by the lowest RMSE in the generated comparison table.

## REFERENCES

1. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
2. [Pandas Documentation](https://pandas.pydata.org/docs/)
3. [NumPy Documentation](https://numpy.org/doc/)
4. [Matplotlib Documentation](https://matplotlib.org/stable/)
5. [Seaborn Documentation](https://seaborn.pydata.org/)
6. [Data Science: A First Introduction — Regression](https://datasciencebook.ca/regression2.html)
