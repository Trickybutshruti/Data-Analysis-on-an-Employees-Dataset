import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load the cleaned dataset 
df = pd.read_csv('cleaned_data.csv')
print(df.columns)


# --- Data Preprocessing ---
# Handling missing values (fill missing Age and Salary with the mean)
imputer = SimpleImputer(strategy='mean')
df['age'] = imputer.fit_transform(df[['age']])  # Fill missing Age with the mean
df['salary'] = imputer.fit_transform(df[['salary']])  # Fill missing Salary with the mean

# Encode categorical variables (e.g., Gender and Department)
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])  # Encode Gender (Male=0, Female=1)

# One-hot encoding for Department (since it's a categorical variable)
df = pd.get_dummies(df, columns=['department'], drop_first=True)

print("Columns after one-hot encoding:")
print(df.columns)

# Create the feature matrix (X) and target vector (y)
X = df[['age', 'gender', 'department_Finance',
       'department_HR', 'department_Marketing']]
y = df['salary']

# --- Split the Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Linear Regression Model ---
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regressor.predict(X_test)

# --- Model Evaluation ---
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) for Linear Regression: {mse}")

# --- Data Visualization ---

# 1. Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title("Actual vs Predicted Salaries")
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.show()

# 2. Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_test, residuals, color='blue')
plt.hlines(y=0, xmin=y_test.min(), xmax=y_test.max(), colors='red', linestyles='--')
plt.title("Residuals vs Actual Salaries")
plt.xlabel("Actual Salary")
plt.ylabel("Residuals")
plt.show()

# 3. Feature vs Target Plot (e.g., Age vs Salary)
plt.figure(figsize=(8, 6))
plt.scatter(df['age'], df['salary'], color='blue', label='Data Points')
plt.plot(df['age'], regressor.predict(df[['age', 'gender', 'department_Finance', 'department_HR', 'department_Marketing']]), color='red', label='Regression Line')
plt.title("Age vs Salary with Regression Line")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.legend()
plt.show()

# 4. Feature vs Target Plot (Gender vs Salary)
plt.figure(figsize=(8, 6))
# Using a box plot to visualize the salary distribution for each gender
sns.boxplot(x='gender', y='salary', data=df)
plt.title("Gender vs Salary")
plt.xlabel("Gender (0: Male, 1: Female)")
plt.ylabel("Salary")
plt.show()