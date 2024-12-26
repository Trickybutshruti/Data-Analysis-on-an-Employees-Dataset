import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the dataset from sample.csv
df = pd.read_csv('sample.csv')

# Preview the data
print("Data Preview:")
print(df.head())

# --- Step 1: Handle Missing Values ---
# Impute missing values in 'Age' and 'Salary' using the mean
imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])
df['Salary'] = imputer.fit_transform(df[['Salary']])

# --- Step 2: Encode Categorical Variables ---

# Label encode 'Gender' (Male=0, Female=1)
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# One-Hot encode 'Department' (HR, Engineering, Finance, Marketing)
df = pd.get_dummies(df, columns=['Department'], drop_first=True)

# --- Step 3: Create New Features ---

# Create an 'Age Group' feature by categorizing 'Age' into groups
bins = [20, 25, 30, 35, 40, 45, 50, 55, np.inf]
labels = ['20-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56+']
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Create an interaction feature 'Age_Gender' as the product of 'Age' and 'Gender'
df['Age_Gender'] = df['Age'] * df['Gender']

# --- Step 4: Preview the Transformed Data ---
print("\nData after Feature Engineering:")
print(df)
