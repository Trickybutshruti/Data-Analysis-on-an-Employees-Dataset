import pandas as pd
import numpy as np

# Load the dataset (replace the path if needed)
df = pd.read_csv('sample.csv')

# Display initial data info
print("Initial Data Info:")
print(df.info())
print("\nInitial Data Preview:")
print(df.head())

# 1. Removing Duplicates
print("\nRemoving Duplicate Rows...")
df = df.drop_duplicates()

# 2. Handling Missing Values
# Display the count of missing values in each column
print("\nMissing Values Count:")
print(df.isnull().sum())

# Option 1: Fill missing values in numeric columns (mean for Age, Salary)
df['Age'] = df['Age'].fillna(df['Age'].mean())  # Fill missing Age with mean
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())  # Fill missing Salary with mean

# Option 2: Fill missing values in categorical columns (mode for Name, Gender)
df['Name'] = df['Name'].fillna('Unknown')  # Fill missing Name with 'Unknown'
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])  # Fill missing Gender with the mode (most frequent value)

# 3. Standardizing Data Types
# Ensure all columns have the appropriate data types (Age and Salary should be numeric)
df['Age'] = df['Age'].astype(int)
df['Salary'] = df['Salary'].astype(float)

# 4. Correcting Inconsistent Formatting
# Standardize column names (lowercase and replace spaces with underscores)
df.columns = df.columns.str.lower().str.replace(' ', '_')

# 5. Detecting and Handling Outliers (using IQR method for salary)
Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for detecting outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove rows where salary is an outlier
df_cleaned = df[(df['salary'] >= lower_bound) & (df['salary'] <= upper_bound)]

# 6. Saving Cleaned Data to a New CSV
df_cleaned.to_csv('cleaned_data.csv', index=False)

# Display the cleaned data info
print("\nCleaned Data Info:")
print(df_cleaned.info())
print("\nCleaned Data Preview:")
print(df_cleaned.head())
