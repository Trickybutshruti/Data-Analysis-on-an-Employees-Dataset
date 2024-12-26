import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('cleaned_data.csv')

# Step 1: Data Summary
print("Data Summary:")
print(df.info())  # Data types and non-null counts
print("\nDescriptive Statistics:")
print(df.describe())  # Summary statistics for numerical columns
print(df.columns)
# Step 2: Univariate Analysis
# 2a. Plot the distribution of 'Age'
plt.figure(figsize=(8, 6))
sns.histplot(df['age'], bins=15, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 2b. Plot the distribution of 'Salary'
plt.figure(figsize=(8, 6))
sns.histplot(df['salary'], bins=15, kde=True, color='green')
plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.show()

# Step 3: Bivariate Analysis
# 3a. Gender vs Salary (Boxplot)
plt.figure(figsize=(8, 6))
sns.boxplot(x='gender', y='salary', data=df, palette='Set2')
plt.title('Gender vs Salary')
plt.xlabel('Gender')
plt.ylabel('Salary')
plt.show()

# 3b. Age vs Salary (Scatter Plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='salary', data=df, color='purple')
plt.title('Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()

# 3c. Correlation heatmap for numeric features
plt.figure(figsize=(8, 6))
corr = df[['age', 'salary']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Step 4: Multivariate Analysis
# 4a. Pairplot of all numerical features
#df['gender'] = df['gender'].astype('category')  # Convert to categorical if needed
#sns.pairplot(df[['age', 'salary']], hue='gender', palette='Set1')
#plt.show()

# 4b. Countplot for Department distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='department', data=df, palette='Set3')
plt.title('Department Distribution')
plt.xlabel('Department')
plt.ylabel('Count')
plt.show()

# Step 5: Feature Engineering Visualization
# Show the relationship between 'Age Group' and 'Salary'
df['Age Group'] = pd.cut(df['age'], bins=[20, 25, 30, 35, 40, 45, 50, 55, np.inf],
                         labels=['20-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56+'])
plt.figure(figsize=(8, 6))
sns.boxplot(x='Age Group', y='salary', data=df, palette='muted')
plt.title('Age Group vs Salary')
plt.xlabel('Age Group')
plt.ylabel('Salary')
plt.show()
