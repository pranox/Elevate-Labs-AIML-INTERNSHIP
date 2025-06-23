# Task 1: Data Cleaning & Preprocessing
# Internship: AI & ML - Elevate Labs

# 🔹 Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 🔹 Step 2: Load the dataset
df = pd.read_csv('train.csv')
print("First 5 rows of the dataset:")
print(df.head())

# 🔹 Step 3: Explore basic information
print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# 🔹 Step 4: Handle missing values
# ✅ FIXED: Use assignment instead of inplace to avoid FutureWarning
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop 'Cabin' due to many missing values
df = df.drop(columns=['Cabin'])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 🔹 Step 5: Encode categorical features
# One-hot encoding for 'Sex' and 'Embarked'
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# 🔹 Step 6: Feature scaling (standardization)
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# 🔹 Step 7: Visualize and remove outliers
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Fare'])
plt.title('Boxplot of Fare (after scaling)')
plt.show()

# Remove extreme outliers in 'Fare' (scaled above 3)
df = df[df['Fare'] < 3]

# 🔹 Step 8: Final cleaned data info
print("\nFinal dataset shape:", df.shape)

# 🔹 Step 9: Save cleaned dataset
df.to_csv('cleaned_titanic.csv', index=False)
print("✅ Cleaned dataset saved as 'cleaned_titanic.csv'")
