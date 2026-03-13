# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kagglehub

sns.set(style="whitegrid")


path = kagglehub.dataset_download("yungbyun/sensor-data")
print("Dataset path:", path)

files = os.listdir(path)
print("Files:", files)


csv_file = [f for f in files if f.endswith(".csv")][0]
data = pd.read_csv(os.path.join(path, csv_file))

print("\nOriginal Dataset:")
print(data.head())

# Clean Dataset Handle Missing Values
print("\nMissing values before cleaning:")
print(data.isnull().sum())

# Fill missing numeric values
data.fillna(data.mean(numeric_only=True), inplace=True)

print("\nMissing values after cleaning:")
print(data.isnull().sum())

# Outlier Removal
numeric_data = data.select_dtypes(include=np.number)

Q1 = numeric_data.quantile(0.25)
Q3 = numeric_data.quantile(0.75)
IQR = Q3 - Q1

data_cleaned = data[
    ~((numeric_data < (Q1 - 1.5 * IQR)) |
      (numeric_data > (Q3 + 1.5 * IQR))).any(axis=1)
]

print("\nCleaned Dataset No Missing Values Reduced Outliers:")
print(data_cleaned.head())

# Normalize 
data_norm = data_cleaned.select_dtypes(include=np.number)
data_norm = (data_norm - data_norm.mean()) / data_norm.std()

print("\nNormalized Numerical Features:")
print(data_norm.head())

# Histogram
data_norm.hist(figsize=(12, 8))
plt.suptitle("Histograms of Normalized Features", fontsize=14)
plt.tight_layout()
plt.show()

# Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=data_norm)
plt.xticks(rotation=45)
plt.title("Boxplots of Normalized Features Outliers")
plt.show()

# Scatter Plot 
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=data_norm.iloc[:, 0],
    y=data_norm.iloc[:, 1]
)
plt.xlabel(data_norm.columns[0])
plt.ylabel(data_norm.columns[1])
plt.title("Scatter Plot Showing Feature Relationship")
plt.show()
