import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the Excel file
file_path = "UW_Churn_Pred_Data.xls"
xls = pd.ExcelFile(file_path)

# print(xls.sheet_names)

# Load B30 Pro sheet into a DataFrame
b30_pro = pd.read_excel(xls, sheet_name='B30 Pro')

# Display the first few rows of the B30 Pro sheet
print(b30_pro.head())

# Check the shape of the B30 Pro data
print(f"B30 Pro Shape: {b30_pro.shape}")

# Basic info and column types
print(b30_pro.info())

# Check for missing values
print(b30_pro.isnull().sum()) # no missing values

# Check for duplicates
duplicates = b30_pro.duplicated().sum()
print(f"Number of duplicate rows in B30 Pro: {duplicates}") # no duplicates

# Check data types and basic statistics
print(b30_pro.info())
print(b30_pro.describe())

# Explore the distribution of 'churn_flag'
if 'Churn Flag' in b30_pro.columns:
    churn_counts = b30_pro['Churn Flag'].value_counts()
    print(f"Churn Flag Distribution:\n{churn_counts}")
    # Plot the distribution
    churn_counts.plot(kind='bar', title='Churn Flag Distribution')
    plt.show()

# Separate the churned (1) and non-churned (0) records
churned = b30_pro[b30_pro['Churn Flag'] == 1]
non_churned = b30_pro[b30_pro['Churn Flag'] == 0]

# Visualize the distribution of 'last boot - interval' for churned vs non-churned
plt.figure(figsize=(12, 6))
sns.histplot(churned['last boot - interval'], color='red', label='Churned', kde=True, stat='density', bins=30)
sns.histplot(non_churned['last boot - interval'], color='blue', label='Non-Churned', kde=True, stat='density', bins=30)
plt.title('Distribution of Last Boot Interval: Churned vs Non-Churned')
plt.legend()
plt.show()

# Visualize the distribution of 'last boot - active' for churned vs non-churned
plt.figure(figsize=(12, 6))
sns.histplot(churned['last boot - active'], color='red', label='Churned', kde=True, stat='density', bins=30)
sns.histplot(non_churned['last boot - active'], color='blue', label='Non-Churned', kde=True, stat='density', bins=30)
plt.title('Distribution of Last Boot Active: Churned vs Non-Churned')
plt.legend()
plt.show()

# Visualize the distribution of 'return - activate' for churned vs non-churned
plt.figure(figsize=(12, 6))
sns.histplot(churned['return - activate'], color='red', label='Churned', kde=True, stat='density', bins=30)
sns.histplot(non_churned['return - activate'], color='blue', label='Non-Churned', kde=True, stat='density', bins=30)
plt.title('Distribution of Return Activate: Churned vs Non-Churned')
plt.legend()
plt.show()

"""
T-test: statistical test used to compare the means of two groups to determine if they are significantly different from each other
"""
# T-test for 'last boot - interval' between churned and non-churned
t_stat_interval, p_val_interval = stats.ttest_ind(churned['last boot - interval'], non_churned['last boot - interval'], equal_var=False)
print(f"T-test for 'last boot - interval' p-value: {p_val_interval}")

# T-test for 'last boot - active' between churned and non-churned
t_stat_active, p_val_active = stats.ttest_ind(churned['last boot - active'], non_churned['last boot - active'], equal_var=False)
print(f"T-test for 'last boot - active' p-value: {p_val_active}")

# T-test for 'return - activate' between churned and non-churned
t_stat_return, p_val_return = stats.ttest_ind(churned['return - activate'], non_churned['return - activate'], equal_var=False)
print(f"T-test for 'return - activate' p-value: {p_val_return}")

# Select only numeric columns for correlation analysis
numeric_b30_pro = b30_pro.select_dtypes(include=['float64', 'int64'])

# Correlation analysis with Churn Flag (excluding non-numeric columns)
corr_with_churn = numeric_b30_pro.corr()['Churn Flag'].sort_values(ascending=False)
print("Correlation with Churn Flag:\n", corr_with_churn)

# Visualize the correlation matrix focusing on numeric columns
corr_matrix = numeric_b30_pro.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

