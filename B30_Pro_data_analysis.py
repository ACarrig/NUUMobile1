import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the Excel sheet into a DataFrame.
def load_data(file_path, sheet_name):
    xls = pd.ExcelFile(file_path)
    # print(f"Sheet Names: {xls.sheet_names}")
    return pd.read_excel(xls, sheet_name=sheet_name)

# Check for missing values and duplicates in the DataFrame
def check_data_integrity(data):
    # check for missing values
    print(f"Missing Values:\n{data.isnull().sum()}") # Results: no missing values

    # check for duplicates
    duplicates = data.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}") # Results: no duplicates

# visualize the distribution of the 'Churn Flag', check for class imbalance
def explore_churn_flag_distribution(data):
    if 'Churn Flag' in data.columns:
        churn_counts = data['Churn Flag'].value_counts()
        print(f"Churn Flag Distribution:\n{churn_counts}")
        churn_counts.plot(kind='bar', title='Churn Flag Distribution')
        plt.show()
    
    """
    Results:
    Churn Flag Distribution:
    Churn Flag
    0    3072
    1      70
    Name: count, dtype: int64
    """

# Plot the distribution of a specific feature for churned vs non-churned customers
def plot_feature_distribution(data, feature, churned_data, non_churned_data):
    # Plot the distributions
    plt.figure(figsize=(12, 6))
    sns.histplot(churned_data[feature], color='red', label='Churned', kde=True, stat='density', bins=30)
    sns.histplot(non_churned_data[feature], color='blue', label='Non-Churned', kde=True, stat='density', bins=30)
    plt.title(f'Distribution of {feature}: Churned vs Non-Churned')
    plt.legend()
    plt.show()

# Perform T-tests on the features between churned and non-churned customers
def perform_t_tests(churned_data, non_churned_data):
    features = ['last boot - interval', 'last boot - active', 'return - activate']
    
    for feature in features:
        t_stat, p_val = stats.ttest_ind(churned_data[feature], non_churned_data[feature], equal_var=False)
        print(f"T-test for '{feature}' p-value: {p_val}")

# Analyze correlation between numeric features and 'Churn Flag'
def analyze_correlation(data):
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    corr_with_churn = numeric_data.corr()['Churn Flag'].sort_values(ascending=False)
    print("Correlation with Churn Flag:\n", corr_with_churn)

    # Visualize the correlation matrix
    corr_matrix = numeric_data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    """
    Results of Correlation with Churn Flag:
    Correlation with Churn Flag:
    Churn Flag              1.000000
    return - activate       0.404630
    last boot - interval    0.171179
    last boot - active     -0.125439
    Name: Churn Flag, dtype: float64
    """

def main():
    # Load the B30 Pro sheet
    file_path = "UW_Churn_Pred_Data.xls"
    b30_pro = load_data(file_path, 'B30 Pro')

    # Data integrity check
    print(f"B30 Pro Shape: {b30_pro.shape}")
    print(b30_pro.info())

    """
    Numeric features:
    6   last boot - interval       3142 non-null   float64
    7   last boot - active         3142 non-null   float64
    8   return - activate          3142 non-null   float64
    """

    check_data_integrity(b30_pro)

    # Explore churn flag distribution
    explore_churn_flag_distribution(b30_pro)

    # Separate churned and non-churned records
    churned = b30_pro[b30_pro['Churn Flag'] == 1]
    non_churned = b30_pro[b30_pro['Churn Flag'] == 0]

    # Plot distributions for key features
    plot_feature_distribution(b30_pro, 'last boot - interval', churned, non_churned)
    plot_feature_distribution(b30_pro, 'last boot - active', churned, non_churned)
    plot_feature_distribution(b30_pro, 'return - activate', churned, non_churned)

    # Perform T-tests on key features
    perform_t_tests(churned, non_churned)

    """
    Results of T-tests:
    T-test for 'last boot - interval' p-value: 3.5534948409265625e-19
    T-test for 'last boot - active' p-value: 7.625494852891254e-43
    T-test for 'return - activate' p-value: 3.692385468617318e-07

    All the T-tests for these features are extremely small => significant difference between churned & non-churned customers
    """

    # Analyze correlations
    analyze_correlation(b30_pro)

# Run the main function to execute the analysis
if __name__ == "__main__":
    main()
