import pandas as pd

def load_data(path):
    """Load dataset from CSV"""
    return pd.read_csv(path)

def check_missing_values(df):
    """Print missing values per column"""
    print("Missing values per column:")
    print(df.isnull().sum())

def target_distribution(df, target_col):
    """Print target variable distribution"""
    print(df[target_col].value_counts())
