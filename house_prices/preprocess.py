import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    This function is used to Load a dataset from the specified file path in the local device.

    Args:
        file_path (str): The path of the dataset file.

    Returns:
        pd.DataFrame: The dataset is loaded as Pandas DataFrame.
    """
    dataset = pd.read_csv(file_path)
    print(f"Dataset loaded from {file_path} with shape {dataset.shape}")
    return dataset

def select_features(data: pd.DataFrame, selected_features: list, target_feature: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    selecting the features and target variable for ml training.

    Args:
        data (pd.DataFrame): The dataset to prepare features from.
        selected_features (list): List of feature names to include.
        target_feature (str): The name of the target feature.

    Returns:
        tuple[pd.DataFrame, pd.Series]: The selected features and target variable.
    """
    X = data[selected_features]
    y = data[target_feature]
    return X, y

def split_dataset(data: pd.DataFrame, target_feature: str, test_size: float = 0.35, random_state: int = 50) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    This function is used to Split the dataset into training and testing sets.

    Args:
        data (pd.DataFrame): The dataset to split.
        target_feature (str): The column_name of the target feature.
        test_size (float): The proportion of the dataset to include in the test split. Defaults to 0.35.
        random_state (int): Seed used by the random number generator. Defaults to 50.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: The training features, testing features, training target, and testing target.
    """
    X = data.drop(target_feature, axis=1)
    y = data[target_feature]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def encode_categorical_features(df: pd.DataFrame, categorical_features: list) -> pd.DataFrame:
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(df[categorical_features])
    encoded_categories = encoder.transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_categories, columns=encoder.get_feature_names_out(categorical_features))
    return encoded_df, encoder

def scale_continuous_features(df: pd.DataFrame, continuous_features: list) -> pd.DataFrame:
    scaler = StandardScaler()
    scaler.fit(df[continuous_features])
    scaled_features = scaler.transform(df[continuous_features])
    scaled_df = pd.DataFrame(scaled_features, columns=continuous_features)
    return scaled_df, scaler

def concatenate_features(continuous_features_df: pd.DataFrame, categorical_features_df: pd.DataFrame) -> pd.DataFrame:
    concatenated = pd.concat([continuous_features_df, categorical_features_df], axis=1)
    return concatenated
