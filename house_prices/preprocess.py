import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    This function is used to Load a dataset from the local device.

    Args:
        file_path (str): The path of the dataset file.

    Returns:
        pd.DataFrame: The dataset is loaded as Pandas DataFrame.
    """
    dataset = pd.read_csv(file_path)
    print(f"Dataset loaded from {file_path} with shape {dataset.shape}")
    return dataset


def select_features(
    data: pd.DataFrame,
    selected_features: List[str],
    target_feature: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select features and the target variable for ML training.

    Args:
        data (pd.DataFrame): The dataset to prepare features from.
        selected_features (List[str]): List of feature names to include.
        target_feature (str): The name of the target feature.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: selected features and target variable.
    """
    X = data[selected_features]
    y = data[target_feature]
    return X, y


def split_dataset(
    data: pd.DataFrame,
    target_feature: str,
    test_size: float = 0.35,
    random_state: int = 50
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets.

    Args:
        data (pd.DataFrame): The dataset to split.
        target_feature (str): The column name of the target feature.
        test_size (float): The proportion of the dataset to be splitted
        random_state (int): Seed used by the random number generator.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        The training features, testing features, training target,
        and testing target.
    """
    X = data.drop(target_feature, axis=1)
    y = data[target_feature]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def encode_categorical_features(
        df: pd.DataFrame,
        categorical_features: list
) -> pd.DataFrame:
    """
    Encode categorical features in a DataFrame using one-hot encoding.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to encode.
        categorical_features (List[str]): List of all categorical
                                         column names.

    Returns:
        Tuple[pd.DataFrame, OneHotEncoder]: A tuple containing:
            - A DataFrame with one-hot encoded categorical features.
            - The fitted OneHotEncoder instance used for encoding.
    """
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(df[categorical_features])
    encoded_categories = encoder.transform(df[categorical_features])
    encoded_df = pd.DataFrame(
        encoded_categories,
        columns=encoder.get_feature_names_out(categorical_features)
    )
    return encoded_df, encoder


def scale_continuous_features(
        df: pd.DataFrame,
        continuous_features: list
) -> pd.DataFrame:
    """
    Scale continuous features in a DataFrame using standard scaling.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to scale.
        continuous_features (List[str]): List of all continuous
                                         column names

    Returns:
        Tuple[pd.DataFrame, StandardScaler]: A tuple containing:
            - A DataFrame with scaled continuous features.
            - The fitted StandardScaler instance used for scaling.
    """
    scaler = StandardScaler()
    scaler.fit(df[continuous_features])
    scaled_features = scaler.transform(df[continuous_features])
    scaled_df = pd.DataFrame(
        scaled_features,
        columns=continuous_features
    )
    return scaled_df, scaler


def concatenate_features(
        continuous_features_df: pd.DataFrame,
        categorical_features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Concatenate continuous and categorical feature DataFrames.

    Combines two DataFrames, one containing continuous features and the other
    containing categorical features, along the column axis
    to produce a single DataFrame with all features.

    Args:
        continuous_features_df (pd.DataFrame):continuous features DF.
        categorical_features_df (pd.DataFrame):categorical features DF.

    Returns:
        pd.DataFrame: A DataFrame with Transformed
                      continuous and categorical features.
    """
    concatenated = pd.concat(
        [continuous_features_df, categorical_features_df],
        axis=1)
    return concatenated
