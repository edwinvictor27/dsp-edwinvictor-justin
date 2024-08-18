import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from IPython.display import display

def load_model_and_transformers(model_path: str, scaler_path: str, encoder_path: str):
    """
    Loads the pre-trained model, scaler, and encoder.

    Args:
        model_path (str): Path to the saved model.
        scaler_path (str): Path to the saved scaler.
        encoder_path (str): Path to the saved encoder.

    Returns:
        model: Loaded model object.
        scaler: Loaded scaler object.
        encoder: Loaded encoder object.
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    return model, scaler, encoder

def preprocess_data_and_predict(Testing_df: pd.DataFrame, scaler, encoder, model, continuos_features: list, categorical_features: list) -> pd.DataFrame:
    """
    Preprocesses the input data by scaling continuous features and encoding categorical features.

    Args:
        input_data (pd.DataFrame): The data to preprocess.
        scaler: Scaler object to scale continuous features.
        encoder: Encoder object to encode categorical features.
        continuous_features (list): List of continuous feature names.
        discrete_features (list): List of discrete feature names.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    test_scaled = scaler.transform(Testing_df[continuos_features])
    test_encoded = encoder.transform(Testing_df[categorical_features])

    test_scaled_df = pd.DataFrame(test_scaled, columns=continuos_features)
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_features))

    transformed_test_df = pd.concat([test_scaled_df, test_encoded_df], axis=1)
    predict_house_price = model.predict(transformed_test_df)
    return predict_house_price

