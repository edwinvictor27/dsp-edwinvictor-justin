import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression


def build_model(
        combined_features_df: pd.DataFrame,
        y_train: pd.Series,
        model_path: str,
        scaler_path: str,
        encoder_path: str,
        scaler,
        encoder):
    """
    Build, train, and save a linear regression model along with its scaler
    and encoder.

    This function trains a linear regression model using the provided features
    and target variable. After training, it saves the model, the scaler, and
    the encoder to the specified destination folder.

    Args:
        combined_features_df (pd.DataFrame): DataFrame to be used .
        y_train (pd.Series): Series contains the target variable for training.
        model_path (str): destination path where the trained model
                          should be saved.
        scaler_path (str): destination where the scaler object
                           should be saved.
        encoder_path (str): Path where the encoder object
                            should be saved.
        scaler (Any): Scaler object used to scale the continuous features.
        encoder (Any): Encoder object used to encode the categorical features.

    Returns:
        LinearRegression: The trained linear regression model.
    """
    model = LinearRegression()
    model.fit(combined_features_df, y_train)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoder, encoder_path)
    return model


def process_testing_set(
        X_test: pd.DataFrame,
        categorical_features: list,
        continuous_features: list,
        encoder,
        scaler,
        model) -> pd.DataFrame:
    """
    Process the testing set and generate predictions.

    Encodes categorical features and scales continuous features
    in the testing set using the provided encoder and scaler.
    Then, combines the processed features and uses
    the model to predict outcomes.

    Args:
        X_test (pd.DataFrame): The DataFrame containing the test features.
        categorical_features (List): List of categorical column names.
        continuous_features (List): List of continuous column names.
        encoder (str): Path to the saved encoder.
        scaler (str): Path to the saved scaler.
        model: The trained model used for making predictions.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: The processed test DataFrame and
                                         the predictions from the model.
    """
    encoder = joblib.load(encoder)
    scaler = joblib.load(scaler)
    categorical_df = X_test[categorical_features]
    continuous_df = X_test[continuous_features]
    encoded_categories = encoder.transform(categorical_df)
    encoded_df = pd.DataFrame(
        encoded_categories,
        columns=encoder.get_feature_names_out(categorical_features)
    )
    scaled_features = scaler.transform(continuous_df)
    scaled_df = pd.DataFrame(
        scaled_features,
        columns=continuous_features
    )
    processed_test_df = pd.concat([scaled_df, encoded_df], axis=1)
    y_pred = model.predict(processed_test_df)
    return processed_test_df, y_pred


def evaluate_model(
        y_test: pd.Series,
        y_pred: np.ndarray) -> dict[str, str]:
    """
    Evaluate the model's performance using
    Root Mean Squared Logarithmic Error (RMSLE) metric.
    It returns a dictionary containing
    the RMSLE value as a string.

    Args:
        y_test (pd.Series): The target values for the test set.
        y_pred (np.ndarray): The predicted target values
                             by the model.

    Returns:
        Dict[str, str]: A dictionary with a descriptive key and
                        the RMSLE value as a string.
    """
    Rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return {'Root Mean Squared Error out of': str(Rmsle)}
