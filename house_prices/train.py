import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

def build_model(combined_features_df: pd.DataFrame, y_train: pd.Series, model_path: str, scaler_path: str, encoder_path: str, scaler, encoder):
    model = LinearRegression()
    model.fit(combined_features_df, y_train)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoder, encoder_path)
    return model

def process_testing_set(X_test: pd.DataFrame, categorical_features: list, continuous_features: list, encoder, scaler, model) -> pd.DataFrame:
    # Encode categorical features
    
    encoder = joblib.load(encoder)
    scaler = joblib.load(scaler)
    categorical_df = X_test[categorical_features]
    continuous_df = X_test[continuous_features]

    encoded_categories = encoder.transform(categorical_df)
    encoded_df = pd.DataFrame(encoded_categories, columns=encoder.get_feature_names_out(categorical_features))
    
    # Scale continuous features
    scaled_features = scaler.transform(continuous_df)
    scaled_df = pd.DataFrame(scaled_features, columns=continuous_features)
    
    # Concatenate features
    processed_test_df = pd.concat([scaled_df, encoded_df], axis =1)
    y_pred = model.predict(processed_test_df)
    return processed_test_df,y_pred

def evaluate_model(y_test: pd.Series, y_pred: np.ndarray) -> dict[str, str]:
    Rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return {'Root Mean Squared Error out of': str(Rmsle)}

