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

def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    
    dataset_for_testing = r"C:/Users/edwin victor/git repositories/dsp-edwinvictor-justin/data/test.csv"
    scaler_location = r'C:/Users/edwin victor/git repositories/dsp-edwinvictor-justin/models/scaler.joblib'
    encoder_location = r'C:/Users/edwin victor/git repositories/dsp-edwinvictor-justin/models/encoder.joblib'
    model_location = r'C:/Users/edwin victor/git repositories/dsp-edwinvictor-justin/models/model.joblib'

    model = joblib.load(model_location)
    Testing_df = pd.read_csv(dataset_for_testing)
    scaler = joblib.load(scaler_location)
    encoder = joblib.load(encoder_location)

    selected_features = ['LotArea', 'GrLivArea', 'Neighborhood', 'HouseStyle'] # 2 continuos and 2 Categorical features
    target_feature = ['SalePrice']
    continuos_datatype_features = ['LotArea', 'GrLivArea']
    discrete_datatype_features = ['Neighborhood', 'HouseStyle']
    
    test_scaled = scaler.transform(Testing_df[continuos_datatype_features])
    test_encoded = encoder.transform(Testing_df[discrete_datatype_features])
    
    test_scaled_df = pd.DataFrame(test_scaled,columns=continuos_datatype_features)
    test_encoded_df = pd.DataFrame(test_encoded,columns=encoder.get_feature_names_out(discrete_datatype_features))
    
    Transformed_test_df = pd.concat([test_scaled_df,test_encoded_df], axis=1)
    

    predict_house_price = model.predict(Transformed_test_df)
    return predict_house_price