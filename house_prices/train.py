def build_model(data: pd.DataFrame) -> dict[str, str]:

## Training_set
    
    # 1) splitting the dataset
    
    X = Training_dataset.drop(target_feature, axis=1)
    y = Training_dataset[target_feature]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=50)


    # 2) Extracting the features from training set
    
    Extracted_Selected_Features_For_Training = X_train[selected_features]
    Extracted_Target_Attribute = y_train[target_feature]
    Training_Features = pd.concat([Extracted_Selected_Features_For_Training,y_train], axis=1)

    # 3) Encoding the categorical columns from training set

    encoder = OneHotEncoder(sparse_output= False)
    encoder.fit(Training_Features[discrete_datatype_features])
    Training_encoded_categories = encoder.transform(Training_Features[discrete_datatype_features])
    encoded_discrete_features_training_df = pd.DataFrame(Training_encoded_categories, columns=encoder.get_feature_names_out(discrete_datatype_features))

    # 4) Scaling the continuos columns from training set

    scaler = StandardScaler()
    scaler.fit(X_train[continuos_datatype_features])
    scaled_continuos_features_training_df = scaler.transform(X_train[continuos_datatype_features])

    # 5) Concatenating the processed training set
    
    training_continuous_features_df = pd.DataFrame(scaled_continuos_features_training_df , columns= continuos_datatype_features)
    Processed_Training_Df = pd.concat([training_continuous_features_df, encoded_discrete_features_training_df] , axis=1)

    # 6) Fitting the model
    
    model = LinearRegression()
    model.fit(Processed_Training_Df, y_train)

## Testing_set   

    # 1) Extracting the features from testing set
    
    Extracted_Features_Testing = X_test[selected_features]
    Testing_Features = pd.concat([Extracted_Features_Testing,y_test], axis=1)

    # 2) Encoding the categorical columns from testing set
    
    encoder.fit(Testing_Features[discrete_datatype_features])
    Testing_encoded_categories = encoder.transform(Testing_Features[discrete_datatype_features])
    encoded_discrete_features_testing_df = pd.DataFrame(Testing_encoded_categories, columns=encoder.get_feature_names_out(discrete_datatype_features))

    # 3) Scaling the continuos columns from testing set
    
    scaler.fit(X_test[continuos_datatype_features])
    scaled_continuos_features_testing_df = scaler.transform(X_test[continuos_datatype_features])

    # 4) Concatenating the processed testing set
    
    testing_continuous_features_df = pd.DataFrame(scaled_continuos_features_testing_df , columns= continuos_datatype_features)
    Processed_Testing_Df = pd.concat([testing_continuous_features_df, encoded_discrete_features_testing_df] , axis=1)
    
    # 5) Making prediction 
    
    y_pred = model.predict(Processed_Testing_Df)

    # 6) Evaluating the model
    
    Rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred)) 
    return {'Root Mean Squared Error out of': str(Rmsle) }
