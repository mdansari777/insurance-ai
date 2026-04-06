from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def get_health_preprocessor():
    numeric = ['age','bmi','children']
    categorical = ['gender','smoker','region']
    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())])
    cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([('num', num_transformer, numeric),
                                      ('cat', cat_transformer, categorical)])
    return preprocessor

def get_car_preprocessor():
    numeric = ['age','driving_experience','vehicle_age','previous_claims','annual_mileage']
    categorical = ['gender','vehicle_type','location']
    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())])
    cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([('num', num_transformer, numeric),
                                      ('cat', cat_transformer, categorical)])
    return preprocessor

def get_life_preprocessor():
    numeric = ['age','annual_income','coverage_amount','term_length']
    categorical = ['gender','smoker','health_status']
    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())])
    cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([('num', num_transformer, numeric),
                                      ('cat', cat_transformer, categorical)])
    return preprocessor

def get_home_preprocessor():
    numeric = ['home_age','sqft','previous_claims']
    categorical = ['location','construction_type','roof_type','security_system']
    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())])
    cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([('num', num_transformer, numeric),
                                      ('cat', cat_transformer, categorical)])
    return preprocessor

def get_fraud_preprocessor():
    numeric = ['claim_amount','witnesses']
    categorical = ['policy_type','incident_type','incident_severity','police_report']
    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())])
    cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([('num', num_transformer, numeric),
                                      ('cat', cat_transformer, categorical)])
    return preprocessor