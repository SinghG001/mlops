import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import mlflow
import mlflow.sklearn
import joblib


def load_data():
    return load_iris()


def create_model(random_state):
    return RandomForestClassifier(random_state = random_state)

def test_train_split(data):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
    return X_train, X_test,y_train, y_test



data = load_data()



X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['target'])

X_train, X_test,y_train, y_test =  test_train_split(data)


# Define parameter grid for tuning

mlflow.set_experiment("Random Forest Classifier")

with mlflow.start_run():
    param_grid = {
    'n_estimators': [100, 200, 300],   # Number of trees
    'max_depth': [None, 10, 20, 30],    # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],    # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],      # Minimum samples required at a leaf node
    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider for the best split
}
    model = create_model(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    mlflow.log_param("n_estimators", param_grid['n_estimators'])
    mlflow.log_param("max_depth",param_grid['max_depth'])
    mlflow.log_param("min_samples_split",param_grid['min_samples_split'])
    mlflow.log_param("min_samples_leaf",param_grid['min_samples_leaf'])
    mlflow.log_param("max_features",param_grid['max_features'])

    grid_search.fit(X_train, y_train.values.ravel())
    
    # Get the best model
    best_rf = grid_search.best_estimator_   
    y_pred = best_rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("accuracy",accuracy)

    mlflow.sklearn.log_model(model, "rf-default")
    joblib.dump(model, 'model/rf-default.joblib')
    mlflow.log_artifact('model/rf-default.joblib')
