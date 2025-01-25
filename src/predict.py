import os
import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import yaml

class Predictor:
    """
    A class for loading a trained model, preparing data, and evaluating the model's performance.
    """

    def __init__(self):
        # Load configuration settings
        self.model_path = self.load_config()['model']['store_path']
        # Load the trained model pipeline
        self.pipeline = self.load_model()

    def load_config(self):
        """
        Loads configuration settings from a YAML file.

        The configuration file should specify the model's storage path under the 'model' section.

        """
        with open('config.yml', 'r') as config_file:
            # Safely loads the YAML file and returns its contents as a dictionary
            return yaml.safe_load(config_file)
        
    def load_model(self):
        """
        Loads the trained machine learning model pipeline from a file.
        The model file path is constructed using the configuration settings.
        """
        model_file_path = os.path.join(self.model_path, 'model.pkl')
        # Load and return the model pipeline
        return joblib.load(model_file_path)

    def feature_target_separator(self, data):
        """
        Separates the features (X) and the target variable (y) from the dataset.
        Assumes the target variable is the last column in the dataset.

        Parameters:
        - data (pandas.DataFrame): The input dataset.

        Returns:
        - tuple: A tuple containing:
          - X (pandas.DataFrame): The feature set.
          - y (pandas.Series): The target variable.
        """
        # Extract all columns except the last as features (X)
        X = data.iloc[:, :-1]
        # Extract the last column as the target variable (y)
        y = data.iloc[:, -1]
        return X, y

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the model's performance on the test dataset.
        Calculates accuracy, a classification report, and the ROC-AUC score.

        Parameters:
        - X_test (pandas.DataFrame): The test feature set.
        - y_test (pandas.Series): The true labels for the test set.

        Returns:
        - tuple: A tuple containing:
          - accuracy (float): The accuracy score.
          - class_report (str): The classification report.
          - roc_auc (float): The ROC-AUC score.
        """
        # Predict labels for the test feature set
        y_pred = self.pipeline.predict(X_test)
        # Compute accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # Generate a detailed classification report
        class_report = classification_report(y_test, y_pred)
        # Compute the ROC-AUC score
        roc_auc = roc_auc_score(y_test, y_pred)
        return accuracy, class_report, roc_auc
