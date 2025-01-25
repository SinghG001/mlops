import os
import joblib
import yaml
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

class Trainer:
    """
    A class for training machine learning models using a configurable pipeline 
    that includes preprocessing, oversampling, and model training.
    """

    def __init__(self):
        self.config = self.load_config()  # Load configuration from YAML file
        self.model_name = self.config['model']['name']  # Name of the model to use
        self.model_params = self.config['model']['params']  # Model parameters
        self.model_path = self.config['model']['store_path']  # Path to save the trained model
        self.pipeline = self.create_pipeline()  # Create the training pipeline

    def load_config(self):
        """
        Loads configuration settings from a YAML file.
        """
        with open('config.yml', 'r') as config_file:
            # Load and return YAML configuration as a dictionary
            return yaml.safe_load(config_file)
        
    def create_pipeline(self):
        """
        Creates a training pipeline that includes:
        1. Preprocessing: Scaling numeric features and one-hot encoding categorical features.
        2. Oversampling: SMOTE to handle class imbalance.
        3. Model: A machine learning model specified in the configuration.
        """
        # Define the preprocessing steps
        preprocessor = ColumnTransformer(transformers=[
            ('minmax', MinMaxScaler(), ['AnnualPremium']),  # Scale 'AnnualPremium' with MinMaxScaler
            ('standardize', StandardScaler(), ['Age', 'RegionID']),  # Standardize 'Age' and 'RegionID'
            ('onehot', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'PastAccident'])  # One-hot encode categorical columns
        ])
        
        # Handle class imbalance using SMOTE
        smote = SMOTE(sampling_strategy=1.0)
        
        # Map model names to corresponding scikit-learn classes
        model_map = {
            'RandomForestClassifier': RandomForestClassifier,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier
        }
        
        # Retrieve the model class and instantiate it with parameters
        model_class = model_map[self.model_name]
        model = model_class(**self.model_params)
        
        # Construct the pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('smote', smote),
            ('model', model)
        ])
        
        return pipeline

    def feature_target_separator(self, data):
        """
        Separates features (X) and target variable (y) from the dataset.
        Assumes the target variable is the last column.
        """
        X = data.iloc[:, :-1]  # All columns except the last as features
        y = data.iloc[:, -1]  # The last column as the target
        return X, y

    def train_model(self, X_train, y_train):
        """
        Trains the model using the provided training data.
        """
        self.pipeline.fit(X_train, y_train)  # Fit the pipeline to the training data

    def save_model(self):
        """
        Saves the trained pipeline to a file.
        The file path is specified in the configuration.
        Saves the pipeline object as a .pkl file at the specified path.
        """
        model_file_path = os.path.join(self.model_path, 'model.pkl')  # Construct the file path
        joblib.dump(self.pipeline, model_file_path)  # Save the pipeline to the file
