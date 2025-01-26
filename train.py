import logging
import yaml
import mlflow
import mlflow.sklearn
from src.ingest import Ingestion
from src.clean import Cleaner
from src.train import Trainer
from src.predict import Predictor
from sklearn.metrics import classification_report

# Configure logging to display progress and key actions during execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    """
    Main function to execute the machine learning pipeline without MLflow tracking.
    Includes:
    - Data ingestion
    - Data cleaning
    - Model training and saving
    """
    # Step 1: Load data using the Ingestion class
    ingestion = Ingestion()  # Create an instance of the Ingestion class
    train, test = ingestion.load_data()  # Load training and testing datasets
    logging.info("Data ingestion completed successfully")  # Log successful data ingestion

    # Step 2: Clean data using the Cleaner class
    cleaner = Cleaner()  # Create an instance of the Cleaner class
    train_data = cleaner.clean_data(train)  # Clean the training data
    test_data = cleaner.clean_data(test)  # Clean the testing data
    logging.info("Data cleaning completed successfully")  # Log successful data cleaning

    # Step 3: Prepare and train the model using the Trainer class
    trainer = Trainer()  # Create an instance of the Trainer class
    X_train, y_train = trainer.feature_target_separator(train_data)  # Separate features and target from training data
    trainer.train_model(X_train, y_train)  # Train the model using the training data
    trainer.save_model()  # Save the trained model to a file
    logging.info("Model training completed successfully")  # Log successful model training and saving

def train_with_mlflow():
    """
    Function to execute the machine learning pipeline with MLflow tracking.
    Tracks the pipeline execution, logs metrics, and saves artifacts.
    Includes:
    - Data ingestion
    - Data cleaning
    - Model training and saving
    """
    # Step 1: Load configuration from a YAML file
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)  # Load the YAML configuration file for experiment settings

    # Step 2: Set the MLflow experiment name
    mlflow.set_experiment("Model Training Experiment")

    # Step 3: Start an MLflow run
    with mlflow.start_run() as run:
        logging.info("MLflow run started")  # Log the start of the MLflow run

        # Step 4: Load data using the Ingestion class
        ingestion = Ingestion()  # Create an instance of the Ingestion class
        train, test = ingestion.load_data()  # Load training and testing datasets
        logging.info("Data ingestion completed successfully")  # Log successful data ingestion

        # Step 5: Clean data using the Cleaner class
        cleaner = Cleaner()  # Create an instance of the Cleaner class
        train_data = cleaner.clean_data(train)  # Clean the training data
        test_data = cleaner.clean_data(test)  # Clean the testing data
        logging.info("Data cleaning completed successfully")  # Log successful data cleaning

        # Step 6: Prepare and train the model using the Trainer class
        trainer = Trainer()  # Create an instance of the Trainer class
        X_train, y_train = trainer.feature_target_separator(train_data)  # Separate features and target from training data
        trainer.train_model(X_train, y_train)  # Train the model using the training data
        trainer.save_model()  # Save the trained model to a file
        logging.info("Model training completed successfully")  # Log successful model training and saving

if __name__ == "__main__":
    """
    Entry point for the script. Provides options to execute:
    - The pipeline without MLflow tracking (main function).
    - The pipeline with MLflow tracking (train_with_mlflow function).
    """
    # main()
    train_with_mlflow()