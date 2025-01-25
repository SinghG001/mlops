import logging
import yaml
import mlflow
import mlflow.sklearn
from src.ingest import Ingestion
from src.clean import Cleaner
from src.train import Trainer
from src.predict import Predictor
from sklearn.metrics import classification_report

# Configure logging to capture the progress and details of the workflow
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    """
    Main function for executing the machine learning pipeline without MLflow tracking.
    Includes:
    - Data ingestion
    - Data cleaning
    - Model training and saving
    """
    # Step 1: Load data
    ingestion = Ingestion()  # Create an instance of the Ingestion class
    train, test = ingestion.load_data()  # Load training and testing datasets
    logging.info("Data ingestion completed successfully")

    # Step 2: Clean data
    cleaner = Cleaner()  # Create an instance of the Cleaner class
    train_data = cleaner.clean_data(train)  # Clean the training data
    test_data = cleaner.clean_data(test)    # Clean the testing data
    logging.info("Data cleaning completed successfully")

    # Step 3: Prepare and train model
    trainer = Trainer()  # Create an instance of the Trainer class
    X_train, y_train = trainer.feature_target_separator(train_data)  # Separate features and target
    trainer.train_model(X_train, y_train)  # Train the model using the training data
    trainer.save_model()  # Save the trained model
    logging.info("Model training completed successfully")

def train_with_mlflow():
    """
    Function for executing the machine learning pipeline with MLflow tracking.
    Includes:
    - Data ingestion
    - Data cleaning
    - Model training and saving
    - Logging metrics and artifacts with MLflow
    """
    # Step 1: Load configuration from a YAML file
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)  # Load the YAML configuration file

    # Step 2: Set the MLflow experiment name
    mlflow.set_experiment("Model Training Experiment")

    with mlflow.start_run() as run:
        logging.info("MLflow run started")

        # Step 3: Load data
        ingestion = Ingestion()  # Create an instance of the Ingestion class
        train, test = ingestion.load_data()  # Load training and testing datasets
        logging.info("Data ingestion completed successfully")

        # Step 4: Clean data
        cleaner = Cleaner()  # Create an instance of the Cleaner class
        train_data = cleaner.clean_data(train)  # Clean the training data
        test_data = cleaner.clean_data(test)    # Clean the testing data
        logging.info("Data cleaning completed successfully")

        # Step 5: Prepare and train model
        trainer = Trainer()  # Create an instance of the Trainer class
        X_train, y_train = trainer.feature_target_separator(train_data)  # Separate features and target
        trainer.train_model(X_train, y_train)  # Train the model using the training data
        trainer.save_model()  # Save the trained model
        logging.info("Model training completed successfully")

        # (Optional) Further steps can include evaluation and logging metrics to MLflow

if __name__ == "__main__":
    """
    Entry point for the script. Executes either:
    - The main pipeline without MLflow tracking, or
    - The pipeline with MLflow tracking.
    """
    # Uncomment main() if MLflow tracking is not needed
    # main()

    # Execute the pipeline with MLflow tracking
    train_with_mlflow()