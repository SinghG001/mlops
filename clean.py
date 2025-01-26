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
    Main function to execute data ingestion and cleaning without MLflow tracking.
    Includes:
    - Data ingestion
    - Data cleaning
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

def cleandata_with_mlflow():
    """
    Function to execute data ingestion and cleaning with MLflow tracking.
    Includes:
    - Data ingestion
    - Data cleaning
    - Logging the process with MLflow
    """
    # Step 1: Load configuration from a YAML file
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)  # Load YAML configuration file

    # Step 2: Set the MLflow experiment name
    mlflow.set_experiment("Model Training Experiment")

    # Step 3: Start an MLflow run for tracking
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

if __name__ == "__main__":
    """
    Entry point for the script. Choose between:
    - `main()`: Executes the pipeline without MLflow tracking.
    - `cleandata_with_mlflow()`: Executes the pipeline with MLflow tracking.
    """
    # main()  
    cleandata_with_mlflow()
