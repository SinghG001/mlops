import logging
import yaml
import mlflow
import mlflow.sklearn
from src.ingest import Ingestion
from src.clean import Cleaner
from src.train import Trainer
from src.predict import Predictor
from sklearn.metrics import classification_report

# Set up logging configuration to capture progress and key steps in the workflow
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    """
    Main function to execute data ingestion without MLflow tracking.
    Includes:
    - Data ingestion
    """
    # Step 1: Load data using the Ingestion class
    ingestion = Ingestion()  # Create an instance of the Ingestion class
    train, test = ingestion.load_data()  # Load training and testing datasets
    logging.info("Data ingestion completed successfully")  # Log successful ingestion

def loaddata_with_mlflow():
    """
    Function to execute data ingestion with MLflow tracking.
    Tracks the ingestion process using MLflow's experiment framework.
    """
    # Step 1: Load configuration from a YAML file
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)  # Load YAML configuration settings

    # Step 2: Set the MLflow experiment name
    mlflow.set_experiment("Model Training Experiment")

    # Step 3: Start an MLflow run for tracking the workflow
    with mlflow.start_run() as run:
        # Log the start of the MLflow run
        logging.info("MLflow run started")

        # Step 4: Load data using the Ingestion class
        ingestion = Ingestion()  # Create an instance of the Ingestion class
        train, test = ingestion.load_data()  # Load training and testing datasets
        logging.info("Data ingestion completed successfully")  # Log successful ingestion

if __name__ == "__main__":
    """
    - Uncomment `main()` to execute data ingestion without MLflow tracking.
    - Uncomment `loaddata_with_mlflow()` to execute data ingestion with MLflow tracking.
    """
    # main()
    loaddata_with_mlflow()