
import logging
import yaml
import mlflow
import mlflow.sklearn
from src.ingest import Ingestion
from src.clean import Cleaner
from src.train import Trainer
from src.predict import Predictor
from sklearn.metrics import classification_report

# Set up logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    # Load data
    ingestion = Ingestion()
    train, test = ingestion.load_data()
    logging.info("Data ingestion completed successfully")

   
def train_with_mlflow():

    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    mlflow.set_experiment("Model Training Experiment")
    
    with mlflow.start_run() as run:
        # Load data
        ingestion = Ingestion()
        train, test = ingestion.load_data()
        logging.info("Data ingestion completed successfully")

  
        
if __name__ == "__main__":
    # main()
    train_with_mlflow()


