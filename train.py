
# from mlflow_train import test_train_split, load_data, create_model


# data = load_data()

# def test_train_test_split():
#     X_train, X_test, y_train, y_test = test_train_split(data)
#     assert X_train.shape[1] == X_test.shape[1], "Train and test feature dimensions mismatch"

# def test_model_accuracy():
#     X_train, X_test, y_train, y_test = test_train_split(data)
#     model = create_model(random_state)
#     model.fit(X_train, y_train.values.ravel())
#     accuracy = model.score(X_test, y_test)
#     assert accuracy > 0.8, "Model accuracy is below acceptable threshold"

# random_state = 42
# test_train_test_split()
# test_model_accuracy()

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

    # Clean data
    cleaner = Cleaner()
    train_data = cleaner.clean_data(train)
    test_data = cleaner.clean_data(test)
    logging.info("Data cleaning completed successfully")

    # Prepare and train model
    trainer = Trainer()
    X_train, y_train = trainer.feature_target_separator(train_data)
    trainer.train_model(X_train, y_train)
    trainer.save_model()
    logging.info("Model training completed successfully")

  

def train_with_mlflow():

    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    mlflow.set_experiment("Model Training Experiment")
    
    with mlflow.start_run() as run:
        # Load data
        ingestion = Ingestion()
        train, test = ingestion.load_data()
        logging.info("Data ingestion completed successfully")

        # Clean data
        cleaner = Cleaner()
        train_data = cleaner.clean_data(train)
        test_data = cleaner.clean_data(test)
        logging.info("Data cleaning completed successfully")

        # Prepare and train model
        trainer = Trainer()
        X_train, y_train = trainer.feature_target_separator(train_data)
        trainer.train_model(X_train, y_train)
        trainer.save_model()
        logging.info("Model training completed successfully")
        
if __name__ == "__main__":
    # main()
    train_with_mlflow()


