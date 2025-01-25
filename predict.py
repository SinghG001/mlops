import logging
import yaml
import mlflow
import mlflow.sklearn
from src.ingest import Ingestion
from src.clean import Cleaner
from src.train import Trainer
from src.predict import Predictor
from sklearn.metrics import classification_report

# Set up logging configuration to display messages with timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    """
    Main function to execute the standard pipeline without MLflow tracking.
    Includes:
    - Data ingestion
    - Data cleaning
    - Model training and evaluation
    """
    # Step 1: Load data
    ingestion = Ingestion()
    train, test = ingestion.load_data()
    logging.info("Data ingestion completed successfully")

    # Step 2: Clean data
    cleaner = Cleaner()
    train_data = cleaner.clean_data(train)
    test_data = cleaner.clean_data(test)
    logging.info("Data cleaning completed successfully")

    # Step 3: Prepare and train the model
    trainer = Trainer()
    X_train, y_train = trainer.feature_target_separator(train_data)
    trainer.train_model(X_train, y_train)  # Train the model
    trainer.save_model()  # Save the trained model
    logging.info("Model training completed successfully")

    # Step 4: Evaluate the model
    predictor = Predictor()
    X_test, y_test = predictor.feature_target_separator(test_data)
    accuracy, class_report, roc_auc_score = predictor.evaluate_model(X_test, y_test)
    logging.info("Model evaluation completed successfully")
    
    # Step 5: Print evaluation results
    print("\n============= Model Evaluation Results ==============")
    print(f"Model: {trainer.model_name}")
    print(f"Accuracy Score: {accuracy:.4f}, ROC AUC Score: {roc_auc_score:.4f}")
    print(f"\n{class_report}")
    print("=====================================================\n")


def train_with_mlflow():
    """
    Function to execute the pipeline with MLflow tracking.
    Includes:
    - Data ingestion
    - Data cleaning
    - Model training and evaluation
    - Logging metrics and model to MLflow
    """
    # Load configuration from the YAML file
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    # Set the MLflow experiment name
    mlflow.set_experiment("Group 89 Mlops Assignment 1")

    with mlflow.start_run() as run:
        logging.info("MLflow run started")

        # Step 1: Load data
        ingestion = Ingestion()
        train, test = ingestion.load_data()
        logging.info("Data ingestion completed successfully")

        # Step 2: Clean data
        cleaner = Cleaner()
        train_data = cleaner.clean_data(train)
        test_data = cleaner.clean_data(test)
        logging.info("Data cleaning completed successfully")

        # Step 3: Prepare and train the model
        trainer = Trainer()
        X_train, y_train = trainer.feature_target_separator(train_data)
        trainer.train_model(X_train, y_train)  # Train the model
        trainer.save_model()  # Save the trained model
        logging.info("Model training completed successfully")
        
        # Step 4: Evaluate the model
        predictor = Predictor()
        X_test, y_test = predictor.feature_target_separator(test_data)
        accuracy, class_report, roc_auc_score = predictor.evaluate_model(X_test, y_test)
        report = classification_report(y_test, trainer.pipeline.predict(X_test), output_dict=True)
        logging.info("Model evaluation completed successfully")
        
        # Step 5: Set MLflow tags for better run identification
        mlflow.set_tag('Model developer', 'group89')
        mlflow.set_tag('Preprocessing', 'OneHotEncoder, Standard Scaler, and MinMax Scaler')
        mlflow.set_tag("User ID", 'Mlops Assignment - Group89')
        
        # Step 6: Log parameters and metrics to MLflow
        model_params = config['model']['params']
        mlflow.log_params(model_params)  # Log model hyperparameters
        mlflow.log_metric("accuracy", accuracy)  # Log accuracy
        mlflow.log_metric("roc_auc", roc_auc_score)  # Log ROC AUC score
        mlflow.log_metric('precision', report['weighted avg']['precision'])  # Log precision
        mlflow.log_metric('recall', report['weighted avg']['recall'])  # Log recall
        mlflow.log_param('data_source', 'Insurance dataset')  # Log data source
        mlflow.sklearn.log_model(trainer.pipeline, "model")  # Log the entire model pipeline
    
        # Step 7: Register the model with MLflow
        model_name = trainer.model_name  # Name of the model
        model_uri = f"runs:/{run.info.run_id}/model"  # URI of the logged model
        mlflow.register_model(model_uri, model_name)  # Register the model in MLflow

        logging.info("MLflow tracking completed successfully")

        # Step 8: Print evaluation results
        print("\n============= Model Evaluation Results ==============")
        print(f"Model: {trainer.model_name}")
        print(f"Accuracy Score: {accuracy:.4f}, ROC AUC Score: {roc_auc_score:.4f}")
        print(f"\n{class_report}")
        print("=====================================================\n")
        
if __name__ == "__main__":
    # main()
    # Execute the pipeline with MLflow tracking
    train_with_mlflow()