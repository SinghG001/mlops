import pandas as pd
import yaml

class Ingestion:
    """
    A class for handling data ingestion processes, including loading configuration files 
    and reading training and testing datasets.
    """
    
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        """
        Loads configuration settings from a YAML file.

        The configuration file should contain paths to the training and testing datasets 
        under the 'data' section.

        """
        with open("config.yml", "r") as file:
            return yaml.safe_load(file)

    def load_data(self):
        """
        Loads the training and testing datasets based on the file paths specified 
        in the configuration file.

        The configuration file must contain the keys:
        - 'data.train_path': Path to the training dataset.
        - 'data.test_path': Path to the testing dataset.
        """
        # Retrieve file paths for training and testing datasets from the configuration
        train_data_path = self.config['data']['train_path']
        test_data_path = self.config['data']['test_path']
        
        # Read the datasets into pandas DataFrames
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
        
        return train_data, test_data
