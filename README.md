# MLOps Assignment-1 
## Group 89 
List of contributors in this assignment:
1.	P. USHASRI – 2023AA05914
2.	GIRISH KUMAR – 2023AB05007
3.	T CHAITRA – 2023AA05851
4.	ARAVINDAN C - 2023aa05674

## Objective : Insurance Cross Sell Prediction
The goal of this project is to predict which customers are most likely to purchase additional insurance products using a machine learning model.

## Architecture:
<img src="css\architecture.jpg" alt="MLOps Architecture" style="width:800px;height:auto;">

## Get Started
### 1. Clone the Repository
Clone the project repository from GitHub:<br>
> git clone https://github.com/SinghG001/mlops.git

### 2. Set Up the Dev Environment
Ensure you have Python 3.8+ installed. Create a virtual environment and install the necessary dependencies:<br>
Dependencies are listed in requirements.txt
> pip install -r requirements.txt

### 2.1 Source code description
Git is configured on teh folder 'mlops'(root folder)<br>
> Class definitions for data ingestion, clean,train & predict are in 'src' folder
> Data Ingestion logic is in mlops/ingest.py
> Data Cleaning login is in mlops/clean.py
> Logic for model training, hyperperameter tuning and packaging is in mlops/train.py
> Predection logic is in mlops/predict.py
> Model monitoring code is in mlops/monitor.ipynb

### 3. Data Preparation
DVC is configured for data versiniong
Pull the data from DVC. If this command doesn't work, the train and test data are already present in the data folder:

> Data is loaded to mlops/Data folder<br>
* production.csv // dataset with insurance data purchased by customers <br>
* test.csv, train.csv // data split for testing and trainign of the model<br>
(for this exercise data is sourced from local disk instead of cloud locations)
> dvc pull

<img src="css\dvcpull_output.jpg" alt="DVC Pull" style="width:600px;height:auto;">

<img src="css\dvc_source.jpg" alt="DVC Pull" style="width:300px;height:auto;"> <img src="css\dvcpull_data.jpg" alt="DVC Pull" style="width:300px;height:auto;">


### 4. Train the Model
To train the model, run the following command:
> python train.py 

### 5. Predect result
To predect result 
> 1. Deploy the model to docker by running Github Actions pipeline:<br>
Note: before running the git actions, makesure the latest code from all feature branches are merged to main branch and code by pull request.

<img src="css\git_actions.jpg" alt="DVC Pull" style="width:auto;height:auto;">

> 2. Run the image in docker:

Run the following command in docker terminal
> docker run -p 8000:8000 -t girish1808/group89

<img src="css\docker_image.jpg" alt="DVC Pull" style="width:auto;height:auto;">

> 3. Use Thunder client to pust the request to docker and check the response.</br>

<img src="css\thunder_client.jpg" alt="DVC Pull" style="width:auto;height:auto;">

### 5. Monitoring 
> 1. run mlops/moniotor.ipynb
> 2. 'mlops\monitor_model_drift_group69.html' file will be created with data drift results summary
<img src="css\data_drift1.jpg" alt="DVC Pull" style="width:auto;height:auto;">


## Note:
This assignment is to gain hands-on experience with MLOps tools and technology by implementing the following.
1. CI/CD pipelines to manage code, data versining & continues deployment of the models developed
2. Use GitHub for code versioning and DVC for data versioning
3. Use docker for deployments and monitor the model drift.
