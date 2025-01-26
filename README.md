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

### 3. Data Preparation
DVC is configured for data versiniong
Pull the data from DVC. If this command doesn't work, the train and test data are already present in the data folder:

> Data is loaded to mlops/Data folder<br>
* production.csv // dataset with insurance data purchased by customers <br>
* test.csv, train.csv // data split for testing and trainign of the model<br>
(for this exercise data is sourced from local disk instead of cloud locations)
> dvc pull

<img src="css\dvcpull.jpg" alt="DVC Pull" style="width:600px;height:auto;">