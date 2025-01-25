from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Initialize the FastAPI app
app = FastAPI()


"""
InputData classrRepresents the structure of input data for the prediction endpoint.
Each attribute corresponds to a feature required by the model.
"""
# Define the data structure for the input 
class InputData(BaseModel):
    Gender: str
    Age: int
    HasDrivingLicense: int
    RegionID: float
    Switch: int
    PastAccident: str
    AnnualPremium: float

# Load the trained model from the file system
model = joblib.load('models/model.pkl')  # Ensure the correct path to the model file

@app.get("/")
async def read_root():
    return {"health_check": "OK", "model_version": 1}

@app.post("/predict")
async def predict(input_data: InputData):
    # Convert the input data to a pandas DataFrame
    df = pd.DataFrame(
        [input_data.model_dump().values()],  # Convert Pydantic model to a dictionary
        columns=input_data.model_dump().keys()  # Use keys as column names
    )
    
    # Generate a prediction using the loaded model
    pred = model.predict(df)
    
    # Return the prediction as a JSON response
    return {"predicted_class": int(pred[0])}
