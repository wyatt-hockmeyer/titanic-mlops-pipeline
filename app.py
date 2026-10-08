# app.py
# REST API that serves survival predictions from the trained model
# Start from the project folder with: python -m uvicorn app:app --reload

# Import os to read the API key from an environment variable
import os

# Import pandas to build a one-row table for the model
import pandas as pd

# Import joblib to load the saved model (Lesson 39 Demo 1)
import joblib

# Import FastAPI, plus Header and HTTPException for the API key check (documented exception)
from fastapi import FastAPI, Header, HTTPException

# Import BaseModel to define the shape of the input data (Lesson 39 Demo 1)
from pydantic import BaseModel

# Load the trained model saved by train.py
model = joblib.load("model.pkl")

# Read the API key from the environment; use a labeled local default if it is not set
API_KEY = os.environ.get("API_KEY", "local-dev-key")

# The highest fare allowed, matching the 95th percentile cap in preprocess.py
fare_cap = 113.275

# The 15 feature columns, in the same order the model was trained on
feature_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "HasCabin", "FamilySize", "IsAlone", "Embarked_Q", "Embarked_S", "Title_Miss", "Title_Mr", "Title_Mrs", "Title_Rare"]

# Create the API application
app = FastAPI()


# Define the passenger fields the API accepts
# Each field lists its type (int, str, float); this syntax is from the FastAPI documentation
class PassengerData(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    Title: str
    HasCabin: int


# Define a function that groups each title into one of five categories (same as preprocess.py)
def group_title(title):
    # Keep the four common titles as they are
    if title == "Mr" or title == "Mrs" or title == "Miss" or title == "Master":
        return title
    # French and modern forms of Miss
    elif title == "Mlle" or title == "Ms":
        return "Miss"
    # French form of Mrs
    elif title == "Mme":
        return "Mrs"
    # Everything else is rare
    else:
        return "Rare"


# Define a simple check page that confirms the API is running
@app.get("/")
def home():
    return {"status": "Titanic survival API is running"}


# Define the prediction endpoint
@app.post("/predict/")
def predict(data: PassengerData, x_api_key: str = Header(None)):
    # Reject the request if the API key is missing or wrong
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # Reject a sex value the model cannot use
    if data.Sex != "male" and data.Sex != "female":
        return {"error": "Sex must be male or female"}

    # Reject a port value the model cannot use
    if data.Embarked != "C" and data.Embarked != "Q" and data.Embarked != "S":
        return {"error": "Embarked must be C, Q, or S"}

    # Encode sex as a number: male 0, female 1
    if data.Sex == "female":
        sex = 1
    else:
        sex = 0

    # Cap the fare the same way preprocess.py does
    fare = data.Fare
    if fare > fare_cap:
        fare = fare_cap

    # Create family size and travelling-alone features
    family_size = data.SibSp + data.Parch + 1
    if family_size == 1:
        is_alone = 1
    else:
        is_alone = 0

    # Group the title
    title = group_title(data.Title)

    # Build one row of features, with 0 or 1 for each port and title column
    row = {
        "Pclass": data.Pclass,
        "Sex": sex,
        "Age": data.Age,
        "SibSp": data.SibSp,
        "Parch": data.Parch,
        "Fare": fare,
        "HasCabin": data.HasCabin,
        "FamilySize": family_size,
        "IsAlone": is_alone,
        "Embarked_Q": int(data.Embarked == "Q"),
        "Embarked_S": int(data.Embarked == "S"),
        "Title_Miss": int(title == "Miss"),
        "Title_Mr": int(title == "Mr"),
        "Title_Mrs": int(title == "Mrs"),
        "Title_Rare": int(title == "Rare")
    }

    # Turn the row into a one-row table with the columns in training order
    features = pd.DataFrame([row], columns=feature_columns)

    # Predict survived (1) or not (0)
    prediction = model.predict(features)

    # Get the predicted probability of survival
    probability = model.predict_proba(features)[0][1]

    # Return the prediction and probability
    return {"prediction": int(prediction[0]), "survival_probability": round(float(probability), 3)}