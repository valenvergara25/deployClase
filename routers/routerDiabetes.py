import pickle
from fastapi import APIRouter
from Diabetesdata import Diabetesdata
from schemas import schemas
import numpy as np

router = APIRouter()

pkl_filename = "RFDiabetes.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

labels = ["Sano", "Posible diabetes"]

@router.get("/")
async def root():
    return{
        "message": "AI Service"
    }

@router.post("/predict")
def predict_diabetes(data:schemas.Diabetesdata):
    data = data.model_dump()
    Pregnancies = data['Pregnancies']
    Glucose = data['Glucose']
    BloodPressure = data['BloodPressure']
    SkinThickness = data['SkinThickness']
    Insulin = data['Insulin']
    BMI = data['BMI']
    DiabetesPedigreeFunction = data['DiabetesPedigreeFunction']
    Age = data['Age']

    xin = np.array([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]).reshape(1,8)

    prediction = model.predict(xin)
    yout = labels[prediction[0]]

    return {
        'prediction': yout
    }