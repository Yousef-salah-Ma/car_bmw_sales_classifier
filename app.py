from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Car Sales Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],  
    allow_headers=["*"],
)


# Load trained classification pipeline
module_KNN = joblib.load("module_KNN.pkl")
module_l_r = joblib.load("module_l_r.pkl")
module_SVC = joblib.load("module_SVC.pkl")



class CarData(BaseModel):
    # feature
    model: str
    region: str
    color: str
    fuel_type: str
    transmission: str
    engine_size_l: float
    mileage_km: float
    sales_volume: int
    age_car: int
    price_usd: float  

@app.post("/predict_module_KNN")
def predict_sales_class_module_KNN(car: CarData):
    # Create a DataFrame with column names
    input_df = pd.DataFrame([{
        'model': car.model,
        'region': car.region,
        'color': car.color,
        'fuel_type': car.fuel_type,
        'transmission': car.transmission,
        'engine_size_l': car.engine_size_l,
        'mileage_km': car.mileage_km,
        'sales_volume': car.sales_volume,
        'age_car': car.age_car,
        'price_usd': car.price_usd
    }])

    try:
        prediction = module_KNN.predict(input_df)

        pred_label = int(prediction[0])
        sales_class = "High" if pred_label == 0 else "Low"

        return {"sales_classification": sales_class}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/predict_module_l_r")
def predict_sales_class_module_l_r(car: CarData):
    # Create a DataFrame with column names
    input_df = pd.DataFrame([{
        'model': car.model,
        'region': car.region,
        'color': car.color,
        'fuel_type': car.fuel_type,
        'transmission': car.transmission,
        'engine_size_l': car.engine_size_l,
        'mileage_km': car.mileage_km,
        'sales_volume': car.sales_volume,
        'age_car': car.age_car,
        'price_usd': car.price_usd
    }])

    try:
        prediction = module_l_r.predict(input_df)

        pred_label = int(prediction[0])
        sales_class = "High" if pred_label == 0 else "Low"

        return {"sales_classification": sales_class}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
        
@app.post("/predict_module_SVC")
def predict_sales_class_module_SVC(car: CarData):
    # Create a DataFrame with column names
    input_df = pd.DataFrame([{
        'model': car.model,
        'region': car.region,
        'color': car.color,
        'fuel_type': car.fuel_type,
        'transmission': car.transmission,
        'engine_size_l': car.engine_size_l,
        'mileage_km': car.mileage_km,
        'sales_volume': car.sales_volume,
        'age_car': car.age_car,
        'price_usd': car.price_usd
    }])

    try:
        prediction = module_SVC.predict(input_df)

        pred_label = int(prediction[0])
        sales_class = "High" if pred_label == 0 else "Low"

        return {"sales_classification": sales_class}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))