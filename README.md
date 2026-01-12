# Car BMW Sales Classifier

FastAPI project for classifying BMW car sales into categories (High / Low) using multiple ML models: **KNN, Logistic Regression, SVC**, and others.

---

## Features
- Preprocessing with `ColumnTransformer` (numeric scaling, OneHotEncoding, BinaryEncoding)
- Handling class imbalance using **SMOTE**
- Trained ML pipelines saved with `joblib`
- Easy API endpoints to predict sales classification

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Yousef-salah-Ma/car_bmw_sales_classifier.git
cd car_bmw_sales_classifier

# Install dependencies
pip install -r requirements.txt
Running the API
bash
Copy code
# Start FastAPI server
uvicorn app:app --reload

# Optionally, open a static server to serve HTML (if needed)
python -m http.server 5500
FastAPI will run by default on http://127.0.0.1:8000

Interactive API docs are available at http://127.0.0.1:8000/docs
| Endpoint              | Method | Description                                   |
| --------------------- | ------ | --------------------------------------------- |
| `/predict_module_KNN` | POST   | Predict sales class using KNN                 |
| `/predict_module_l_r` | POST   | Predict sales class using Logistic Regression |
| `/predict_module_SVC` | POST   | Predict sales class using SVC                 |
Request body example:

{
  "model": "5 Series",
  "region": "Asia",
  "color": "Red",
  "fuel_type": "Petrol",
  "transmission": "Manual",
  "engine_size_l": 3.5,
  "mileage_km": 151748.0,
  "sales_volume": 8300,
  "age_car": 8,
  "price_usd": 98740.0
}


Response example:

{
  "sales_classification": "High"
}

ML Models

The project uses multiple pipelines with preprocessing and SMOTE:

KNeighborsClassifier

LogisticRegression

SVC

Other models tested: DecisionTree, RandomForest, GaussianNB, XGBoost, CatBoost, LightGBM

All models are stored as .pkl files and loaded dynamically by the API.
