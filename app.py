from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
import os

app = FastAPI(title="Loan Prediction")
MAX_ROWS = 10000  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODEL_PATH_RENDER = "/mnt/data/loan.pkl"
PREPROCESSOR_PATH_RENDER = "/mnt/data/preprocessor.pkl"
MODEL_PATH_LOCAL = os.path.join(BASE_DIR, "loan.pkl")
PREPROCESSOR_PATH_LOCAL = os.path.join(BASE_DIR, "preprocessor.pkl")

# Load model + preprocessor
if os.path.exists(MODEL_PATH_RENDER) and os.path.exists(PREPROCESSOR_PATH_RENDER):
    model_path, preprocessor_path = MODEL_PATH_RENDER, PREPROCESSOR_PATH_RENDER
elif os.path.exists(MODEL_PATH_LOCAL) and os.path.exists(PREPROCESSOR_PATH_LOCAL):
    model_path, preprocessor_path = MODEL_PATH_LOCAL, PREPROCESSOR_PATH_LOCAL
else:
    raise FileNotFoundError("Model or preprocessor not found.")

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

# ✅ Setup templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # render form_loan.html instead of static file
    return templates.TemplateResponse("form_loan.html", {"request": request})


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file, nrows=MAX_ROWS)
    df_test = df.drop(columns=["loan_status"], errors="ignore")

    # Preprocess and predict
    X = preprocessor.transform(df_test)
    preds = model.predict(X)

    # Probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
    else:
        probs = [[1 if p == 1 else 0, 1 if p == 0 else 0] for p in preds]

    results = []
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        results.append({
            "Index": int(i),
            "Prediction": "Approved" if pred == 1 else "Rejected",
            "Approved Probability": round(float(prob[1]), 4),
            "Rejected Probability": round(float(prob[0]), 4),
        })

    return JSONResponse(content={"predictions": results})
