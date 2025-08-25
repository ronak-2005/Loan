from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib

app = FastAPI()
MAX_ROWS = 10000  

templates = Jinja2Templates(directory="templates")

model = joblib.load("loan.pkl")
preprocessor = joblib.load("preprocessor.pkl")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file, nrows=MAX_ROWS)
    df_test = df.drop(columns=["loan_status"], errors="ignore")

    X = preprocessor.transform(df_test)

    preds = model.predict(X)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
    else:
        probs = [[1 if p == 1 else 0, 1 if p == 0 else 0] for p in preds]

    results = []
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        results.append({
            "index": i,
            "prediction_label": "Approved" if pred == 1 else "Rejected",
            "confidence": {
                "approved": float(prob[1]),
                "rejected": float(prob[0])
            }
        })

    return {
        "predictions": results,
        "rows_processed": len(df_test),
        "note": f"⚠️ Processed only first {MAX_ROWS} rows for performance"
    }
