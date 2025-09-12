from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib
import os
import numpy as np

app = FastAPI(title="Loan Prediction")
MAX_ROWS = 10000  # Prevent too large uploads

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths (Render vs Local)
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
    raise FileNotFoundError(
        f"Model or preprocessor file not found in Render or Local paths."
    )

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Upload form for CSV."""
    html_content = """
    <html>
    <head>
        <title>Loan Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; text-align: center; }
            form { margin: 20px auto; }
            input[type=file] { margin-bottom: 10px; }
            button { padding: 8px 16px; background: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <h2>Upload CSV for Loan Prediction</h2>
        <form action="/predict_csv" enctype="multipart/form-data" method="post">
            <input type="file" name="file" accept=".csv" required><br>
            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/predict_csv", response_class=HTMLResponse)
async def predict_csv(file: UploadFile = File(...)):
    """Predict loan approvals from CSV and return HTML table."""
    df = pd.read_csv(file.file, nrows=MAX_ROWS)
    df_test = df.drop(columns=["loan_status"], errors="ignore")

    # Preprocess and predict
    X = preprocessor.transform(df_test)
    preds = model.predict(X)

    # Probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
    else:
        # fallback for models without predict_proba
        probs = [[1 if p == 1 else 0, 1 if p == 0 else 0] for p in preds]

    # Prepare results
    results = []
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        results.append({
            "index": i,
            "prediction_label": "Approved" if pred == 1 else "Rejected",
            "approved_prob": float(prob[1]),
            "rejected_prob": float(prob[0])
        })

    # Build HTML table
    table_html = f"""
    <html>
    <head>
        <title>Loan Predictions</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 80%; margin: auto; }}
            th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: center; }}
            th {{ background-color: #f4f4f4; }}
            h2 {{ text-align: center; }}
            .back {{ display: block; text-align: center; margin-top: 20px; }}
            .btn {{ padding: 6px 12px; background: #007BFF; color: white; border: none; border-radius: 4px; text-decoration: none; }}
            .btn:hover {{ background: #0056b3; }}
        </style>
    </head>
    <body>
        <h2>Loan Predictions ({len(df_test)} rows)</h2>
        <p>⚠️ Processed only first {MAX_ROWS} rows for performance</p>
        <table>
            <thead>
                <tr><th>Index</th><th>Prediction</th><th>Approved Prob</th><th>Rejected Prob</th></tr>
            </thead>
            <tbody>
    """

    for row in results:
        table_html += f"""
            <tr>
                <td>{row['index']}</td>
                <td>{row['prediction_label']}</td>
                <td>{row['approved_prob']:.4f}</td>
                <td>{row['rejected_prob']:.4f}</td>
            </tr>
        """

    table_html += """
            </tbody>
        </table>
        <div class="back">
            <a class="btn" href="/">Upload Another CSV</a>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(content=table_html)
