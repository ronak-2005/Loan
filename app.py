from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib
import os

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


def render_page(table_html: str = "") -> str:
    """Reusable page renderer with optional results table."""
    return f"""
    <html>
    <head>
        <title>Loan Prediction</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; text-align: center; }}
            form {{ margin: 20px auto; }}
            input[type=file] {{ margin-bottom: 10px; }}
            button {{ padding: 8px 16px; background: #007BFF; color: white;
                      border: none; border-radius: 4px; cursor: pointer; }}
            button:hover {{ background: #0056b3; }}
            table {{ border-collapse: collapse; width: 80%; margin: 20px auto; }}
            th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: center; }}
            th {{ background-color: #f4f4f4; }}
        </style>
    </head>
    <body>
        <h2>Upload CSV for Loan Prediction</h2>
        <form action="/" enctype="multipart/form-data" method="post">
            <input type="file" name="file" accept=".csv" required><br>
            <button type="submit">Predict</button>
        </form>
        {table_html}
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
async def home():
    """Initial upload form page."""
    return HTMLResponse(content=render_page())


@app.post("/", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    """Upload CSV, predict, and render results below form."""
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

    # Build HTML table
    rows_html = ""
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        rows_html += f"""
            <tr>
                <td>{i}</td>
                <td>{"Approved" if pred == 1 else "Rejected"}</td>
                <td>{float(prob[1]):.4f}</td>
                <td>{float(prob[0]):.4f}</td>
            </tr>
        """

    table_html = f"""
        <h2>Loan Predictions ({len(df_test)} rows)</h2>
        <p>⚠️ Processed only first {MAX_ROWS} rows for performance</p>
        <table>
            <thead>
                <tr><th>Index</th><th>Prediction</th><th>Approved Prob</th><th>Rejected Prob</th></tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    """

    return HTMLResponse(content=render_page(table_html))
