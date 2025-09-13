from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Loan Approval Prediction")
MAX_ROWS = 10000  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODEL_PATH_RENDER = "/mnt/data/loan.pkl"
PREPROCESSOR_PATH_RENDER = "/mnt/data/preprocessor.pkl"
MODEL_PATH_LOCAL = os.path.join(BASE_DIR, "loan.pkl")
PREPROCESSOR_PATH_LOCAL = os.path.join(BASE_DIR, "preprocessor.pkl")

# Load model + preprocessor
try:
    if os.path.exists(MODEL_PATH_RENDER) and os.path.exists(PREPROCESSOR_PATH_RENDER):
        model_path, preprocessor_path = MODEL_PATH_RENDER, PREPROCESSOR_PATH_RENDER
        logger.info("Using Render paths for model files")
    elif os.path.exists(MODEL_PATH_LOCAL) and os.path.exists(PREPROCESSOR_PATH_LOCAL):
        model_path, preprocessor_path = MODEL_PATH_LOCAL, PREPROCESSOR_PATH_LOCAL
        logger.info("Using local paths for model files")
    else:
        raise FileNotFoundError("Model or preprocessor not found.")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    logger.info("Model and preprocessor loaded successfully")
    
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise e

# Setup templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main HTML page"""
    try:
        return templates.TemplateResponse("form_loan.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving home page: {str(e)}")
        return HTMLResponse("""
        <!DOCTYPE html>
        <html><head><title>Loan Approval Predictor</title></head>
        <body>
        <h1>Loan Approval Predictor</h1>
        <form action="/predict_csv" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".csv" required>
            <button type="submit">Upload & Predict</button>
        </form>
        </body></html>
        """)

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    """Handle CSV upload and return loan approval predictions as JSON"""
    
    logger.info(f"Received file: {file.filename}")
    
    try:
        # Validate file type
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Please upload a CSV file")
        
        # Read CSV
        try:
            df = pd.read_csv(file.file, nrows=MAX_ROWS)
            logger.info(f"CSV loaded. Shape: {df.shape}")
            logger.info(f"CSV columns: {list(df.columns)}")
        except Exception as e:
            logger.error(f"Error reading CSV: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")
        
        # Prepare test data (remove target column if present)
        try:
            df_test = df.drop(columns=["loan_status"], errors="ignore")
            logger.info(f"Test data prepared. Shape: {df_test.shape}")
        except Exception as e:
            logger.error(f"Error preparing test data: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error preparing data: {str(e)}")
        
        # Preprocess data
        try:
            X = preprocessor.transform(df_test)
            logger.info(f"Data preprocessed. Shape: {X.shape}")
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error preprocessing data: {str(e)}")
        
        # Make predictions
        try:
            preds = model.predict(X)
            logger.info(f"Predictions completed. Count: {len(preds)}")
            
            # Get probabilities if available
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                logger.info("Probabilities calculated")
            else:
                # Fallback: create dummy probabilities
                probs = [[1.0 if p == 0 else 0.0, 1.0 if p == 1 else 0.0] for p in preds]
                logger.info("Using fallback probabilities")
                
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")
        
        # Prepare all results
        all_results = []
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            all_results.append({
                "index": i,
                "Prediction": "Approved" if pred == 1 else "Rejected",
                "Approved Probability": round(float(prob[1]), 4),
                "Rejected Probability": round(float(prob[0]), 4),
            })
        
        logger.info(f"Returning all {len(all_results)} predictions")
        
        return {
            "status": "success",
            "predictions": all_results,
            "total_rows_processed": len(all_results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)