from fastapi import FastAPI, HTTPException
from app.schema.input_data import LoanInput
from app.service.predict import predict_loan_default_from_file

app = FastAPI()


@app.get("/")
def root():
    return "Welcome to the Loan Default Prediction API"


@app.post("/predict")
def predict_loan_default() -> dict:
    try:
        result = predict_loan_default_from_file()
        return {"predictions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
