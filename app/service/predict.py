import joblib
import os
import pandas as pd
import logging

# Load the saved model and encoders

model_path = os.path.join(os.path.dirname(__file__), "../model/loan_model.joblib")
encoders_path = os.path.join(os.path.dirname(__file__), "../model/loan_encoders.joblib")

model = joblib.load(model_path)
label_encoders = joblib.load(encoders_path)


# Setup logging
logging.basicConfig(level=logging.INFO)


def predict_loan_default_from_file() -> list:

    # load csv file
    df = pd.read_csv("app/data/loan_data.csv")
    target_column = "loan_status"
    if target_column in df.columns:
        df = df.drop(columns=[target_column])

    for col, encoder in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except ValueError as e:
                raise ValueError(f"Error encoding column {col}")
        else:
            raise ValueError(f"Column {col} not found in the dataset!!!!")

    predictions = model.predict(df)  # predict the loan status
    return ["Approved" if prediction == 0 else "Rejected" for prediction in predictions]
