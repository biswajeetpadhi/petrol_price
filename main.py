
from pandas import read_csv
from pandas import to_datetime
import pandas as pd
from fastapi import FastAPI, UploadFile
import pickle
import uvicorn
from fastapi.responses import FileResponse

model = pickle.load(open("petrol_price.pkl", "rb"))

app = FastAPI(
    title="Petrol_Price_Forecasting",
    description="A simple API to forecast petrol price",
    version="0.1")


@app.post("/upload_file/")
async def create_upload_file(file: UploadFile):
    df_test = read_csv("test_data.csv")
    df_test.columns = ["date", "prediction"]
    df_test["date"] = to_datetime(df_test["date"])

    forecast = model.predict(n_periods=16)
    forecast_df = pd.DataFrame(forecast, columns=["prediction"])
    forecast_df.set_index(df_test["date"], inplace=True)
    forecast_df.to_csv("forecast_df.csv")

    return FileResponse("forecast_df.csv")

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host='127.0.0.1')
