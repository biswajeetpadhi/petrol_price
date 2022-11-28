
import pickle
from fastapi import FastAPI, File, UploadFile, Request, Response, HTTPException
from fastapi.templating import Jinja2Templates
from io import BytesIO
import pandas as pd
import uvicorn

model = pickle.load(open("petrol_price.pkl", "rb"))

templates = Jinja2Templates(directory='templates')

app = FastAPI(
    title="Petrol_Price_Forecasting",
    description="A simple API to forecast petrol price",
    version="0.1")


@app.get('/')
async def func(request: Request):
    return templates.TemplateResponse('home.html', {'request': request})


@app.post("/petrol_price")
async def create_upload_file(file: UploadFile = File(...)):

    try:
        contents = file.file.read()
        buffer = BytesIO(contents)
        df = pd.read_csv(buffer)
    except:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        buffer.close()
        file.file.close()

    df.columns = ["date", "prediction"]
    df["date"] = pd.to_datetime(df["date"])

    forecast = model.predict(n_periods=16)
    forecast_df = pd.DataFrame(forecast, columns=["prediction"])
    forecast_df.set_index(df["date"], inplace=True)

    headers = {'Content-Disposition': 'attachment; filename="forecast_data.csv"'}
    return Response(forecast_df.to_csv(), headers=headers, media_type='text/csv')

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
