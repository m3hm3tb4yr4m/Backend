import pandas as pd
import numpy as np
#from datetime import date, timedelta
from fastapi import FastAPI, File, HTTPException
import lightgbm as lgb
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import joblib
from joblib import dump, load
import uvicorn
from sklearn.exceptions import NotFittedError
import sklearn

app = FastAPI(
    title="Home Credit Default Risk",
    description="""Obtain information related to probability of a client defaulting on loan.""",
#    version="0.1.0",
)


########################################################
# Reading the csv
########################################################
df_clients = pd.read_csv("Data/reduced_X_test.csv")
df_clients["SK_ID_CURR"] = df_clients["SK_ID_CURR"].astype(int)
# Loading the model
filename = "Data/lgbm_classifier.pkl"
with open(filename, 'rb') as fo:
    model = joblib.load(fo)

@app.get("/api/clients")
async def clients_id():
    #liste des clients
    clients_id = df_clients["SK_ID_CURR"].tolist()
    return {"clientsId": clients_id}

@app.get("/api/clients/{id}")
async def client_details(id: int):
    #client parmi liste des clients 
    clients_id = df_clients["SK_ID_CURR"].tolist()

    if id not in clients_id:
        raise HTTPException(status_code=404, detail="client's id not found")
    else:
        # Filtering by client's id
        client=df_clients[df_clients["SK_ID_CURR"] == id].to_dict()
    return client


@app.get("/api/predictions/clients/{id}")
async def predict(id: int):
    """ 
    EndPoint to get the probability honor/compliance of a client
    """ 

    clients_id = df_clients["SK_ID_CURR"].tolist()

    if id not in clients_id:
        raise HTTPException(status_code=404, detail="client's id not found")
    else:
        # Loading the model
        filename= "Data/lgbm_classifier.pkl"
        with open(filename, 'rb') as fo:
            model=joblib.load(fo)
        threshold = 0.103448

        # Filtering by client's id
        df_prediction_by_id = df_clients[df_clients["SK_ID_CURR"] == id]
        #df_prediction_by_id = df_prediction_by_id.drop(df_prediction_by_id.columns[[0, 1]], axis=1)
        # Predicting
        try:
            result_proba = model.predict_proba(df_prediction_by_id)
            y_prob = result_proba[:, 1]
            result = (y_prob >= threshold).astype(int)
            if (int(result[0]) == 0):
                result = "Yes"
            else:
                result = "No"
            return {
                "repay": result,
                "probability0": result_proba[0][0],
                "probability1": result_proba[0][1],
                "threshold": threshold
            }
        except NotFittedError as e:
            print(e)

@app.get("/api/model")
async def load_model():
    return model








    
    
if __name__=='__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)    