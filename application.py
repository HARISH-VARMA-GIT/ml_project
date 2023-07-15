from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            laufkont=request.form.get('laufkont'),
            laufzeit=request.form.get('laufzeit'),
            moral=request.form.get('moral'),
            verw=request.form.get('verw'),
            hoehe=request.form.get('hoehe'),
            sparkont=request.form.get('sparkont'),
            beszeit=request.form.get('beszeit'),
            rate=float(request.form.get('rate')),
            famges=float(request.form.get('famges')),
            buerge=float(request.form.get('buerge')),
            wohnzeit=float(request.form.get('wohnzeit')),
            verm=float(request.form.get('verm')),
            alter=float(request.form.get('alter')),
            weitkred=float(request.form.get('weitkred')),
            wohn=float(request.form.get('wohn')),
            bishkred=float(request.form.get('bishkred')),
            beruf=float(request.form.get('beruf')),
            pers=float(request.form.get('pers')),
            telef=float(request.form.get('telef')),
            gastarb=float(request.form.get('gastarb'))
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('prediction.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")