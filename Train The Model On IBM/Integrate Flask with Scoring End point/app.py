import flask
from flask import request, render_template
from flask_cors import CORS
import joblib
import pandas as pd
from xgboost import XGBRegressor
import requests
app = flask.Flask(__name__)
CORS(app)

# purposely kept API KEY since cuh is very less

API_KEY = "t1xJwH_pNvesyStso2tawTlpypHX0HEQJVMev99cmAtK"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/services')
def services():
    return render_template('services.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/windapi',methods=['POST'])
def windapi():
    city=request.form.get('city')
    apikey="86b1a085e43cad23bfd9c45d5fd88fc3"
    url="http://api.openweathermap.org/data/2.5/weather?q="+city+"&appid="+apikey
    resp = requests.get(url)
    resp=resp.json()
    temp = str(float(resp["main"]["temp"])-273.15)+" °C"
    humid = str(resp["main"]["humidity"])+" %"
    pressure = str(resp["main"]["pressure"])+" mmHG"
    speed = str(float(resp["wind"]["speed"])*0.44704)+" m/s"
    return render_template('predict.html', temp=temp, humid=humid, pressure=pressure, speed=speed)
@app.route('/y_predict',methods=['POST'])
def y_predict():
    ws = float(request.form['theo'])
    wd = float(request.form['wind'])
    X = [[ws, wd]]
    xgr = XGBRegressor()
    df = pd.DataFrame(X, columns=['WindSpeed(m/s)', 'WindDirection'])
    payload_scoring = {"input_data": [{"field": [['ws', 'wd']], "values":X}]}
    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/0644c680-478f-475f-bc23-2a64fc6490a5/predictions?version=2022-10-24', json=payload_scoring,headers={'Authorization': 'Bearer ' + mltoken})
    print(response_scoring)
    predictions = response_scoring.json()
    print(predictions)
    output = predictions['predictions'][0]['values'][0][0]
    print("Final prediction :", predict)
    return render_template('predict.html', prediction_text="The energy predicted is {:.2f} KWh".format(output))

if __name__ == "__main__":
    app.run()
