from flask import Flask
from flask import jsonify, request, make_response, url_for, redirect
from json import dumps
from requests import post
from tensorflow.keras.models import load_model
import logging as log

app = Flask(__name__)


@app.route('/hello/', methods=['GET', 'POST'])
def welcome():
    return "Hello World!"



# Expecting a json of features
'''
{
    'Issue Date': 20180313.0,
    'Issue Time': 832.0,
    'RP State Plate': 'CA',
    'Plate Expiry Date': 201811.0,
    'Body Style': 'PU',
    'Color': 'GY',
    'Location': '1205 SHERBOURNE DR',
    'Route': '00146',
    'Agency': 51.0,
    'Violation Code': '80.69BS',
    'Violation Description': 'NO PARK/STREET CLEAN',
    'Fine Amount': 73.0,
    'Latitude': 6446731.0,
    'Longitude': 1842787.0
}
'''


@app.route('/predict/', methods=['POST'])
def query_model():
    # Because I wasn't able to get a model working for the car data, I am using a model trained on the
    # pima-indian-diabetes dataset for the purposes of building this API

    ###################### this code would be used with a model trained on the car data ######################
    # apply same preprocessing as was applied to the training data for the respective fields to get them
    # all to be floats

    # issue_date = request.json['Issue Date']
    # issue_time = request.json['Issue Date']
    # plate_expiry_date = request.json['Issue Date']
    # agency = request.json['Agency']
    # fine_amount = request.json['Fine Amount']
    # latitude = request.json['Latitude']
    # longitude = request.json['Longitude']
    #
    # x_new = [issue_date, issue_time, plate_expiry_date, agency, fine_amount, latitude, longitude]

    ##########################################################################################################

    Pregnancies = request.json['Pregnancies']
    Glucose = request.json['Glucose']
    BloodPressure = request.json['BloodPressure']
    SkinThickness = request.json['SkinThickness']
    Insulin = request.json['Insulin']
    BMI = request.json['BMI']
    DiabetesPedigreeFunction = request.json['DiabetesPedigreeFunction']
    Age = request.json['Age']

    x_new = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    # x_new = [6, 148, 72, 35, 0, 33.6, 0.627, 50]

    model = load_model('model')
    log.info(model.summary())
    # y_new = model.predict_proba(x_new)
    y_new = 1
    log.info(y_new)

    # determine which indices represent the probabilities for the top 25 cars and add those up and return that value

    return str(y_new)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
