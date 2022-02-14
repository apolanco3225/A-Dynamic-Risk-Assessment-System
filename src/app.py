"""
Script for Deploying Machine Learning Model as API
Author: Arturo Polanco
Date: February 2022
"""
# import necessary packages
import pickle
import json
import os
import re
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import subprocess
from diagnostics import execution_time
from diagnostics import missing_data_percentage
from diagnostics import dataframe_summary
from diagnostics import outdated_packages_list
from diagnostics import model_predictions

with open('config.json','r') as file:
    config = json.load(file) 

# ingested data path
output_folder_path = config['output_folder_path']
ingested_data_path = os.path.join(output_folder_path, 'final_data.csv')

test_data = pd.read_csv(ingested_data_path)
test_data.drop(columns=["corporation", "exited"], inplace=True)



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as file:
    config = json.load(file) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


@app.route('/')
def index():
    return "Greetings World!"



#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    data_json = request.get_json()
    data = pd.DataFrame(data_json['data'])
    data.drop(columns=['corporation', 'exited'], inplace=True)
    print(data)
    predictions = model_predictions(data)
    print(predictions)
    return jsonify(predictions)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring_data():        
    output_score = subprocess.check_output('python src/scoring.py',
                            shell=True)
    output_score = output_score.decode('UTF-8')
        
    return jsonify(output_score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary_statistics_data():    
    summary = dataframe_summary(test_data)
    return jsonify(summary)


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def measure_time_exectution():        
    #check timing and percent NA values
    time = execution_time()
    missing_data = missing_data_percentage(test_data)
    outdated_modules = outdated_packages_list()

    return jsonify(
        time,
        missing_data, 
        #outdated_modules
    )

if __name__ == "__main__": 
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
