"""
Script for Scoring Machine Learning Model
Author: Arturo Polanco
Date: February 2022
"""
import pandas as pd
import numpy as np
import pickle
import os
import logging
import pickle
from sklearn import metrics

import json

logging.basicConfig(level=logging.INFO)



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

score_artifact_path = os.path.join(config['output_folder_path'], "latest_score.txt") 
test_data_path = os.path.join(config['test_data_path'], "testdata.csv") 
model_path = os.path.join(config['output_model_path'], "trained_model.pckl") 


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    logging.info("Loading test data.")
    test_data = pd.read_csv(test_data_path)
    test_target = test_data.pop("exited")
    test_features = test_data
    test_features.drop(["corporation"], axis=1, inplace=True)

    logging.info("Loading trained model.")
    model_handler = open(model_path, "rb")
    model = pickle.load(model_handler)

    logging.info("Model inference.")
    predictions = model.predict(test_features)
    f1_score_value = metrics.f1_score(predictions, test_target)
    f1_score_value = str(f1_score_value)

    logging.info("Saving scores in text file.")

    with open(score_artifact_path, 'w') as file:
        file.write(f"F1 Score Value: {f1_score_value}")

    score_message = str(f"F1 Score Value: {f1_score_value}")
    return score_message



if __name__ == '__main__':
    message = score_model()
    print(message)
