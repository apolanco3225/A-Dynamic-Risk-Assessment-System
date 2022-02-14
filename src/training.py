"""
Script for Training Machine Learning Model
Author: Arturo Polanco
Date: February 2022
"""
# import necessary packages
import pandas as pd
import numpy as np
import pickle
import os
import logging
from sklearn.linear_model import LogisticRegression
import json

logging.basicConfig(level=logging.INFO)


###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], "final_data.csv") 
model_path = os.path.join(config['output_model_path'], "trained_model.pckl") 


#################Function for training the model
def train_model():

    logging.info("Loading data.")
    data = pd.read_csv(dataset_csv_path)
    target = data.pop("exited")
    features = data.drop("corporation", axis=1)
    
    #use this logistic regression for training
    model = LogisticRegression(
        C=1.0, 
        class_weight=None, 
        dual=False, 
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,            
        multi_class='auto', 
        n_jobs=None, 
        penalty='l2',            
        random_state=0, 
        solver='liblinear', 
        tol=0.0001, 
        verbose=0,            
        warm_start=False
        )
    
    #fit the logistic regression to your data
    logging.info("Training machine learning algorithm.")
    model.fit(features, target)
    #write the trained model to your workspace in a file called trainedmodel.pkl
    logging.info("Saving model.")
    model_handler = open(model_path, "wb")
    pickle.dump(model, model_handler)

if __name__ == "__main__":
    train_model()

