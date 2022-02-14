"""
Script for Generating Confusion Matrix
Author: Arturo Polanco
Date: February 2022
"""
import os
import json
import logging 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from diagnostics import model_predictions

logging.basicConfig(level=logging.INFO)

###############Load config.json and get path variables
with open('config.json','r') as file:
    config = json.load(file) 

# paths
dataset_csv_path = os.path.join(config['output_folder_path'], "final_data.csv") 
conf_matrix_path = os.path.join(config["output_model_path"], "confusion_matrix.png")


##############Function for reporting
def score_model():
    """
    Calculate a confusion matrix using the test data and the deployed model
    Write the confusion matrix to the workspace
    """
    logging.info("Reading data.")
    data = pd.read_csv(dataset_csv_path)
    data.drop(columns=['corporation'], inplace=True)
    target = data.pop("exited")
    features = data

    logging.info("Model Inference.")
    predictions = model_predictions(features)

    logging.info("Saving Confusion Matrix.")
    plot_confusion_matrix(target, predictions)

def plot_confusion_matrix(true_values, prediction_values, labels=['not exited', 'exited']):
    """
    Calculate a confusion matrix and save it
    """

    logging.info("Plotting and saving confusion matrix")
    conf_matriz = confusion_matrix(
        true_values, 
        prediction_values) 
            
    figure = ConfusionMatrixDisplay(
        confusion_matrix=conf_matriz,
        display_labels=labels)
    
    figure.plot()

    plt.savefig(conf_matrix_path)


  


if __name__ == '__main__':
    score_model()
