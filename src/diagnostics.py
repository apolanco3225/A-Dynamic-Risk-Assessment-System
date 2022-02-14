"""
Script for Diagnostics Machine Learning Project
Author: Arturo Polanco
Date: February 2022
"""
# import necessary packages
import pandas as pd
import numpy as np
import timeit
import os
import json
import logging
import pickle
import subprocess


# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

# deployed model path
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(prod_deployment_path, "trained_model.pckl")

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])

# ingested data path
output_folder_path = config['output_folder_path']
ingested_data_path = os.path.join(output_folder_path, 'final_data.csv')

logging.basicConfig(level=logging.INFO)


# Function to get model predictions
def model_predictions(test_features):
    # read the deployed model and a test dataset, calculate predictions
    logging.info("Loading deployed model.")
    model_handler = open(model_path, "rb")
    model = pickle.load(model_handler)

    logging.info("Making predictions.")
    predictions = model.predict(test_features)
    return predictions

# Function to get summary statistics


def dataframe_summary(data):
    # calculate summary statistics here

    logging.info("Calculating data statistics.")
    data_statistics = {}
    for column in data.columns:
        mean = data[column].mean()
        median = data[column].median()
        std = data[column].std()

        data_statistics[column] = {'mean': mean, 'median': median, 'std': std}

    return data_statistics


def measure_time(file_name):
    """
    Measure execution time of a file.
    """
    tic = timeit.default_timer()
    _ = subprocess.run(['python', file_name], capture_output=True)
    toc = timeit.default_timer()
    total_time = toc - tic
    return total_time


# Function to get timings
def execution_time():
    # calculate timing of training.py and ingestion.py

    logging.info("Calculating Ingestion Time.")
    ingestion_time = []
    for _ in range(20):
        time = measure_time("ingestion.py")
        ingestion_time.append(time)

    logging.info("Calculating Training Time.")
    training_time = []
    for _ in range(20):
        time = measure_time("training.py")
        training_time.append(time)

    execution_time_output = [
        {'ingestion_time_mean': np.mean(ingestion_time)},
        {'train_time_mean': np.mean(training_time)}
    ]

    return execution_time_output

# Function to check dependencies


def outdated_packages_list():
    """
    Check dependencies status from requirements.txt file using pip-outdated
    which checks each package status if it is outdated or not
    Returns:
        str: stdout of the pip-outdated command
    """
    logging.info("Checking outdated dependencies")
    
    current_working_directory = os.path.dirname(__file__)
    requirements_directory = os.path.join(current_working_directory, "../")
    requirements_path = os.path.join(requirements_directory, "requirements.txt")

    outdated_dependencies = subprocess.check_output(
        f'pip-outdated {requirements_path}',
        shell=True
    )
    outdated_dependencies = outdated_dependencies.decode('UTF-8')

    return outdated_dependencies


def missing_data_percentage(data):
    """
    Calculate the percentage of missing values
    in each column.
    """
    logging.info("Calculating missing data percentage.")
    missing_data_list = {
        column: {
            'percentage': percentage} for column,
        percentage in zip(
            data.columns,
            data.isna().sum() /
            len(data) *
            100)}

    return missing_data_list


if __name__ == "__main__":

    test_data = pd.read_csv(ingested_data_path)
    test_data.drop(['corporation', 'exited'], axis=1, inplace=True)

    print("Model predictions:", model_predictions(test_data), end='\n\n')

    print("Summary statistics")
    print(json.dumps(dataframe_summary(test_data), indent=4), end='\n\n')

    print("Missing percentage")
    print(json.dumps(missing_data_percentage(test_data), indent=4), end='\n\n')

    print("Execution time")
    print(json.dumps(execution_time(), indent=4), end='\n\n')

    print("Outdated Packages")
    dependencies = outdated_packages_list()

