"""
Script for running ML Pipeline
Author: Arturo Polanco
Date: February 2022
"""
# import necessary packages
import os
from ingestion import merge_multiple_dataframe
from training import train_model
from scoring import score_model
from deployment import deploy_model
from reporting import score_model
import json
import glob
import logging

with open('config.json','r') as file:
    config = json.load(file) 

input_folder_path = config["input_folder_path"]

prod_deployment_path = config["prod_deployment_path"]
ingested_metadata_path = os.path.join(prod_deployment_path, "ingested_files.txt")
metrics_metadata_path = os.path.join(prod_deployment_path, "latest_score.csv")

logging.basicConfig(level=logging.INFO)


##################Check and read new data
#first, read ingestedfiles.txt

def check_for_csv(file_list):
    filtered_list = []
    for file in file_list:
        try:
            if file[-4:] == ".csv":
                filtered_list.append(file)
        except:
            continue
    return filtered_list

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
def check_for_new_data():
    logging.info("Checking for new data available for ingestion.")
    # read ingested files
    with open(ingested_metadata_path) as file:
        ingested_files = {line.strip('\n') for line in file.readlines()[1:]}
    # read source folder 
    source_files = set(os.listdir(input_folder_path))
    source_files = check_for_csv(source_files)

    print(source_files)

    # check if there are new files 
    files_up_to_date = set(source_files).issubset(ingested_files)

    if files_up_to_date:
        logging.info("There are not new files available for ingestion.")
        return False

    else:
        logging.info("There are new files available for ingestion.")
        return True


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
def check_model_drift():
    
    logging.info('Checking for model drift.')
    with open(metrics_metadata_path, 'r') as file:
        previous_score = float(file.readline().strip())

    merge_multiple_dataframe()
    train_model()
    new_score = score_model()

    return new_score > previous_score  



##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
def main():

    if check_for_new_data() == False:
        return None

    logging.info("Execute Machine Learning Pipeline.")

    drift = check_model_drift()

    if not drift:
        print('Production model performs better.')
        exit()

    deploy_model()
    report()





##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here



##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model

if __name__ == "__main__":
    check_for_new_data()



