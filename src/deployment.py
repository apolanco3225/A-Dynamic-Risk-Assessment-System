"""
Script for Deploying Machine Learning Model
Author: Arturo Polanco
Date: February 2022
"""
# import necessary packages
import os
import json
import shutil   
import logging


logging.basicConfig(level=logging.INFO)


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

# model path
trained_model_path = os.path.join(config['output_model_path'], "trained_model.pckl") 
# score metadata path
score_artifact_path = os.path.join(config['output_folder_path'], "latest_score.txt") 
# ingested metadata path
output_folder_path = config['output_folder_path']
ingested_metadata_path = os.path.join(output_folder_path, 'ingested_files.txt')

# production path 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


####################function for deployment
def deploy_model():
    # Copy the following files into the deployment directory:
    # 1. Model Pickle file 
    # 2. Meta data latest_score.txt  
    # 3. Meta data ingestfiles.txt 

    logging.info("Deploying model in production.")
    
    shutil.copy(
        trained_model_path,
        prod_deployment_path
    )
    
    shutil.copy(
        score_artifact_path,
        prod_deployment_path
    )

    shutil.copy(
        ingested_metadata_path,
        prod_deployment_path
    )




        

if __name__ == '__main__':
    deploy_model()

