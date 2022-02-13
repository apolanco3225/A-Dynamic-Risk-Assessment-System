"""
This script ingests multiple data sources
and returns a unique csv file
Arturo Polanco Lozano
February 2022
"""
# import necessary packages
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging


logging.basicConfig(level=logging.INFO)

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


#############Function for data ingestion
def merge_multiple_dataframe():
    # output master dataframe
    output_dataframe = pd.DataFrame()

    ingested_files = []

    file_list = os.listdir(input_folder_path)
    logging.info(f"[INFO] Starting merging from folder {input_folder_path}.")
    print(65*"-")

    for file in file_list:
        logging.info(f"Reading {file}")
        file_path = os.path.join(input_folder_path, file)
        try:
            data = pd.read_csv(file_path)
        except:
            logging.info(f"The file {file} has not a valid format.")
            print(65*"-")
            continue

        output_dataframe = pd.concat([output_dataframe, data], ignore_index=True)
        ingested_files.append(file)

        logging.info(f"File {file} successfully merged.")
        print(65*"-")
    
    logging.info("Dropping duplicated values.")
    output_dataframe.drop_duplicates(inplace=True)

    logging.info("Saving ingestion metadata.")
    ingested_metadata_path = os.path.join(output_folder_path, 'ingested_files.txt')
    
    with open(ingested_metadata_path, "w") as file:
        file.write(
            f"Ingestion date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        file.write("\n".join(ingested_files))

    logging.info("Saving ingested data in csv file.")
    output_data_path = os.path.join(output_folder_path, 'final_data.csv')
    output_dataframe.to_csv(output_data_path, index=False)
    logging.info("Done!")
    
            



if __name__ == '__main__':
    merge_multiple_dataframe()

