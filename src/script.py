from diagnostics import outdated_packages_list
import pandas as pd
import os 
import json


with open('config.json','r') as file:
    config = json.load(file) 

# ingested data path
output_folder_path = config['output_folder_path']
ingested_data_path = os.path.join(output_folder_path, 'final_data.csv')

test_data = pd.read_csv(ingested_data_path)
test_data.drop(columns=["corporation", "exited"], inplace=True)
print(test_data.columns)

outdated_modules = outdated_packages_list()

print(type(outdated_modules))
print(outdated_modules)
