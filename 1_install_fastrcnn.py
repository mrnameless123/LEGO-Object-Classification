from __future__ import print_function
import zipfile
import os, sys

base_folder = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(base_folder, 'DataSets', 'Grocery'))
from install_grocery import download_grocery_data
download_grocery_data()

sys.path.append(os.path.join(base_folder, 'DataSets', "PretrainedModels"))
from models_util import download_model_by_name
download_model_by_name("AlexNet")
