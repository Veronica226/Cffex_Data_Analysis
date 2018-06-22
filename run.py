import sys

from code.preprocessing import data_preprocessing
import os

sys.path.append("..")
BASE_DIR = os.path.abspath('..')

ORIGIN_DATA = os.path.join(BASE_DIR, "raw_data", "cffex-host-info")
PRE_DATA = os.path.join(BASE_DIR, "raw_data", "pre_data")

output_dir= os.path.join(BASE_DIR,"output_data")
origin_dir = ORIGIN_DATA
pre_dir = PRE_DATA
root_dir = os.getcwd()
raw_data_dir = os.path.join(root_dir, 'raw_data').replace('\\', '/')


data_preprocessing.data_preprocessing_process(origin_dir,output_dir)