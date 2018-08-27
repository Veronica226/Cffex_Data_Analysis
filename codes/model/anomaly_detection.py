import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_time_series_decomposition_model(input_file_path):
    read_col_list = ['hostname', 'archour', 'cpu_max', 'cpu_min', 'monitor_max', 'monitor_min', 'rt_max', 'rt_min', 'tmp_max', 'tmp_min', 'mem_max', 'mem_min']
    kpi_data = pd.read_csv(input_file_path, usecols=read_col_list, dtype=str)

    return