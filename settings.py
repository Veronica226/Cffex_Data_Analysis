import os

base_dir = os.getcwd()

origin_data_dir = os.path.join(base_dir, "raw_data", "cffex-host-info")
pre_data_dir = os.path.join(base_dir, "raw_data", "pre_data")

output_dir = os.path.join(base_dir,"output_data")
output_cffex_info_dir= os.path.join(output_dir, "cffex-host-info")
plot_data_dir = os.path.join(output_dir,"plot-data")
plot_data_1_dir = os.path.join(output_dir,"plot-data-1")
alarm_data_dir = os.path.join(output_dir,"cffex-host-alarm")
# output_dir= os.path.join(base_dir,"output_data")
plot_dir = os.path.join(output_dir,"kpi-plot")
predict_data_dir =  os.path.join(output_dir,"predicting_data")
metric_figures_dir = os.path.join(output_dir, 'metric_figures')
subplot_data_dir = os.path.join(output_dir,'subplot_data')