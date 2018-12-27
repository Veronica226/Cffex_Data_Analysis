import os

base_dir = os.getcwd()

origin_data_dir = os.path.join(base_dir, "raw_data", "cffex-host-info")
origin_alarm_data_dir = os.path.join(base_dir, "raw_data", "cffex-host-alarm")
pre_data_dir = os.path.join(base_dir, "raw_data", "pre_data")

output_dir = os.path.join(base_dir,"output_data")
new_output_dir =os.path.join(base_dir,"new_output_data")
output_cffex_info_dir= os.path.join(output_dir, "cffex-host-info")
cpu_mem_info_dir = os.path.join(new_output_dir, "cffex-host-info-cpu-mem")
plot_data_dir = os.path.join(output_dir,"plot-data")
roc_plot_data_dir = os.path.join(output_dir,"ROC_plot_data")
new_plot_data_dir = os.path.join(output_dir,"new-plot-data")
plot_data_1_dir = os.path.join(output_dir,"plot-data-1")
alarm_data_dir = os.path.join(output_dir,"cffex-host-alarm")
# output_dir= os.path.join(base_dir,"output_data")
plot_dir = os.path.join(output_dir,"kpi-plot")
new_predict_data_dir = os.path.join(output_dir,"new_predicting_data")
predict_data_dir =  os.path.join(output_dir,"predicting_data")
metric_figures_dir = os.path.join(output_dir, 'metric_figures')
history_metric_figures_dir = os.path.join(output_dir, 'history_metric_figures')
subplot_data_dir = os.path.join(output_dir,'subplot_data')
hist_plot_dir = os.path.join(output_dir, 'hist_plot')
cluster_data_dir =  os.path.join(output_dir,"cluster_data")
cluster_plot_dir = os.path.join(output_dir,"cluster_plot")
multiclass_data_dir = os.path.join(output_dir,"multi_classification_data")
kpi_level_model_dir = os.path.join(output_dir, 'kpi_level_model')

host_data_dir = os.path.join(output_dir, 'host_data')

