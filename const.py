# column names in df
names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'person',
         'experiment', 'valence', 'arousal']
# channels
channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

sampling = 128
persons = 32
exp = 40

# Features used in KNN
knn_features = ['AF3_energy', 'AF3_entropy', 'F7_energy', 'F7_entropy',
                'F3_energy', 'F3_entropy', 'FC5_energy', 'FC5_entropy',
                'T7_energy', 'T7_entropy', 'P7_energy', 'P7_entropy', 'O1_energy',
                'O1_entropy', 'O2_energy', 'O2_entropy', 'P8_energy', 'P8_entropy',
                'T8_energy', 'T8_entropy', 'FC6_energy', 'FC6_entropy',
                'F4_energy', 'F4_entropy', 'F8_energy', 'F8_entropy', 'AF4_energy',
                'AF4_entropy']

# csv data dir - CSV for each experiment/person
selected_channel_data_dir = '/home/wigdis/emotiv/DEAP/selected_channel_data/'

# Data preprocess according to paper:
# for KNN purposes - used gamma waves
knn_downsampled_pkl = './data_for_knn_downsampled.pkl'

# NOTE - all data is kept as pickle files containing dataframes list

# Read data, kept as 1280 separated dataframes for each person/experiment
# with basic pre processing including Average Mean Reference method
separated_data = './data_separated_df.pkl'
# Same as separated, but without Average Mean Reference
separated_no_amr = './data_separated_no_amr.pkl'

# Windowed data (with AMR) - window 4, step 2
windowed_data = './data_windowed.pkl'  # Basic preprocessing

# DWT coefficients  windowed - window 4, step 2
dwt_data = './data_dwt.pkl'

# DWT coefficients calculated based on the data, that wasn't processed with AMR method
dwt_data_no_amr = './data_dwt_no_amr.pkl'

# DWT coefficients calculated based on the data, that wasn't processed with AMR method
# and was normalized additionally using min/max normalization
# with min/max values for each channel from all of the experiments
dwt_data_no_amr_normalized = './data_dwt_no_amr_normalized.pkl'

log_dir = './graph3'
model_dir = './model'

encoder_dir = './encoder'
encoder_log_dir = './encoder_graph'

plots_dir = './plots/'

# Encoded features (3 elements array)
encoded_features = './encoded_features_3.pkl'

# Beta waves - AMR
beta_data = './beta_data.pkl'
beta_data_normalized = './beta_data_normalized.pkl'

#Alpha waves
alpha_data = './alpha_data.pkl'

#Theta waves
theta_data = './theta_data.pkl'


# DWT Coefficients for each wave band grouped by channel, saved in 1280 dataframes (as list of dfs)
gamma_coefficients_by_channel = "./gamma_dwt_by_channel.pkl"
beta_coefficients_by_channel = "./beta_dwt_by_channel.pkl"
alpha_coefficients_by_channel = "./alpha_dwt_by_channel.pkl"
theta_coefficients_by_channel = "./theta_dwt_by_channel.pkl"

rating = './rating.pkl'

energy_power_minmax_minmax_diff_amr_gamma = './energy_power_minmax_diff_no_amr_gamma.pkl'
energy_power_entropy_mean_st_dev = './energy_power_entropy_mean_st_dev.pkl'''
