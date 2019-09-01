import KNN_utils
import pandas as pd
from const import knn_downsampled_pkl

# Reads directly from csv files and creates processed data prepared for KNN, based on Liu paper.
dir = '/home/wigdis/emotiv/DEAP/selected_channel_data/'
# detail coefficients - high frequencies (high pass filter)
# approximations coefficients - low frequencies (low pass filter)

persons = 32
exp = 40
window_size = 4
step = 2

names = ['entropy', 'energy', 'valence', 'arousal', 'dominance', 'class']
channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

df = pd.DataFrame(columns=names)
data_list = []

for p in range(persons):
    fp = 'person' + str(p + 1)
    filename_rating = dir + fp + '_rating.csv'
    rating = KNN_utils.read_rating(filename_rating)

    print(p)
    for e in range(exp):
        fe = 'exp' + str(e + 1) + 'data_data.csv'
        filename_eeg = dir + fp + fe
        exp_rating = rating.loc[e]
        df = pd.read_csv(filename_eeg, delimiter=',', header=None)
        df = KNN_utils.preprocess(df)
        df = KNN_utils.windows_by_channel(df, window_size, step, exp_rating['valence'], exp_rating['arousal'])
        data_list.append(df)

data = pd.concat(data_list, ignore_index=True)
data.to_pickle(knn_downsampled_pkl)
