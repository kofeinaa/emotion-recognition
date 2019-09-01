import pandas as pd
from sklearn import preprocessing
import pywt
import math

# Constants
SAMPLING = 128
CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']


class UtilsOld:

    @staticmethod
    def process_eeg_by_samples(filename, window_size, step, rating):
        df = pd.read_csv(filename, delimiter=',', header=None)
        df = UtilsOld.preprocess(df)
        ft = UtilsOld.windows_by_samples(df, window_size, step, rating)
        return ft

    @staticmethod
    def process_eeg_by_channels(filename, window_size, step):
        df = pd.read_csv(filename, delimiter=',', header=None)
        df = UtilsOld.preprocess(df)
        ft = UtilsOld.windows_by_channel(df, window_size, step)
        return ft

    @staticmethod
    def preprocess(df):
        # drop first 3 seconds (no stimuli)
        df.drop(df.columns[:384], axis=1, inplace=True)

        # doubled method?
        # average mean reference
        #TODO verify results
        #df = df.sub(df.mean(axis=1), axis=0)

        # normalize rows
        df = df.transpose()
        df.columns = CHANNELS

        for channel in df.columns:
            min = df[channel].min()
            max = df[channel].max()
            df[channel] = df[channel].apply(lambda x: (x - min) / (max - min))
        #
        # x = df.values  # returns a numpy array
        # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        # x_scaled = min_max_scaler.fit_transform(x)
        # df = pd.DataFrame(x_scaled)
        # df.columns = CHANNELS
        return df

    @staticmethod
    def calculate_features(coeffs):
        entropy = 0.
        energy = 0.

        for i in coeffs:
            power = i * i if i != 0 else 0.000000000000001
            energy += power

            v = power * math.log(power)
            entropy += v

        return (-1) * entropy, energy

    @staticmethod
    def windows_by_samples(df, window_size, step, rating):
        if ((len(df[df.columns[0]]) / SAMPLING) - window_size) % step != 0:
            raise Exception("Cannot part data to specified windows")

        names = ['entropy', 'energy', 'valence', 'arousal', 'dominance', 'class']

        windows = int(((len(df[df.columns[0]]) / SAMPLING) - window_size) / step + 1)
        samples = pd.DataFrame(columns=names)

        step = step * SAMPLING

        valence = rating['valence']
        arousal = rating['arousal']
        class_id = UtilsOld.assign_class_id(valence, arousal)

        for channel in df.columns:
            start = 0
            end = window_size * SAMPLING

            for i in range(windows):
                values = df[channel][start:end]
                _, d = pywt.dwt(values, 'db4')

                entropy, energy = UtilsOld.calculate_features(d)

                index = df.columns.get_loc(channel) * windows + i
                samples.loc[index] = [entropy, energy, valence, arousal, rating['dominance'], class_id]
                start += step
                end += step

        return samples

    @staticmethod
    def windows_by_channel(df, window_size, step):
        if ((len(df[df.columns[0]]) / SAMPLING) - window_size) % step != 0:
            raise Exception("Cannot part data to specified windows")

        windows = int(((len(df[df.columns[0]]) / SAMPLING) - window_size) / step + 1)
        parted = pd.DataFrame(index=range(windows), columns=df.columns, dtype=object)
        step = step * SAMPLING

        for channel in df.columns:
            start = 0
            end = window_size * SAMPLING

            for i in range(windows):
                values = df[channel][start:end]
                _, d = pywt.dwt(values, 'db4')

                features = UtilsOld.calculate_features(d)
                parted.iat[i, df.columns.get_loc(channel)] = features

                start += step
                end += step

        return parted

    @staticmethod
    def read_rating(rating):
        df = pd.read_csv(rating, delimiter=',', header=None)
        df.columns = ['valence', 'arousal', 'dominance', 'liking']
        return df

    @staticmethod
    def assign_class_name(valence, arousal):
        # classes V/A: HH 1, HL 2, LH 3, LL 4
        if valence >= 4.5:
            if arousal >= 4.5:
                return 'HH'
            else:
                return 'HL'
        elif arousal >= 4.5:
            return 'LH'
        else:
            return 'LL'

    @staticmethod
    def assign_class_id(valence, arousal):
        # classes V/A: HH 1, HL 2, LH 3, LL 4
        if valence >= 4.5:
            if arousal >= 4.5:
                return 1
            else:
                return 2
        elif arousal >= 4.5:
            return 3
        else:
            return 4


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
    rating = UtilsOld.read_rating(filename_rating)

    for e in range(exp):
        fe = 'exp' + str(e + 1) + 'data_data.csv'
        filename_eeg = dir + fp + fe
        exp_rating = rating.loc[e]
        eeg = UtilsOld.process_eeg_by_samples(filename_eeg, window_size, step, exp_rating)
        rows, columns = eeg.shape
        data_list.append(eeg)

data = pd.concat(data_list, ignore_index=True)
data.to_pickle('./data_for_knn_without_amr.pkl')
