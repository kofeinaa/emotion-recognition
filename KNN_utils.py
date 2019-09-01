import pandas as pd
import pywt
import math
from const import sampling, channels, knn_features


def process_eeg_by_samples(filename, window_size, step):
    df = pd.read_csv(filename, delimiter=',', header=None)
    df = preprocess(df)
    ft = windows_by_samples(df, window_size, step)
    return ft


def process_eeg_by_channels(filename, window_size, step):
    df = pd.read_csv(filename, delimiter=',', header=None)
    df = preprocess(df)
    ft = windows_by_channel(df, window_size, step)
    return ft


# Basic data preprocessing for single read df from csv: removes 3 seconds of stimuli, AMR, minmax normalization.
def preprocess(df):
    # drop first 3 seconds (no stimuli)
    df.drop(df.columns[:384], axis=1, inplace=True)

    # doubled method?
    # average mean reference
    df = df.sub(df.mean(axis=1), axis=0)

    # normalize rows
    df = df.transpose()
    df.columns = channels

    for channel in df.columns:
        min = df[channel].min()
        max = df[channel].max()
        df[channel] = df[channel].apply(lambda x: (x - min) / (max - min))

    return df


# Returns tuple of entropy and energy
def calculate_features(coeffs):
    entropy = 0.
    energy = 0.

    for i in coeffs:
        power = i * i if i != 0 else 0.000000000000001
        energy += power

        v = power * math.log(power)
        entropy += v

    return (-1) * entropy, energy


# Treats each value for channel as sample with two features
def windows_by_samples(df, window_size, step):
    rows, _ = df.shape

    if ((rows / sampling) - window_size) % step != 0:
        raise Exception("Cannot part data to specified windows")

    names = ['entropy', 'energy']
    windows = int(((rows / sampling) - window_size) / step + 1)
    samples = pd.DataFrame(columns=names)

    step = step * sampling
    for channel in df.columns:
        start = 0
        end = window_size * sampling

        for i in range(windows):
            values = df[channel][start:end]
            _, d = pywt.dwt(values, 'db4')

            d = d[::2]
            entropy, energy = calculate_features(d)

            index = df.columns.get_loc(channel) * windows + i
            samples.loc[index] = [entropy, energy]
            start += step
            end += step

    return samples


# Treats each row as separate sample with 28 (14*2) features.
def windows_by_channel(df, window_size, step, valence, arousal):
    rows_num, _ = df.shape
    if ((rows_num / sampling) - window_size) % step != 0:
        raise Exception("Cannot part data to specified windows")

    windows = int(((rows_num / sampling) - window_size) / step + 1)
    parted = pd.DataFrame(index=range(windows), columns=knn_features)
    step = step * sampling

    for channel in df.columns:
        start = 0
        end = window_size * sampling

        loc = df.columns.get_loc(channel)
        for i in range(windows):
            values = df[channel][start:end]
            _, d = pywt.dwt(values, 'db4')
            d = d[::2]

            entropy, energy = calculate_features(d)
            parted.iat[i, 2 * loc] = energy
            parted.iat[i, 2 * loc + 1] = entropy

            start += step
            end += step

    parted['valence'] = valence
    parted['arousal'] = arousal
    return parted


# Treats each row as separate sample with 28 (14*2) features.
def windows_by_channel_raw(df, window_size, step, person, exp, valence, arousal):
    rows_num, _ = df.shape
    if ((rows_num / sampling) - window_size) % step != 0:
        raise Exception("Cannot part data to specified windows")

    windows = int(((rows_num / sampling) - window_size) / step + 1)
    parted = pd.DataFrame(index=range(windows), columns=knn_features)
    step = step * sampling

    for channel in df.columns:
        start = 0
        end = window_size * sampling

        loc = df.columns.get_loc(channel)
        for i in range(windows):
            values = df[channel][start:end]
            _, d = pywt.dwt(values, 'db4')
            d = d[::2]

            coeff = pd.DataFrame[{channel: d}]

            parted.append(coeff)
            start += step
            end += step

    parted['valence'] = valence
    parted['arousal'] = arousal
    parted['person'] = person
    parted['experiment'] = exp

    return parted


# Treats each row as separate sample with 28 (14*2) features.
def windows_by_channel(df, window_size, step, valence, arousal):
    rows_num, _ = df.shape
    if ((rows_num / sampling) - window_size) % step != 0:
        raise Exception("Cannot part data to specified windows")

    windows = int(((rows_num / sampling) - window_size) / step + 1)
    parted = pd.DataFrame(index=range(windows), columns=knn_features)
    step = step * sampling

    for channel in df.columns:
        start = 0
        end = window_size * sampling

        loc = df.columns.get_loc(channel)
        for i in range(windows):
            values = df[channel][start:end]
            _, d = pywt.dwt(values, 'db4')
            d = d[::2]

            entropy, energy = calculate_features(d)
            parted.iat[i, 2 * loc] = energy
            parted.iat[i, 2 * loc + 1] = entropy

            start += step
            end += step

    parted['valence'] = valence
    parted['arousal'] = arousal
    return parted


def read_rating(rating):
    df = pd.read_csv(rating, delimiter=',', header=None)
    df.columns = ['valence', 'arousal', 'dominance', 'liking']
    return df

#
# # todo remove
# def assign_class_name(valence, arousal):
#     # classes V/A: HH 1, HL 2, LH 3, LL 4
#     if valence >= 4.5:
#         if arousal >= 4.5:
#             return 'HH'
#         else:
#             return 'HL'
#     elif arousal >= 4.5:
#         return 'LH'
#     else:
#         return 'LL'
#
#
# # todo remove
# def assign_class_id(valence, arousal):
#     # classes V/A: HH 1, HL 2, LH 3, LL 4
#     if valence >= 4.5:
#         if arousal >= 4.5:
#             return 1
#         else:
#             return 2
#     elif arousal >= 4.5:
#         return 3
#     else:
#         return 4
