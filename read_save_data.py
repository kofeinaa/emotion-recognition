import pickle as pkl

import pandas as pd
import pywt

# fixme move to one file(?)
persons = 32
exp = 40
selected_channel_data_dir = '/home/wigdis/emotiv/DEAP/selected_channel_data/'
names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'person',
         'experiment', 'valence', 'arousal']
sampling = 128
channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

# files
pkl_data = './data.pkl'
pkl_dropper = './data_dropped.pkl'
pkl_processed = './data_processed.pkl'
pkl_separated_df = './data_separated_df.pkl'
pkl_separated_df_without_amr = './data_separated_no_amr.pkl'


# Saves data for each exp/person as list to pickle
def save_separated_df_to_pickle(filename_to_save, process_function):
    data_list = []

    for p in range(persons):
        fp = 'person' + str(p + 1)
        filename_rating = selected_channel_data_dir + fp + '_rating.csv'
        rating = pd.read_csv(filename_rating, delimiter=',', header=None, usecols=[0, 1])
        print(p)

        for e in range(exp):
            fe = 'exp' + str(e + 1) + 'data_data.csv'
            filename_eeg = selected_channel_data_dir + fp + fe
            exp_rating = rating.loc[e]

            df = pd.read_csv(filename_eeg, delimiter=',', header=None)

            df = process_function(df)
            df[len(df.columns)] = p
            df[len(df.columns)] = e
            df[len(df.columns)] = exp_rating[0]
            df[len(df.columns)] = exp_rating[1]
            df.columns = names
            data_list.append(df)

    pkl.dump(data_list, open(filename_to_save, 'wb'))
    print('Separated dataframes for each participant/experiment saved')


# Drops first 3 seconds (no stimuli) and then min-max normalization
def process(df):
    # Pre-processing part
    # drop first 3 seconds (no stimuli)
    df.drop(df.columns[:384], axis=1, inplace=True)
    df = df.transpose()
    for channel in range(len(df.columns)):
        min = df[channel].min()
        max = df[channel].max()
        df[channel] = df[channel].apply(lambda x: (x - min) / (max - min))
    return df


# Drops first 3 seconds (no stimuli), use Average Mean Reference method and then min-max normalization
def process_amr(df):
    # Pre-processing part
    # drop first 3 seconds (no stimuli)
    df.drop(df.columns[:384], axis=1, inplace=True)
    # doubled method?
    # average mean reference
    df = df.sub(df.mean(axis=1), axis=0)
    df = df.transpose()
    for channel in range(len(df.columns)):
        min = df[channel].min()
        max = df[channel].max()
        df[channel] = df[channel].apply(lambda x: (x - min) / (max - min))
    return df


def read_pkl_df(filename):
    return pd.read_pickle(filename)


def read_pkl_list(filename):
    return pkl.load(open(filename, 'rb'))


# Calculate number of windows for selected window size and step
# If can't fit the data - throws exception
def calculate_windows(df, window_size, step):
    rows_num, _ = df.shape
    if ((rows_num / sampling) - window_size) % step != 0:
        raise Exception("Cannot part data to specified windows")

    return int(((rows_num / sampling) - window_size) / step + 1)


def generate_windowed_data(df, window_size, step):
    windows = calculate_windows(df, window_size, step)

    step = step * sampling
    windowed = pd.DataFrame(columns=channels, index=range(windows))

    start = 0
    end = window_size * sampling

    for i in range(windows):
        for channel in channels:
            values = df[channel][start:end]
            windowed.at[i, channel] = values

        start += step
        end += step

    # rewrite rating
    windowed['person'] = df.iloc[0]['person']
    windowed['experiment'] = df.iloc[0]['experiment']
    windowed['valence'] = df.iloc[0]['valence']
    windowed['arousal'] = df.iloc[0]['arousal']

    windowed.reset_index(drop=True, inplace=True)
    return windowed


# detail coefficients - high frequencies (high pass filter)
# approximations coefficients - low frequencies (low pass filter)
def calculate_dwt(windowed):
    rows_num, _ = windowed.shape
    dwt = pd.DataFrame(columns=channels, index=range(rows_num))

    for i in range(rows_num):
        for channel in channels:
            values = windowed[channel][i]
            _, detail = pywt.dwt(values, 'db4')

            dwt.at[i, channel] = detail[::2]

    # rewrite rating
    dwt['person'] = windowed.iloc[0]['person']
    dwt['experiment'] = windowed.iloc[0]['experiment']
    dwt['valence'] = windowed.iloc[0]['valence']
    dwt['arousal'] = windowed.iloc[0]['arousal']

    return dwt


# beta waves
def calculate_beta(windowed):
    rows_num, _ = windowed.shape
    dwt = pd.DataFrame(columns=channels, index=range(rows_num))

    for i in range(rows_num):
        for channel in channels:
            values = windowed[channel][i]

            approximation_level0, _ = pywt.dwt(values, 'db4')
            approximation = approximation_level0[::2]

            _, detail_level1 = pywt.dwt(approximation, 'db4')
            detail = detail_level1[::2]

            dwt.at[i, channel] = detail

    # rewrite rating
    dwt['person'] = windowed.iloc[0]['person']
    dwt['experiment'] = windowed.iloc[0]['experiment']
    dwt['valence'] = windowed.iloc[0]['valence']
    dwt['arousal'] = windowed.iloc[0]['arousal']

    return dwt


# alpha waves
def calculate_alpha(windowed):
    rows_num, _ = windowed.shape
    dwt = pd.DataFrame(columns=channels, index=range(rows_num))

    for i in range(rows_num):
        for channel in channels:
            values = windowed[channel][i]

            approximation_level0, _ = pywt.dwt(values, 'db4')
            approximation = approximation_level0[::2]

            approximation_level1, _ = pywt.dwt(approximation, 'db4')
            approximation1 = approximation_level1[::2]

            _, detail_level2 = pywt.dwt(approximation1, 'db4')
            detail_downsampled = detail_level2[::2]

            dwt.at[i, channel] = detail_downsampled
    # rewrite rating
    dwt['person'] = windowed.iloc[0]['person']
    dwt['experiment'] = windowed.iloc[0]['experiment']
    dwt['valence'] = windowed.iloc[0]['valence']
    dwt['arousal'] = windowed.iloc[0]['arousal']

    return dwt


# theta waves
def calculate_theta(windowed):
    rows_num, _ = windowed.shape
    dwt = pd.DataFrame(columns=channels, index=range(rows_num))

    for i in range(rows_num):
        for channel in channels:
            values = windowed[channel][i]

            approximation_level0, _ = pywt.dwt(values, 'db4')
            approximation = approximation_level0[::2]

            approximation_level1, _ = pywt.dwt(approximation, 'db4')
            approximation1 = approximation_level1[::2]

            approximation_level2, _ = pywt.dwt(approximation1, 'db4')
            approximation2 = approximation_level2[::2]

            _, detail_level3 = pywt.dwt(approximation2, 'db4')
            detail_downsampled = detail_level3[::2]

            dwt.at[i, channel] = detail_downsampled

    # rewrite rating
    dwt['person'] = windowed.iloc[0]['person']
    dwt['experiment'] = windowed.iloc[0]['experiment']
    dwt['valence'] = windowed.iloc[0]['valence']
    dwt['arousal'] = windowed.iloc[0]['arousal']

    return dwt

def main():
    save_separated_df_to_pickle(pkl_separated_df_without_amr, process)
    # d = read_pkl_list(pkl_separated_df)
    # print(d[0])


if __name__ == "__main__":
    main()
