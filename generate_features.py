import pickle as pkl

import numpy as np
import pandas as pd

import read_save_data as rd
import matplotlib.pyplot as plt

from const import separated_data, dwt_data, windowed_data, separated_no_amr, dwt_data_no_amr, \
    dwt_data_no_amr_normalized, names, channels, encoded_features, beta_data, beta_data_normalized, \
    gamma_coefficients_by_channel, alpha_data, theta_data


# FIXME probably useless after file changes
# log_dir = './graph_encoded3_1'
# Model saving directory
# model_dir = './model'

# Encoder model saving directory
# encoder_dir = './encoder'
# Graph saving directory for encoder - tensorboard
# encoder_log_dir = './encoder_graph'

# Plots directory for visualisation
# plots_dir = './plots/'

# conv_log_dir = './graph_conv_beta4'
# conv_log_dir_encoded = './graph_conv_encoded'


# Creates windowed data with given window_size and step for raw data
def calculate_windowed_for_raw(filename, window_size, step, result):
    df_list = rd.read_pkl_list(filename)
    print('file read')
    windowed = list(map(lambda x: rd.generate_windowed_data(x, window_size, step), df_list))
    print('windowed')
    pkl.dump(windowed, open(result, 'wb'))
    print('Windowed data saved to pickle')


# generated gamma waves (DWT) for all the data
def calculate_dwt_for_all(filename, result):
    df_list = rd.read_pkl_list(filename)
    print('file read')
    windowed = map(lambda x: rd.generate_windowed_data(x, 4, 2), df_list)
    print('windowed')
    dwt = list(map(rd.calculate_dwt, windowed))
    print('dwt')
    pkl.dump(dwt, open(result, 'wb'))
    print('DWT (Gamma) data saved to pickle')


# Generates gamma waves (DWT) for all the data with additional normalization by channel
def calculate_dwt_for_all_normalized(filename, result):
    df_list = rd.read_pkl_list(filename)
    print('file read')
    windowed = map(lambda x: rd.generate_windowed_data(x, 4, 2), df_list)
    print('windowed')
    dwt = list(map(rd.calculate_dwt, windowed))
    print('dwt')
    dwt_normalized = normalize_by_channel(dwt)
    print("normilized")
    pkl.dump(dwt_normalized, open(result, 'wb'))
    print('DWT data saved to pickle')


# Generates beta waves (DWT) for all the data with additional normalization by channel
def calculate_beta_for_all_normalized(filename, result):
    df_list = rd.read_pkl_list(filename)
    print('file read')
    windowed = map(lambda x: rd.generate_windowed_data(x, 4, 2), df_list)
    print('windowed')
    dwt = list(map(rd.calculate_beta, windowed))
    print('beta waves')
    dwt_normalized = normalize_by_channel(dwt)
    print("normilized")
    pkl.dump(dwt_normalized, open(result, 'wb'))
    print('DWT (Beta normalized) data saved to pickle')


# Generates beta waves (DWT) for all the data
def calculate_beta_for_all(filename, result):
    df_list = rd.read_pkl_list(filename)
    print('file read')
    windowed = map(lambda x: rd.generate_windowed_data(x, 4, 2), df_list)
    print('windowed')
    dwt = list(map(rd.calculate_beta, windowed))
    print('beta waves')
    pkl.dump(dwt, open(result, 'wb'))
    print('DWT (Beta) data saved to pickle')


# Generates alpha waves (DWT) for all the data
def calculate_alpha_for_all(filename, result):
    df_list = rd.read_pkl_list(filename)
    print('file read')
    windowed = map(lambda x: rd.generate_windowed_data(x, 8, 2), df_list)
    print('windowed')
    dwt = list(map(rd.calculate_alpha, windowed))
    print('alpha waves')
    pkl.dump(dwt, open(result, 'wb'))
    print('DWT (Alpha) data saved to pickle')


# Generates beta waves (DWT) for all the data
def calculate_theta_for_all(filename, result):
    df_list = rd.read_pkl_list(filename)
    print('file read')
    windowed = map(lambda x: rd.generate_windowed_data(x, 8, 2), df_list)
    print('windowed')
    dwt = list(map(rd.calculate_theta, windowed))
    print('theta waves')
    pkl.dump(dwt, open(result, 'wb'))
    print('DWT (Theta) data saved to pickle')


# Additional normalization by channel
def normalize_by_channel(data_list):
    mins = dict()
    maxs = dict()

    data = pd.concat(data_list, ignore_index=True)

    for channel in channels:
        lst = data[channel].values
        f = [val for sublist in lst for val in sublist]
        max_channel = np.max(f)
        min_channel = np.min(f)
        maxs[channel] = max_channel
        mins[channel] = min_channel

    print("min max found")

    for df in data_list:
        for index, row in df.iterrows():
            for channel in channels:
                min = mins.get(channel)
                max = maxs.get(channel)
                value = np.array(list(map(lambda x: (x - min) / (max - min), row[channel])))
                df.at[index, channel] = value
                print('row done')

    print("data normalized")
    return data_list


# Shows plot for each experiment using [bandname]_coefficients_by_channel file
def show_data():
    data = rd.read_pkl_list(dwt_data)

    x = np.arange(130)
    plt.figure(figsize=(80, 20))

    for exp in range(1280):
        plt.figure(figsize=(80, 20))
        d0p0 = data[exp]

        for i in range(20):
            plt.subplot(2, 10, i + 1)
            for channel in channels:
                y = d0p0[channel][i]
                valence = d0p0['valence'][0]
                arousal = d0p0['arousal'][0]

                plt.plot(x, y, label=channel)
            plt.gca().set_title("Window no. " + str(i))

        plt.legend()
        plt.title("Valence: " + str(valence) + "Arousal: " + str(arousal))
        plt.savefig(plots_dir + str(exp) + "_valence=" + str(valence) + "_arousal=" + str(arousal) + ".png",
                    bbox_inches='tight')
        # plt.show()


# Show results of autoencoder
def zip_data(df_list):
    # df_list = rd.read_pkl_list(dwt_data_no_amr_normalized)
    rows_num = 1280 * 29
    data = list()

    i = 0
    for df in df_list:
        print(i)
        for index, row in df.iterrows():
            value = list(zip(row['AF3'], row['F7'], row['F3'], row['FC5'], row['T7'], row['P7'], row['O1'], row['O2'],
                             row['P8'], row['T8'], row['FC6'], row['F4'], row['F8'], row['AF4']))
            data.append(value)
        i += 1

    data = np.array(data)
    return np.reshape(data, (rows_num * 130, 1, 14))


def calculate_energy(window):
    energy = 0
    for i in window:
        power = i * i if i != 0 else 0.000000000000001
        energy += power
    return energy


def calculate_power(energy, size):
    m = 1.0 / size
    return m * energy


def calculate_minmax_difference(window):
    return max(window) - min(window)


def calculate_features(data):
    print('next df')
    features_names = ['AF3_energy', 'AF3_power', 'AF3_minmax_diff', 'F7_energy', 'F7_power', 'F7_minmax_diff',
                      'F3_energy', 'F3_power', 'F3_minmax_diff', 'FC5_energy', 'FC5_power', 'FC5_minmax_diff',
                      'T7_energy', 'T7_power', 'T7_minmax_diff', 'P7_energy', 'P7_power', 'P7_minmax_diff', 'O1_energy',
                      'O1_power', 'O1_minmax_diff', 'O2_energy', 'O2_power', 'O2_minmax_diff', 'P8_energy', 'P8_power',
                      'P8_minmax_diff', 'T8_energy', 'T8_power', 'T8_minmax_diff', 'FC6_energy', 'FC6_power',
                      'FC6_minmax_diff', 'F4_energy', 'F4_power', 'F4_minmax_diff', 'F8_energy', 'F8_power',
                      'F8_minmax_diff', 'AF4_energy', 'AF4_power', 'AF4_minmax_diff']

    windows = len(data['AF3'])
    features = pd.DataFrame(columns=features_names)

    for channel in channels:

        energy_col = channel + '_' + 'energy'
        power_col = channel + '_' + 'power'
        amp_col = channel + '_' + 'minmax_diff'

        for window_id in range(windows):
            window = data[channel][window_id]

            energy = calculate_energy(window)
            power = calculate_power(energy, len(window))
            minmax = calculate_minmax_difference(window)

            features.at[window_id, energy_col] = energy
            features.at[window_id, power_col] = power
            features.at[window_id, amp_col] = minmax

    return features


def calculate_power_energy_minmax_diff(filename, output):
    data = rd.read_pkl_list(filename)
    features = list(map(lambda x: calculate_features(x), data))
    pkl.dump(features, open(output, 'wb'))


def main():
    calculate_power_energy_minmax_diff(dwt_data_no_amr, './energy_power_minmax_diff_no_amr_gamma.pkl')

    # calculate_alpha_for_all(separated_data, alpha_data)
    # calculate_theta_for_all(separated_data, theta_data)
    # model_raw(windowed_data)
    # model_dwt(dwt_data_no_amr)
    # model_dwt(dwt_data)
    # show_data()
    # convolution_model(dwt_data)
    # calculate_dwt_for_all_normalized(separated_no_amr, './ .pkl')
    # calculate_beta_for_all(separated_data, beta_data)

    # convolution_model_encoded_features(dwt_data, encoded_features)
    # calculate_beta_for_all_normalized(separated_data, beta_data_normalized)


if __name__ == "__main__":
    main()
