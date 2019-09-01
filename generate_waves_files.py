import pathlib
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt

import read_save_data as rd
from const import channels, separated_data, alpha_coefficients_by_channel, beta_coefficients_by_channel, \
    theta_coefficients_by_channel, gamma_coefficients_by_channel, dwt_data


# Generates gamma waves for each channel
def gamma_dwt_by_channel():
    data_list = list()

    data = rd.read_pkl_list(separated_data)

    for i in range(1280):
        print(i)
        dwt = pd.DataFrame(columns=channels)
        for channel in channels:
            values = data[i][channel]

            _, detail_level0 = pywt.dwt(values, 'db4')
            detail_downsampled = detail_level0[::2]

            dwt[channel] = detail_downsampled
        data_list.append(dwt)

    return data_list


# Generates beta waves for each channel
def beta_dwt_by_channel():
    data_list = list()

    data = rd.read_pkl_list(separated_data)

    for i in range(1280):
        print(i)
        dwt = pd.DataFrame(columns=channels)
        for channel in channels:
            values = data[i][channel]

            approximation_level0, _ = pywt.dwt(values, 'db4')
            approximation = approximation_level0[::2]

            _, detail_level1 = pywt.dwt(approximation, 'db4')
            detail_downsampled = detail_level1[::2]

            dwt[channel] = detail_downsampled
        data_list.append(dwt)

    return data_list


# Generates alpha waves for each channel
def alpha_dwt_by_channel():
    data_list = list()

    data = rd.read_pkl_list(separated_data)

    for i in range(1280):
        print(i)
        dwt = pd.DataFrame(columns=channels)
        for channel in channels:
            values = data[i][channel]

            approximation_level0, _ = pywt.dwt(values, 'db4')
            approximation = approximation_level0[::2]

            approximation_level1, _ = pywt.dwt(approximation, 'db4')
            approximation1 = approximation_level1[::2]

            _, detail_level2 = pywt.dwt(approximation1, 'db4')
            detail_downsampled = detail_level2[::2]

            dwt[channel] = detail_downsampled
        data_list.append(dwt)

    return data_list


# Generates theta waves for each channel
def theta_dwt_by_channel():
    data_list = list()

    data = rd.read_pkl_list(separated_data)

    for i in range(1280):
        print(i)
        dwt = pd.DataFrame(columns=channels)
        for channel in channels:
            values = data[i][channel]

            approximation_level0, _ = pywt.dwt(values, 'db4')
            approximation = approximation_level0[::2]

            approximation_level1, _ = pywt.dwt(approximation, 'db4')
            approximation1 = approximation_level1[::2]

            approximation_level2, _ = pywt.dwt(approximation1, 'db4')
            approximation2 = approximation_level2[::2]

            _, detail_level3 = pywt.dwt(approximation2, 'db4')
            detail_downsampled = detail_level3[::2]

            dwt[channel] = detail_downsampled
        data_list.append(dwt)

    return data_list


def visualize_data(file, save_dir):
    data = rd.read_pkl_list(file)
    x = range(len(data[0]['AF3']))

    rating = pkl.load(open("rating.pkl", "rb"))

    for i in range(1280):

        plt.figure(figsize=(80, 20))
        exp = data[i]

        # for channel in channels:
        for channel in ['AF3', 'AF4']:
            waves = exp[channel]
            plt.plot(x, waves, label=channel)

        plt.legend()
        # plt.title("Valence: " + str(valence) + "Arousal: " + str(arousal))
        plt.title("Experiment: " + str(i) + " valence " + str(rating.at[i, 'valence']) + " arousal " + str(
            rating.at[i, 'arousal']))
        # plt.savefig(save_dir + "exp_" + str(i) + ".png", bbox_inches='tight')
        plt.show()


# Save ratings for each dataframe
def save_ratings():
    data = rd.read_pkl_list(dwt_data)

    rating = pd.DataFrame(columns=['valence', 'arousal', 'person', 'experiment'])
    valence = list()
    arousal = list()
    person = list()
    experiment = list()

    for i in range(1280):
        print(i)
        valence.append(data[i]['valence'][0])
        arousal.append(data[i]['arousal'][0])
        person.append(data[i]['person'][0])
        experiment.append(data[i]['experiment'][0])
        # rating.append([valence, arousal, person, experiment])

    rating['valence'] = np.array(valence)
    rating['arousal'] = np.array(arousal)
    rating['person'] = np.array(person)
    rating['experiment'] = np.array(experiment)

    pkl.dump(rating, open("rating.pkl", "wb"))


def main():
    save_ratings()
    # dir = "theta_dir"
    # pathlib.Path(dir).mkdir(exist_ok=True)
    # visualize_data(theta_coefficients_by_channel, dir)
    # pkl.dump(gamma_dwt_by_channel(), open("./gamma_dwt_by_channel.pkl", "wb"))
    # pkl.dump(beta_dwt_by_channel(), open("./beta_dwt_by_channel.pkl", "wb"))
    # pkl.dump(alpha_dwt_by_channel(), open("./alpha_dwt_by_channel.pkl", "wb"))
    # pkl.dump(theta_dwt_by_channel(), open("./theta_dwt_by_channel.pkl", "wb"))


if __name__ == "__main__":
    main()
