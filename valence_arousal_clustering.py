import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans

import read_save_data as rd
from const import energy_power_entropy_mean_st_dev, rating, channels


# Shows data labelling
def cluster_data(filename):
    data = rd.read_pkl_list(filename)
    valence = list()
    arousal = list()

    arousal_column = 'arousal'
    valence_column = 'valence'
    for df in data:
        a = df[arousal_column][0]
        v = df[valence_column][0]

        for index, row in df.iterrows():
            arousal.append(a)
            valence.append(v)

    valence = np.array(valence)
    valence = (valence - 1) / 8

    arousal = np.array(arousal)
    arousal = (arousal - 1) / 8

    valence_arousal_data = {
        valence_column: valence,
        arousal_column: arousal
    }

    df = DataFrame(valence_arousal_data, columns=[valence_column, arousal_column])

    plt.figure(figsize=(12, 12))

    # Kmeans
    clusters = 8
    kmeans = KMeans(n_clusters=clusters).fit_predict(df)
    plt.title("Kmeans")
    plt.scatter(df[valence_column], df[arousal_column], c=kmeans)
    plt.show()

    # MeanShift
    # bandwidth = estimate_bandwidth(df, quantile=0.2, n_samples=500)
    # mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit_predict(df)
    # plt.title("mean shift")
    # plt.scatter(df[valence_column], df[arousal_column], c=mean_shift)
    # plt.show()

    # Spectral clustering
    # spectral = SpectralClustering(n_clusters=clusters).fit_predict(df)
    # plt.title("Spectral clustering")
    # plt.scatter(df[valence_column], df[arousal_column], c=spectral)
    # plt.subplot(111)
    # plt.show()


def visualize_entropy():
    features = pkl.load(open(energy_power_entropy_mean_st_dev, 'rb'))
    features_flat = pd.concat(features, ignore_index=True)

    all_ratings = pkl.load(open(rating, 'rb'))
    print("Data loaded")

    valence = all_ratings['valence']
    valence = [item for item in valence for i in range(29)]
    v = np.array(valence)
    v = (v - 1) / 8

    arousal = all_ratings['arousal']
    arousal = [item for item in arousal for i in range(29)]

    a = np.array(arousal)
    a = (a - 1) / 8

    for channel in channels:
        name = channel + '_entropy'
        entropy = features_flat[name]

        plt.title('Entropy/Valence: ' + channel)
        plt.scatter(entropy, v, marker='.')
        plt.legend()
        plt.show()

        plt.title('Entropy/Arousal: ' + channel)
        plt.scatter(entropy, a, marker='.')
        plt.legend()
        plt.show()


def visualize_energy():
    features = pkl.load(open(energy_power_entropy_mean_st_dev, 'rb'))
    features_flat = pd.concat(features, ignore_index=True)

    all_ratings = pkl.load(open(rating, 'rb'))
    print("Data loaded")

    valence = all_ratings['valence']
    valence = [item for item in valence for i in range(29)]
    v = np.array(valence)
    v = (v - 1) / 8

    arousal = all_ratings['arousal']
    arousal = [item for item in arousal for i in range(29)]

    a = np.array(arousal)
    a = (a - 1) / 8

    for channel in channels:
        name = channel + '_energy'
        energy = features_flat[name]

        plt.title('Energy/Valence: ' + channel)
        plt.scatter(energy, v, marker='.', color=['red'])
        plt.legend()
        plt.show()

        plt.title('Energy/Arousal: ' + channel)
        plt.scatter(energy, a, marker='.', color=['red'])
        plt.legend()
        plt.show()


def visualize_entropy_energy():
    features = pkl.load(open(energy_power_entropy_mean_st_dev, 'rb'))
    features_flat = pd.concat(features, ignore_index=True)
    #
    all_ratings = pkl.load(open(rating, 'rb'))
    print("Data loaded")

    valence = all_ratings['valence']
    valence = [item for item in valence for i in range(29)]
    v = np.array(valence)
    v = (v - 1) / 8

    valence_labels = pd.DataFrame()
    valence_labels['0.2'] = list(map(lambda x: 1 if x < 0.33 else 0, v))
    valence_labels['0.5'] = list(map(lambda x: 1 if 0.33 <= x < 0.66 else 0, v))
    valence_labels['1'] = list(map(lambda x: 1 if x >= 0.66 else 0, v))
    c = valence_labels.idxmax(axis=1)

    for channel in channels:
        energy_name = channel + '_energy'
        entropy_name = channel + '_entropy'
        entropy = features_flat[energy_name]
        energy = features_flat[entropy_name]

        plt.title('Energy/Entropy: ' + channel)
        plt.scatter(energy, entropy, marker='.', color=c)
        plt.legend()
        plt.show()


def main():
    visualize_entropy_energy()


if __name__ == "__main__":
    main()
