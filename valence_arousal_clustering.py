from sklearn.cluster import SpectralClustering, KMeans, MiniBatchKMeans, AffinityPropagation, MeanShift, \
    estimate_bandwidth
from pandas import DataFrame

import read_save_data as rd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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


def main():
    cluster_data('./data_dwt.pkl')


if __name__ == "__main__":
    main()
