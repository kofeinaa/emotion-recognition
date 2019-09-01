import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from const import knn_features, knn_downsampled_pkl

df = pd.read_pickle(knn_downsampled_pkl)

# binary classification
df['valence_high'] = np.where(df['valence'] > 4.5, 1, 0)
df['arousal_high'] = np.where(df['arousal'] > 4.5, 1, 0)

# 3-class classification
valence_labels3 = pd.DataFrame()
valence_labels3['0'] = list(map(lambda x: 1 if x < 0.33 else 0, df['valence']))
valence_labels3['1'] = list(map(lambda x: 1 if 0.33 <= x < 0.66 else 0, df['valence']))
valence_labels3['2'] = list(map(lambda x: 1 if x >= 0.66 else 0, df['valence']))

arousal_labels3 = pd.DataFrame()
arousal_labels3['0'] = list(map(lambda x: 1 if x < 0.33 else 0, df['arousal']))
arousal_labels3['1'] = list(map(lambda x: 1 if 0.33 <= x < 0.66 else 0, df['arousal']))
arousal_labels3['2'] = list(map(lambda x: 1 if x >= 0.66 else 0, df['arousal']))

features = df[knn_features]
labels_valence = df['valence_high']
labels_arousal = df['arousal_high']

cf_v = KNeighborsClassifier(n_neighbors=3)
cf_a = KNeighborsClassifier(n_neighbors=3)

cf_v3 = KNeighborsClassifier(n_neighbors=3)
cf_a3 = KNeighborsClassifier(n_neighbors=3)

# binary classification
c = KFold(shuffle=True, n_splits=10)
valence_scores = cross_val_score(cf_v, features, labels_valence, cv=c, scoring='accuracy')
arousal_scores = cross_val_score(cf_a, features, labels_arousal, cv=c, scoring='accuracy')

# 3-class classification
c3 = KFold(shuffle=True, n_splits=10)
valence_scores3 = cross_val_score(cf_v3, features, arousal_labels3, cv=c3, scoring='accuracy')
arousal_scores3 = cross_val_score(cf_a3, features, arousal_labels3, cv=c3, scoring='accuracy')

print(valence_scores)
print(arousal_scores)

print(valence_scores3)
print(arousal_scores3)
