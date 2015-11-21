# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
import numpy as np

# 学習データ
data_file = np.loadtxt('CodeIQ_data.txt', delimiter=' ')
data = np.array([[x[0], x[1]] for x in data_file])

kmeans_model = KMeans(n_clusters=3, random_state=10).fit(data)

# 分類先となったラベルを取得する
labels = kmeans_model.labels_

# ラベルとデータ表示する
for label, feature in zip(labels, data):
    print(label, " ".join([str(v) for v in feature]))
