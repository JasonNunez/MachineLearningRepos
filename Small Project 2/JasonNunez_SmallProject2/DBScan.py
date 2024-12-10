import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('wine_no_label.csv')

X = data[['Alcohol', 'Malic.acid']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

eps_value = 0.15400
min_samples_value = 3

dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value, metric='euclidean')
y_db = dbscan.fit_predict(X_scaled)

n_clusters = len(set(y_db)) - (1 if -1 in y_db else 0)
n_noise = list(y_db).count(-1)

print(f'\nEstimated number of clusters: {n_clusters}')
print(f'Estimated number of noise points: {n_noise}')


def print_cluster_sizes(labels, algorithm='DBSCAN'):
    cluster_series = pd.Series(labels)
    cluster_sizes = cluster_series.value_counts().sort_index()
    print(f'\nCluster sizes for {algorithm}:')
    for cluster_label, size in cluster_sizes.items():
        if cluster_label == -1:
            print(f"Noise: {size} samples")
        else:
            print(f"Cluster {cluster_label + 1}: {size} samples")


print_cluster_sizes(y_db, algorithm='DBSCAN')

core_samples_mask = np.zeros_like(y_db, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True


def plot_clusters_with_core(X, labels, core_mask, eps, min_samples):
    unique_labels = set(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    plt.figure(figsize=(12, 8))

    for label in unique_labels:
        if label == -1:
            # Noise
            color = 'k'
            label_name = 'Noise'
            marker = 'x'
        else:
            color = colors(label)
            label_name = f'Cluster {label + 1}'
            marker = 'o'

        # Plot non-core points (border points)
        non_core = (labels == label) & (~core_mask)
        if np.any(non_core):
            plt.scatter(
                X[non_core, 0],
                X[non_core, 1],
                c=[color],
                marker=marker,
                edgecolor='black',
                s=30,
                label=f'{label_name} (Border)'
            )

        # Plot core points
        core = (labels == label) & (core_mask)
        if np.any(core):
            plt.scatter(
                X[core, 0],
                X[core, 1],
                c=[color],
                marker='o',
                edgecolor='black',
                s=100,
                label=f'{label_name} (Core)'
            )

    plt.title(f'DBSCAN Clustering with Core Points Highlighted (eps={eps}, min_samples={min_samples})')
    plt.xlabel('Alcohol (standardized)')
    plt.ylabel('Malic.acid (standardized)')
    plt.legend(loc='best', fontsize='small', markerscale=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/dbscan_alcohol_malic_acid.png', dpi=300)
    plt.show()


plot_clusters_with_core(X_scaled, y_db, core_samples_mask, eps_value, min_samples_value)
