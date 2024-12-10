import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np

data = pd.read_csv('wine_no_label.csv')

features = ['Alcohol', 'Malic.acid']
X = data[features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

X_train, X_test = train_test_split(X_scaled_df, test_size=0.3, random_state=0)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

cluster_range = [3, 4, 5]

kmeans_models = {}

for k in cluster_range:
    km = KMeans(
        n_clusters=k,
        init='random',
        n_init=10,
        max_iter=30000,
        tol=1e-4,  # Smaller tolerance for better convergence
        random_state=0
    )
    km.fit(X_train)
    kmeans_models[k] = km
    print(f"K-Means clustering completed for k={k}")


def plot_clusters(k, km, X_train, y_km, save_path):
    plt.figure(figsize=(10, 7))

    colors = cm.get_cmap('viridis')(np.linspace(0, 1, k))
    markers = ['s', 'o', 'v', '^', 'D', 'P', 'X']
    labels = [f'Cluster {i + 1}' for i in range(k)]

    for i in range(k):
        plt.scatter(
            X_train.loc[y_km == i, 'Alcohol'],
            X_train.loc[y_km == i, 'Malic.acid'],
            s=50,
            color=colors[i],
            marker=markers[i % len(markers)],
            edgecolor='black',
            label=labels[i]
        )

    cluster_centers = km.cluster_centers_

    plt.scatter(
        cluster_centers[:, 0],  # Alcohol
        cluster_centers[:, 1],  # Malic.acid
        s=250,
        marker='*',
        c='red',
        edgecolor='black',
        label='Centroids'
    )

    plt.title(f'K-Means Clustering on Wine Dataset (k={k})')
    plt.xlabel('Alcohol (Standardized)')
    plt.ylabel('Malic Acid (Standardized)')
    plt.legend(scatterpoints=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_silhouette(k, km, X_train, save_path):

    fig, ax1 = plt.subplots(figsize=(10, 7))

    ax1.set_xlim([-0.1, 1])

    ax1.set_ylim([0, len(X_train) + (k + 1) * 10])

    sample_silhouette_values = silhouette_samples(X_train, km.labels_)
    silhouette_avg = silhouette_score(X_train, km.labels_)
    print(f"For k={k}, the average silhouette score is {silhouette_avg:.4f}")

    y_lower = 10
    for i in range(k):
        ith_cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / k)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i + 1))

        y_lower = y_upper + 10  # 10 for spacing between clusters

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_title(f"Silhouette Plot for k={k}")
    ax1.set_xlabel("Silhouette Coefficient Values")
    ax1.set_ylabel("Cluster Label")

    ax1.set_yticks([])
    ax1.set_xticks(np.arange(-0.1, 1.1, 0.2))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


for k in cluster_range:
    km = kmeans_models[k]
    y_km = km.predict(X_train)

    cluster_plot_path = f'images/kmeans_clusters_k{k}.png'
    plot_clusters(k, km, X_train, y_km, cluster_plot_path)

    silhouette_plot_path = f'images/silhouette_k{k}.png'
    plot_silhouette(k, km, X_train, silhouette_plot_path)
