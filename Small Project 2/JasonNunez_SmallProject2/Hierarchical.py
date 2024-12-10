import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('wine_no_label.csv')

X = data.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

row_clusters = linkage(X_scaled, method='complete', metric='euclidean')

cluster_assignments = fcluster(row_clusters, t=3, criterion='maxclust')

data['Cluster'] = cluster_assignments

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

data['PC1'] = X_pca[:, 0]
data['PC2'] = X_pca[:, 1]

colors = {1: 'red', 2: 'green', 3: 'blue'}

plt.figure(figsize=(10, 7))
for cluster in range(1, 4):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'],
                c=colors[cluster], label=f'Cluster {cluster}', edgecolor='black', s=50)

plt.title('Hierarchical Clustering (3 Clusters) on Wine Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('images/hierarchical.png', dpi=300)
plt.show()
