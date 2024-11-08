from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np

# Standardizing the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Parameters
chunk_size = 50000  # Adjust based on memory
n_clusters_per_chunk = 100  # Number of clusters per chunk
final_n_clusters = 50  # Final number of clusters desired

# Step 1: Cluster each chunk and store centroids
chunk_centroids = []

num_chunks = int(np.ceil(data_scaled.shape[0] / chunk_size))
for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, data_scaled.shape[0])
    chunk_data = data_scaled[start_idx:end_idx]

    # Cluster the chunk
    chunk_clustering = AgglomerativeClustering(n_clusters=n_clusters_per_chunk, affinity='euclidean', linkage='ward')
    chunk_labels = chunk_clustering.fit_predict(chunk_data)

    # Compute centroids of clusters in this chunk
    centroids = []
    for label in np.unique(chunk_labels):
        cluster_points = chunk_data[chunk_labels == label]
        centroids.append(cluster_points.mean(axis=0))  # Use mean as centroid

    chunk_centroids.extend(centroids)

# Convert centroids to a numpy array
chunk_centroids = np.array(chunk_centroids)

# Step 2: Cluster the centroids to get the final clusters
final_clustering = AgglomerativeClustering(n_clusters=final_n_clusters, affinity='euclidean', linkage='ward')
final_labels = final_clustering.fit_predict(chunk_centroids)

# Step 3: Assign each original data point to the nearest final cluster centroid
final_centroids = []
for label in np.unique(final_labels):
    points_in_cluster = chunk_centroids[final_labels == label]
    final_centroids.append(points_in_cluster.mean(axis=0))

final_centroids = np.array(final_centroids)
closest_centroids, _ = pairwise_distances_argmin_min(data_scaled, final_centroids)

print(f"Final number of clusters: {len(np.unique(closest_centroids))}")
print(f"Cluster assignments: {closest_centroids}")
