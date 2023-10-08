import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def find_optimal_k(data, max_k):
    best_k = None
    best_combined_metric = -np.inf  
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data.reshape(-1, 1))  # Reshape data to a column vector.
        
        silhouette = silhouette_score(data.reshape(-1, 1), kmeans.labels_)
        davies_bouldin = davies_bouldin_score(data.reshape(-1, 1), kmeans.labels_)
        inertia = kmeans.inertia_
        
        epsilon = 1e-10  
        combined_metric = silhouette - davies_bouldin + (1 / (inertia + epsilon))
        
        if combined_metric > best_combined_metric:
            best_combined_metric = combined_metric
            best_k = k
    
    return best_k

def find_optimal_cluster(data):
    MAX_CLUSTER = 10
    data_np = np.array(data)
    best_k = find_optimal_k(data_np, MAX_CLUSTER)
    return best_k
