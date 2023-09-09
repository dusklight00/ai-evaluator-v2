import numpy as np
from sklearn.cluster import KMeans
from preprocessor.cluster import find_optimal_cluster

def find_num_lines(words):
    y_coords = [ word["y"] for word in words ]
    clusters = find_optimal_cluster(y_coords)
    return clusters

def find_optimal_central_line(words):
    y_coords = np.array([ word["y"] for word in words ])
    num_lines = find_num_lines(words) 

    kmeans = KMeans(n_clusters=num_lines)
    kmeans.fit(y_coords.reshape(-1, 1))  

    cluster_centers = kmeans.cluster_centers_
    cluster_centers = np.sort(cluster_centers, axis=0)

    return list(cluster_centers.reshape(-1))

def find_word_line(word, central_lines):
    y_coord = word["y"]
    print(word, central_lines)
    distances = [ abs(y_coord - line) for line in central_lines ]
    return np.argmin(distances)