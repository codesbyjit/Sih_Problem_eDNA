import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score


def perform_analysis(sequences, num_clusters, cluster_algo):
    """Runs the full analysis pipeline: vectorizing, clustering, and PCA."""
    print("Converting sequences to numerical data (k-mer counting)...")
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(6, 6))
    X = vectorizer.fit_transform(sequences)

    print(f"Clustering with {cluster_algo}...")
    if cluster_algo == 'kmeans':
        if len(sequences) < num_clusters:
            print(f"Error: More clusters ({num_clusters}) requested than samples ({len(sequences)}).")
            exit()
        model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    elif cluster_algo == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=5) # These parameters may need tuning
    
    clusters = model.fit_predict(X)
    
    unique_labels = np.unique(clusters)
    score = -999
    if len(unique_labels) > 1 and len(unique_labels) < len(sequences):
        print("Calculating clustering quality (Silhouette Score)...")
        score = silhouette_score(X, clusters)
        print(f"Silhouette Score: {score:.3f}")
    
    print("Identifying representative sequence for each cluster...")
    representative_sequences = []
    for label in unique_labels:
        if label == -1: continue # Skip noise points in DBSCAN
        indices = np.where(clusters == label)[0]
        cluster_vectors = X[indices]
        centroid = np.asarray(cluster_vectors.mean(axis=0))
        closest_index, _ = pairwise_distances_argmin_min(centroid, cluster_vectors)
        representative_sequences.append(sequences[indices[closest_index[0]]])

    print("Reducing data to 2 dimensions for plotting using PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X.toarray())

    results = {
        'model': model, 'pca_components': X_pca, 'clusters': clusters,
        'vectorizer': vectorizer, 'X_vectors': X, 'silhouette_score': score,
        'representative_sequences': representative_sequences
    }
    return results