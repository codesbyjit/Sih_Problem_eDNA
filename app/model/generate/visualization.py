import base64
from io import BytesIO
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

def generate_visualizations(results, sequences):
    """Generates all plots and returns them as a dictionary of base64 strings."""
    print("Generating all visualizations...")
    df = pd.DataFrame({
        'pca1': results['pca_components'][:, 0],
        'pca2': results['pca_components'][:, 1],
        'cluster': results['clusters']
    })
    X = results['X_vectors']
    clusters = results['clusters']
    plots = {}
    
    # Plot 1: Main Dashboard (Abundance + PCA)
    fig_dash, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig_dash.suptitle('Biodiversity Analysis Dashboard', fontsize=20)
    sns.countplot(ax=ax1, x='cluster', data=df, palette='viridis', hue='cluster', legend=False, order=df['cluster'].value_counts().index)
    ax1.set_title('Cluster Abundance', fontsize=16)
    sns.scatterplot(ax=ax2, x='pca1', y='pca2', hue='cluster', data=df, palette='viridis', s=50, alpha=0.7, legend='full')
    ax2.set_title('Cluster Visualization (PCA)', fontsize=16)
    plots['dashboard'] = to_base64(fig_dash)

    # Plot 2: K-mer Heatmap
    fig_heat, ax_heat = plt.subplots(figsize=(18, 10))
    cluster_means = []
    unique_labels = sorted([l for l in np.unique(results['clusters']) if l != -1])
    for label in unique_labels:
        cluster_means.append(results['X_vectors'][results['clusters'] == label].mean(axis=0).A1)
    
    if cluster_means:
        heatmap_df = pd.DataFrame(cluster_means, index=[f"Cluster {l}" for l in unique_labels])
        top_features = heatmap_df.var().nlargest(min(50, heatmap_df.shape[1])).index
        sns.heatmap(heatmap_df[top_features], cmap='viridis', ax=ax_heat)
    ax_heat.set_title('Top 50 Distinguishing K-mer Frequencies per Cluster', fontsize=16)
    ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0)
    plots['heatmap'] = to_base64(fig_heat)
    
    # --- NEW MONITORING PLOTS ---
    # Plot 3: Per-Cluster Silhouette Plot
    fig_sil, ax_sil = plt.subplots(1, 1, figsize=(10, 7))
    if results['silhouette_score'] != -999:
        n_clusters = len(np.unique(clusters))
        y_lower = 10
        silhouette_avg = results['silhouette_score']
        sample_silhouette_values = silhouette_samples(X, clusters)
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.viridis(float(i) / n_clusters)
            ax_sil.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
            y_lower = y_upper + 10
        ax_sil.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax_sil.set_title("Silhouette Plot for Each Cluster")
    plots['silhouette'] = to_base64(fig_sil)

    # Plot 4: Elbow Plot for K-Means Inertia
    fig_elbow, ax_elbow = plt.subplots(1, 1, figsize=(10, 7))
    inertias = []
    k_range = range(2, min(16, len(sequences)))
    if k_range:
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
            inertias.append(kmeans.inertia_)
        ax_elbow.plot(k_range, inertias, 'bo-')
    ax_elbow.set_title('Elbow Method For Optimal K (K-Means)')
    ax_elbow.set_xlabel('Number of clusters (k)')
    ax_elbow.set_ylabel('Inertia')
    plots['elbow'] = to_base64(fig_elbow)
    
    return plots

def to_base64(figure):
    """Converts a matplotlib figure to a base64 string."""
    buffer = BytesIO()
    figure.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')