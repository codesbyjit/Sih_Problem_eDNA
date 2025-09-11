# Import necessary libraries
import argparse
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score, silhouette_samples, pairwise_distances_argmin_min
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.cm as cm

# --- 1. CLASSIFICATION MODULE ---

def train_dummy_classifier(vectorizer):
    """
    Creates and trains a dummy Naive Bayes classifier.
    In a real application, this data would come from a large, labeled reference database (e.g., SILVA).
    """
    print("Training a dummy taxonomic classifier on a reference set...")
    # Dummy training data: representative sequences and their known labels
    ref_sequences = [
        "AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGC", # Fungi
        "GCTATTACGGCCGCGGCTAACACATGCAAGTCGAACGGTAACAGGAAGAAGCTTGCTTCTTTGCTGACGA", # Protista
        "GTCGTAGTGGGGACTAACGGCTCACCTAGCCCGGACACCGGGACACGTGCCGGATGCTGCACCCCAGTGC", # Metazoa (Animal)
        "GCCGCGTGCAGGAATGGACGGAGGGCCGCACCTGGACCAGATGGCCCGCGGGATCAGCCCGGATGGGGAC", # Fungi
        "CGCCGCCGTCCGGTTAATTCGAGTAACCGGCGCGAGCGGCGCACCGGGCGGAGCGGCGAGCGGCGCGGAG", # Protista
        "AGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGA", # Metazoa (Animal)
        "ABCDEFGHAGADBKJGHUWBDHSHUSHDBHGUIDHOEGEUIFEHOIHUOHUOEHOIHGUOHOHUOHOIEHU", #IDK
    ]
    ref_labels = ["Fungi", "Protista", "Metazoa", "Fungi", "Protista", "Metazoa", "IDK"]
    
    # Create a scikit-learn pipeline to ensure vectorization is consistent
    model = make_pipeline(vectorizer, MultinomialNB())
    model.fit(ref_sequences, ref_labels)
    print("Classifier training complete.")
    return model

def classify_clusters(representative_sequences, model):
    """Predicts taxonomy using the trained model and its confidence."""
    print("Classifying representative sequences...")
    
    # Get predictions and the probabilities for each class
    predictions = model.predict(representative_sequences)
    probabilities = model.predict_proba(representative_sequences)
    
    # Get the confidence score for the winning class
    confidence_scores = probabilities.max(axis=1)
    
    # If confidence is low, it might be a novel or unclassified organism
    final_predictions = []
    for pred, conf in zip(predictions, confidence_scores):
        if conf < 0.75: # Confidence threshold can be tuned
            final_predictions.append(f"Unclassified (Confidence: {conf:.2f})")
        else:
            final_predictions.append(pred)
            
    return final_predictions

# --- 2. CORE ANALYSIS & VISUALIZATION ---

def load_sequences(fasta_file):
    """Loads sequences from a FASTA file."""
    print(f"Loading sequences from {fasta_file}...")
    try:
        records = list(SeqIO.parse(fasta_file, "fasta"))
        sequences = [str(rec.seq) for rec in records]
        print(f"Successfully loaded {len(sequences)} sequences.")
        return sequences
    except FileNotFoundError:
        print(f"Error: The file '{fasta_file}' was not found.")
        exit()

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

def generate_html_report(results, plots, classifications, args, num_sequences):
    """Generates the final self-contained HTML report with all sections."""
    print("Generating final HTML report...")
    cluster_details_html = ""
    unique_labels = sorted(np.unique(results['clusters']))
    rep_seq_idx = 0
    for label in unique_labels:
        if label == -1:
            cluster_size = (results['clusters'] == -1).sum()
            cluster_details_html += f"<tr><td>Noise (DBSCAN)</td><td>{cluster_size}</td><td>-</td><td>-</td></tr>"
            continue
        
        seq = results['representative_sequences'][rep_seq_idx]
        pred = classifications[rep_seq_idx]
        display_seq = seq[:60] + '...' if len(seq) > 60 else seq
        cluster_size = (results['clusters'] == label).sum()
        cluster_details_html += f"<tr><td>{label}</td><td>{cluster_size}</td><td><strong>{pred}</strong></td><td><code>{display_seq}</code></td></tr>"
        rep_seq_idx += 1

    html_content = f"""
    <html><head><title>Biodiversity Analysis Report</title><style>body{{font-family:sans-serif;margin:2em;background-color:#f4f4f9;}}h1,h2,h3{{color:#2c3e50;}}details{{background-color:#ffffff;border:1px solid #ddd;padding:1.5em;margin-top:1em;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);}}summary{{font-weight:bold;cursor:pointer;font-size:1.2em;}}table{{width:100%;border-collapse:collapse;margin-top:1em;}}th,td{{padding:12px;border:1px solid #ddd;text-align:left;}}th{{background-color:#4CAF50;color:white;}}tr:nth-child(even){{background-color:#f2f2f2;}}img{{max-width:100%;height:auto;border:1px solid #ccc;margin-top:1em;border-radius:5px;}}code{{background-color:#e8e8e8;padding:2px 5px;border-radius:3px;}}</style></head>
    <body>
        <h1>AI-Driven eDNA Biodiversity Report</h1>
        <details open><summary>Run Summary</summary>
            <p><strong>Input File:</strong> <code>{args.input_file}</code></p>
            <p><strong>Total Sequences Analyzed:</strong> {num_sequences}</p>
            <p><strong>Clustering Algorithm Used:</strong> <code>{args.cluster_algo}</code></p>
            <p><strong>Overall Clustering Quality (Silhouette Score):</strong> <strong>{results['silhouette_score']:.3f}</strong> (Range: -1 to 1, higher is better)</p>
        </details>
        <details open><summary>Analysis Dashboard</summary>
            <img src="data:image/png;base64,{plots['dashboard']}" alt="Analysis Dashboard">
        </details>
        <details open><summary>Cluster Details & Taxonomic Classification</summary>
            <p>Each cluster's representative sequence is classified using a pre-trained model. Low confidence predictions may indicate novel taxa.</p>
            <table>
                <tr><th>Cluster ID</th><th>Sequences in Cluster</th><th>Predicted Taxonomy</th><th>Representative Sequence (first 60 bases)</th></tr>
                {cluster_details_html}
            </table>
        </details>
        <details><summary>Model Diagnostics & Monitoring</summary>
            <h3>Elbow Method for Optimal K (K-Means)</h3>
            <p>This plot helps identify the optimal number of clusters. The 'elbow' (the point of inflection on the curve) is a good candidate for 'k'.</p>
            <img src="data:image/png;base64,{plots['elbow']}" alt="Elbow Method Plot">
            <h3>Silhouette Analysis for Each Cluster</h3>
            <p>This plot shows how well each cluster is defined. Bars that are wider and extend beyond the red average line indicate well-separated clusters.</p>
            <img src="data:image/png;base64,{plots['silhouette']}" alt="Silhouette Plot">
        </details>
        <details><summary>Distinguishing K-mer Heatmap</summary>
            <p>This heatmap shows the frequency of the top 50 most variable k-mers across each cluster. Bright areas indicate a k-mer is highly frequent in a cluster, helping to distinguish it from others.</p>
            <img src="data:image/png;base64,{plots['heatmap']}" alt="K-mer Heatmap">
        </details>
    </body></html>
    """
    with open("final_report.html", "w") as f: f.write(html_content)
    print("Saved final report to 'final_report_new.html'")

# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced AI-driven biodiversity clustering for eDNA datasets.')
    parser.add_argument('input_file', help='Path to the input FASTA file.')
    parser.add_argument('-k', '--num_clusters', type=int, default=10, help='Number of clusters to find (for K-Means).')
    parser.add_argument('-a', '--cluster_algo', type=str, default='kmeans', choices=['kmeans', 'dbscan'], help='Clustering algorithm to use.')
    args = parser.parse_args()

    sequences = load_sequences(args.input_file)
    analysis_results = perform_analysis(sequences, args.num_clusters, args.cluster_algo)
    
    # The vectorizer created during analysis is used to ensure the classifier sees data in the same way
    trained_model = train_dummy_classifier(analysis_results['vectorizer'])
    classifications = classify_clusters(analysis_results['representative_sequences'], trained_model)
    
    plots = generate_visualizations(analysis_results, sequences)
    
    generate_html_report(analysis_results, plots, classifications, args, len(sequences))
    
    print("\nAnalysis finished successfully!")