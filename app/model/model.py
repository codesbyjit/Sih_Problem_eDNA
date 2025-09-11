# Import necessary libraries
import argparse

# custom
from training.train_classifier import train_dummy_classifier
from main.classify_claster import classify_clusters
from main.load_sequences import load_sequences
from generate.analysis import perform_analysis
from generate.visualization import generate_visualizations

#html part
from gen_html import generate_html_report


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