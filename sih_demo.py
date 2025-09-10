# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

# --- 1. DATA LOADING AND PROCESSING ---

# Define the input file name
FASTA_FILE = "data/marine_environmental_DNA_18S(1-50).fasta"

print(f"Loading sequences from {FASTA_FILE}...")
# Use Biopython to read the FASTA file
# This creates a list of DNA sequences as strings
try:
    sequences = [str(record.seq) for record in SeqIO.parse(FASTA_FILE, "fasta")]
    print(f"Successfully loaded {len(sequences)} sequences.")
except FileNotFoundError:
    print(f"Error: The file '{FASTA_FILE}' was not found.")
    print("Please make sure your data file is in the same directory and named correctly.")
    exit()

print("Converting sequences to numerical data (k-mer counting)...")
# Use CountVectorizer to convert DNA sequences into numerical vectors
# We'll count the frequency of 6-letter DNA "words" (k-mers)
vectorizer = CountVectorizer(analyzer='char', ngram_range=(6, 6))
X = vectorizer.fit_transform(sequences)

# --- 2. AI CLUSTERING AND DIMENSIONALITY REDUCTION ---

# Define the number of clusters to find
# This is a hyperparameter you can tune for your demo
NUM_CLUSTERS = 10

print(f"Clustering sequences into {NUM_CLUSTERS} groups using K-Means...")
# Apply the K-Means algorithm to group the sequences
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
# The 'clusters' variable now holds the cluster ID for each sequence
clusters = kmeans.fit_predict(X)

print("Reducing data to 2 dimensions for plotting using PCA...")
# Use PCA to reduce the complex data down to 2 dimensions for visualization
pca = PCA(n_components=2, random_state=42)
# X_pca now contains the (x, y) coordinates for each sequence
X_pca = pca.fit_transform(X.toarray())

# --- 3. VISUALIZATION ---

# Create a pandas DataFrame to hold our results for easy plotting
df = pd.DataFrame({
    'pca1': X_pca[:, 0],
    'pca2': X_pca[:, 1],
    'cluster': clusters
})

# Plot 1: Abundance of each cluster
print("Creating abundance plot...")
plt.figure(figsize=(12, 7))
sns.countplot(x='cluster', data=df, palette='viridis', order=df['cluster'].value_counts().index)
plt.title('Abundance of Each Discovered Taxonomic Cluster', fontsize=16)
plt.xlabel('Cluster ID', fontsize=12)
plt.ylabel('Number of Sequences', fontsize=12)
plt.savefig('abundance_plot.png') # Save the plot as a file
print("Saved 'abundance_plot.png'")

# Plot 2: Scatter plot of the clusters
print("Creating PCA cluster plot...")
plt.figure(figsize=(12, 9))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=df, palette='viridis', s=50, alpha=0.7, legend='full')
plt.title('AI-Driven Biodiversity Clustering (PCA Visualization)', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend(title='Cluster ID')
plt.savefig('cluster_visualization.png') # Save the plot as a file
print("Saved 'cluster_visualization.png'")

print("\nScript finished! Check for the two .png plot files in your folder.")
plt.show() # Display the plots if running in an interactive environment