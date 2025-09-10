#!/usr/bin/env python3
# sih_gpu_safe.py
import argparse
import logging
import sys
import traceback

import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO

# CPU fallback imports
from sklearn.feature_extraction.text import HashingVectorizer as SKHashingVectorizer
from sklearn.cluster import KMeans as SKKMeans
from sklearn.decomposition import PCA as SKPCA

# Try GPU imports (optional)
USE_GPU = True
try:
    import cudf
    import cupy as cp
    import cuml
    from cuml.feature_extraction.text import HashingVectorizer as CUHashingVectorizer
    from cuml.cluster import KMeans as CUKMeans
    from cuml.decomposition import PCA as CUPCA
except Exception as e:
    USE_GPU = False
    # we will log below

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_arg_parser():
    parser = argparse.ArgumentParser(description='GPU-Accelerated biodiversity clustering for eDNA datasets.')
    parser.add_argument('input_file', help='Path to the input FASTA file.')
    parser.add_argument('-k', '--num_clusters', type=int, default=50, help='Number of clusters to find.')
    parser.add_argument('--n_features', type=int, default=2**16, help='Hashing vectorizer feature count (reduce to lower memory).')
    parser.add_argument('--ngram_min', type=int, default=3)
    parser.add_argument('--ngram_max', type=int, default=6)
    parser.add_argument('--no_gpu', action='store_true', help='Force CPU-only fallback.')
    return parser

def load_sequences(fasta_file):
    logging.info("Loading sequences from file...")
    sequences_cpu = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]
    if not sequences_cpu:
        logging.error("No sequences found in the file.")
        return None
    logging.info(f"Loaded {len(sequences_cpu)} sequences.")
    return sequences_cpu

# ---------- GPU pipeline
def gpu_pipeline(sequences_cpu, k, n_features, ngram_min, ngram_max):
    logging.info("Attempting GPU pipeline...")
    try:
        # Move to cudf Series
        sequences_gpu = cudf.Series(sequences_cpu)
        logging.info("Sequences moved to cudf Series (GPU).")

        # Vectorize on GPU
        logging.info("Vectorizing on GPU with cuML HashingVectorizer...")
        vectorizer = CUHashingVectorizer(analyzer='char', ngram_range=(ngram_min, ngram_max), n_features=n_features)
        X_gpu = vectorizer.transform(sequences_gpu)  # likely sparse/cupy-backed

        # Convert to dense float32 if needed (watch memory)
        try:
            # If X_gpu supports .astype, ensure float32
            X_gpu = X_gpu.astype('float32')
        except Exception:
            # Try converting via cupy if sparse to dense is small
            logging.info("Could not astype in place; attempting cupy conversion if small.")
            X_gpu = X_gpu.todense().astype('float32')

        logging.info("Clustering on GPU with cuML KMeans...")
        kmeans = CUKMeans(n_clusters=k, random_state=42, n_init=10)
        clusters_gpu = kmeans.fit_predict(X_gpu).astype('int32')

        logging.info("Running PCA on GPU for 2 components...")
        pca = CUPCA(n_components=2, random_state=42)
        Xp_gpu = pca.fit_transform(X_gpu)

        # Return as GPU arrays/series
        return {
            'pca': Xp_gpu,      # usually cupy.ndarray or cudf-backed
            'clusters': clusters_gpu
        }

    except Exception as e:
        logging.error("GPU pipeline failed with exception:\n" + "".join(traceback.format_exception_only(type(e), e)))
        logging.debug(traceback.format_exc())
        raise

# ---------- CPU pipeline (fallback)
def cpu_pipeline(sequences_cpu, k, n_features, ngram_min, ngram_max):
    logging.info("Running CPU pipeline (scikit-learn)...")
    vec = SKHashingVectorizer(analyzer='char', ngram_range=(ngram_min, ngram_max), n_features=n_features)
    X_cpu = vec.transform(sequences_cpu)  # sparse matrix

    logging.info("Clustering on CPU with scikit-learn KMeans...")
    kmeans = SKKMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cpu)

    logging.info("Running PCA on CPU for plotting...")
    pca = SKPCA(n_components=2, random_state=42)
    # convert to dense if necessary (for small datasets it's fine)
    X_dense = X_cpu.toarray()
    Xp = pca.fit_transform(X_dense)

    return {'pca': Xp, 'clusters': clusters}

def generate_plots(pca_arr, clusters_arr, out_prefix='gpu_fallback'):
    logging.info("Generating plots (on CPU matplotlib)...")
    # pca_arr: n x 2, clusters_arr: n
    import pandas as pd
    df = pd.DataFrame({'pca1': pca_arr[:,0], 'pca2': pca_arr[:,1], 'cluster': clusters_arr})
    plt.figure(figsize=(12,7))
    sns.countplot(x='cluster', data=df, order=df['cluster'].value_counts().index)
    plt.title('Cluster Abundance')
    plt.savefig(f'{out_prefix}_abundance.png')
    logging.info(f"Saved {out_prefix}_abundance.png")

    plt.figure(figsize=(12,9))
    sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=df, s=50, alpha=0.7, legend='full')
    plt.title('Cluster Visualization (PCA)')
    plt.savefig(f'{out_prefix}_pca.png')
    logging.info(f"Saved {out_prefix}_pca.png")

if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    logging.info(f"Starting analysis on {args.input_file} with k={args.num_clusters}")

    sequences = load_sequences(args.input_file)
    if not sequences:
        sys.exit(1)

    result = None
    # try GPU unless user requested --no_gpu or GPU libs missing
    if not args.no_gpu and USE_GPU:
        try:
            result_gpu = gpu_pipeline(sequences, args.num_clusters, args.n_features, args.ngram_min, args.ngram_max)
            # Try to export to CPU arrays for plotting safely
            try:
                # If pca is cudf/cupy, convert to numpy
                pca_gpu = result_gpu['pca']
                clusters_gpu = result_gpu['clusters']
                # Attempt to convert with .get() or .to_array() or .to_pandas
                import numpy as np
                if hasattr(pca_gpu, 'to_array'):
                    pca_cpu = pca_gpu.to_array()
                elif hasattr(pca_gpu, 'to_pandas'):
                    pca_cpu = pca_gpu.to_pandas().values
                else:
                    # assume cupy
                    pca_cpu = pca_gpu.get() if hasattr(pca_gpu, 'get') else np.array(pca_gpu)
                if hasattr(clusters_gpu, 'to_array'):
                    clusters_cpu = clusters_gpu.to_array().astype(int)
                elif hasattr(clusters_gpu, 'to_pandas'):
                    clusters_cpu = clusters_gpu.to_pandas().astype(int).values
                else:
                    clusters_cpu = clusters_gpu.get() if hasattr(clusters_gpu, 'get') else np.array(clusters_gpu)
                result = {'pca': pca_cpu, 'clusters': clusters_cpu}
                logging.info("GPU run successful; converted results to CPU for plotting.")
            except Exception as e:
                logging.warning("Could not convert GPU outputs to CPU arrays cleanly: " + repr(e))
                raise
        except Exception:
            logging.warning("GPU path failed — falling back to CPU.")
    else:
        logging.info("Skipping GPU (either --no_gpu or GPU libraries not available).")

    if result is None:
        # CPU fallback
        result = cpu_pipeline(sequences, args.num_clusters, args.n_features, args.ngram_min, args.ngram_max)

    generate_plots(result['pca'], result['clusters'], out_prefix='sih_result')
    logging.info("Analysis complete.")
