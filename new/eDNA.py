#!/usr/bin/env python3
"""
Enhanced eDNA Clustering & Classification Pipeline
--------------------------------------------------
Usage:
    python backend_pipeline.py input.fasta -k 8 -a kmeans
"""

import argparse, base64
from io import BytesIO
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import SeqIO
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score, silhouette_samples, pairwise_distances_argmin_min
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


# ============================================================
# -------------------- HELPERS -------------------------------
# ============================================================

def to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding in HTML."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def compute_gc_content(seq):
    """Compute GC% for one sequence."""
    g = seq.count("G") + seq.count("C")
    return 100 * g / len(seq) if len(seq) > 0 else 0


# ============================================================
# -------------------- CLASSIFIER ----------------------------
# ============================================================

def train_dummy_classifier(vectorizer):
    """Train dummy Naive Bayes classifier (placeholder for SILVA/NCBI DB)."""
    refs = [
        "AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGC",
        "GCTATTACGGCCGCGGCTAACACATGCAAGTCGAACGGTAACAGGAAGAAGCTTGCTTCTTTGCTGACGA",
        "GTCGTAGTGGGGACTAACGGCTCACCTAGCCCGGACACCGGGACACGTGCCGGATGCTGCACCCCAGTGC",
    ]
    labels = ["Fungi", "Protista", "Metazoa"]
    model = make_pipeline(vectorizer, MultinomialNB())
    model.fit(refs, labels)
    return model


def classify_clusters(rep_sequences, model):
    """Predict taxonomy with confidence filter."""
    preds = model.predict(rep_sequences)
    probs = model.predict_proba(rep_sequences)
    out = []
    for p, conf in zip(preds, probs.max(axis=1)):
        if conf < 0.7:
            out.append(f"Unclassified (conf={conf:.2f})")
        else:
            out.append(p)
    return out


# ============================================================
# -------------------- CORE ANALYSIS -------------------------
# ============================================================

def load_sequences(fasta_file):
    """Load sequences from FASTA."""
    records = list(SeqIO.parse(fasta_file, "fasta"))
    if not records:
        raise ValueError("No sequences found in FASTA.")
    return [str(r.seq).upper() for r in records]


def perform_analysis(seqs, k, algo):
    """Vectorize, cluster, reduce dimensions, extract representatives."""
    vect = CountVectorizer(analyzer="char", ngram_range=(6, 6))
    X = vect.fit_transform(seqs)

    if algo == "kmeans":
        model = KMeans(n_clusters=min(k, len(seqs)), n_init=10, random_state=42)
    else:
        model = DBSCAN(eps=0.5, min_samples=3)

    clusters = model.fit_predict(X)
    score = silhouette_score(X, clusters) if len(set(clusters)) > 1 else -1

    rep_seqs = []
    for lbl in np.unique(clusters):
        if lbl == -1:
            continue
        idx = np.where(clusters == lbl)[0]
        centroid = np.asarray(X[idx].mean(axis=0))
        closest, _ = pairwise_distances_argmin_min(centroid, X[idx])
        rep_seqs.append(seqs[idx[closest[0]]])

    pca = PCA(n_components=2, random_state=42).fit_transform(X.toarray())

    return dict(
        model=model,
        vectorizer=vect,
        clusters=clusters,
        silhouette=score,
        X=X,
        PCA=pca,
        rep_seqs=rep_seqs,
    )


# ============================================================
# -------------------- VISUALIZATIONS ------------------------
# ============================================================

def make_plots(results, seqs, max_k=10):
    """Generate base64 plots for embedding in report."""
    df = pd.DataFrame({
        "pca1": results["PCA"][:, 0],
        "pca2": results["PCA"][:, 1],
        "cluster": results["clusters"]
    })
    plots = {}

    # PCA + Abundance
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.countplot(x="cluster", data=df, ax=ax[0], palette="viridis")
    sns.scatterplot(x="pca1", y="pca2", hue="cluster", data=df, ax=ax[1],
                    palette="viridis", s=50, alpha=0.8)
    plots["dashboard"] = to_base64(fig)

    # Silhouette Plot
    if results["silhouette"] > -1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sil = silhouette_samples(results["X"], results["clusters"])
        y_lower = 10
        for i, lbl in enumerate(np.unique(results["clusters"])):
            sil_vals = sil[results["clusters"] == lbl]
            sil_vals.sort()
            y_upper = y_lower + len(sil_vals)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, sil_vals, alpha=0.7)
            y_lower = y_upper + 10
        ax.axvline(x=results["silhouette"], color="red", linestyle="--")
        ax.set_title("Silhouette Analysis")
        plots["silhouette"] = to_base64(fig)

    # Elbow Method (for kmeans)
    if isinstance(results["model"], KMeans):
        distortions = []
        X = results["X"]
        for kk in range(2, min(max_k, X.shape[0]) + 1):
            km = KMeans(n_clusters=kk, n_init=5, random_state=42).fit(X)
            distortions.append(km.inertia_)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(range(2, len(distortions) + 2), distortions, "o-")
        ax.set_title("Elbow Plot (k-min selection)")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Distortion (Inertia)")
        plots["elbow"] = to_base64(fig)

    # Sequence Length Distribution
    lengths = [len(s) for s in seqs]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(lengths, bins=20, kde=True, color="steelblue", ax=ax)
    ax.set_title("Sequence Length Distribution")
    ax.set_xlabel("Length (bp)")
    plots["length_dist"] = to_base64(fig)

    # GC Content Distribution
    gcs = [compute_gc_content(s) for s in seqs]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(gcs, bins=20, kde=True, color="darkgreen", ax=ax)
    ax.set_title("GC Content Distribution")
    ax.set_xlabel("GC%")
    plots["gc_dist"] = to_base64(fig)

    return plots


# ============================================================
# -------------------- REPORT GENERATION ---------------------
# ============================================================

def write_html(results, plots, classes, args, nseqs):
    silhouette_status = (
        "🟢 Good" if results["silhouette"] >= 0.5 else
        "🟡 Moderate" if results["silhouette"] >= 0.3 else
        "🔴 Weak"
    )

    rows, idx = "", 0
    for lbl in np.unique(results["clusters"]):
        size = (results["clusters"] == lbl).sum()
        if lbl == -1:
            rows += f"<tr><td>Noisy/Outliers</td><td>{size}</td><td>-</td><td>-</td></tr>"
        else:
            tax = classes[idx]
            novelty = "🧬 Novel?" if "Unclassified" in tax else "✅ Known"
            seq_preview = (results["rep_seqs"][idx][:50] + "...") if results["rep_seqs"][idx] else "-"
            rows += f"<tr><td>{lbl}</td><td>{size}</td><td>{tax}</td><td>{novelty}</td><td><code>{seq_preview}</code></td></tr>"
            idx += 1

    html = f"""
    <html>
    <head>
        <title>eDNA Biodiversity Report</title>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 10px; background: #f9f9fb; }}
            h1 {{ color: #2c3e50; text-align:center; }}
            .card {{ background: white; padding: 15px; margin: 10px 0;
                     border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 14px; }}
            th, td {{ border: 1px solid #ddd; padding: 6px; text-align: center; }}
            th {{ background: #2c3e50; color: white; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 15px; }}
            .badge {{ padding: 4px 8px; border-radius: 6px; color: white; font-weight: bold; font-size:12px; }}
            .good {{ background: #27ae60; }}
            .warn {{ background: #f39c12; }}
            .bad {{ background: #e74c3c; }}
            code {{ background:#f4f4f4; padding:2px 4px; border-radius:4px; }}
        </style>
    </head>
    <body>
        <h1>🌍 eDNA Biodiversity Report</h1>

        <div class="card">
            <h2>📄 Run Summary</h2>
            <p><b>File:</b> {args.input}</p>
            <p><b>Total Sequences:</b> {nseqs}</p>
            <p><b>Algorithm:</b> {args.algo.upper()}</p>
            <p><b>Silhouette Score:</b> {results['silhouette']:.3f} 
                <span class="badge {'good' if results['silhouette']>=0.5 else 'warn' if results['silhouette']>=0.3 else 'bad'}">{silhouette_status}</span>
            </p>
        </div>

        <div class="grid">
            <div class="card"><h2>📊 Cluster Abundance & PCA</h2>
                <img style="width:100%" src="data:image/png;base64,{plots['dashboard']}"/>
            </div>
            <div class="card"><h2>📈 Silhouette Analysis</h2>
                <img style="width:100%" src="data:image/png;base64,{plots.get('silhouette','')}"/>
            </div>
            <div class="card"><h2>📏 Sequence Lengths</h2>
                <img style="width:100%" src="data:image/png;base64,{plots['length_dist']}"/>
            </div>
            <div class="card"><h2>🧪 GC Content</h2>
                <img style="width:100%" src="data:image/png;base64,{plots['gc_dist']}"/>
            </div>
            <div class="card"><h2>📉 Elbow Method (k-min)</h2>
                <img style="width:100%" src="data:image/png;base64,{plots.get('elbow','')}"/>
            </div>
        </div>

        <div class="card">
            <h2>🧬 Cluster Details</h2>
            <table>
                <tr><th>Cluster ID</th><th>Size</th><th>Predicted Taxonomy</th><th>Status</th><th>Representative Seq (preview)</th></tr>
                {rows}
            </table>
        </div>

        <footer style="margin-top:20px; text-align:center; color:#777; font-size:12px;">
            <p>Generated by eDNA Pipeline | SIH Hackathon Prototype</p>
        </footer>
    </body>
    </html>
    """

    with open("final_report.html", "w") as f:
        f.write(html)
    print("✅ Saved responsive report -> final_report.html")


# ============================================================
# -------------------- MAIN ---------------------------------
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Input FASTA file")
    p.add_argument("-k", type=int, default=5, help="Clusters for kmeans")
    p.add_argument("-a", "--algo", choices=["kmeans", "dbscan"], default="kmeans")
    args = p.parse_args()

    seqs = load_sequences(args.input)
    results = perform_analysis(seqs, args.k, args.algo)
    model = train_dummy_classifier(results["vectorizer"])
    classes = classify_clusters(results["rep_seqs"], model)
    plots = make_plots(results, seqs)
    write_html(results, plots, classes, args, len(seqs))
    print("🎉 Done!")


if __name__ == "__main__":
    main()
