#!/usr/bin/env python3
"""
Enhanced eDNA Clustering & Real Classification Pipeline (v3.2 - Pro UI)
----------------------------------------------------------------------
This script analyzes an unknown eDNA sample from a FASTA file.
This version features a completely redesigned, fully responsive HTML report
and now includes training artifacts (confusion matrix, feature importance)
if they are available in the same directory.
"""

import argparse
import base64
import pathlib
import joblib
from io import BytesIO
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import SeqIO
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, pairwise_distances_argmin_min

# ============================================================
# -------------------- HELPERS -------------------------------
# ============================================================

def to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding in HTML."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def image_to_base64(filepath):
    """Reads an image file and converts it to a base64 string if it exists."""
    try:
        with open(filepath, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"⚠️  Info: Training artifact '{filepath.name}' not found. It will not be included in the report.")
        return ""

def compute_gc_content(seq):
    g = seq.count("G")
    c = seq.count("C")
    return 100 * (g + c) / len(seq) if len(seq) > 0 else 0

# ============================================================
# -------------------- CLASSIFIER (No Changes) ---------------
# ============================================================

def load_trained_model():
    script_dir = pathlib.Path(__file__).parent.resolve()
    model_path = script_dir / "trained_model.joblib"
    vectorizer_path = script_dir / "vectorizer.joblib"
    print(f"🔍 Loading pre-trained model from '{model_path}'...")
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("✅ Model and vectorizer loaded successfully.")
        return model, vectorizer
    except FileNotFoundError:
        print(f"🔴 ERROR: Model or vectorizer not found!")
        print(f"Please run the training pipeline first.")
        exit(1)

def classify_clusters(rep_sequences, model):
    preds = model.predict(rep_sequences)
    probs = model.predict_proba(rep_sequences)
    results = []
    for prediction, confidence_scores in zip(preds, probs):
        max_confidence = confidence_scores.max()
        if max_confidence < 0.7:
            results.append(f"Unclassified (conf={max_confidence:.2f})")
        else:
            results.append(prediction)
    return results

# ============================================================
# -------------------- CORE ANALYSIS (No Changes) ------------
# ============================================================

def load_sequences(fasta_file):
    print(f"🧬 Loading sequences from '{fasta_file}'...")
    try:
        records = list(SeqIO.parse(fasta_file, "fasta"))
        if not records:
            raise ValueError("No sequences in FASTA.")
        print(f"   Found {len(records)} sequences.")
        return [str(r.seq).upper() for r in records]
    except FileNotFoundError:
        print(f"🔴 ERROR: Input file '{fasta_file}' not found.")
        exit(1)

def perform_analysis(seqs, k, algo, vectorizer):
    print("🔬 Performing core analysis...")
    X = vectorizer.transform(seqs)
    if algo == "kmeans":
        model = KMeans(n_clusters=min(k, X.shape[0]), n_init='auto', random_state=42)
    else:
        model = DBSCAN(eps=0.5, min_samples=3)
    clusters = model.fit_predict(X)
    score = silhouette_score(X, clusters) if len(set(clusters)) > 1 else -1
    print(f"   Clustering complete. Found {len(set(clusters))} clusters.")
    rep_seqs_indices = []
    for lbl in np.unique(clusters):
        if lbl == -1: continue
        idx_in_cluster = np.where(clusters == lbl)[0]
        centroid = np.asarray(X[idx_in_cluster].mean(axis=0))
        closest, _ = pairwise_distances_argmin_min(centroid, X[idx_in_cluster])
        rep_seqs_indices.append(idx_in_cluster[closest[0]])
    pca = PCA(n_components=2, random_state=42).fit_transform(X.toarray())
    return dict(model=model, clusters=clusters, silhouette=score, X=X, PCA=pca, rep_seqs_indices=rep_seqs_indices)

# ============================================================
# -------------------- VISUALIZATIONS (No Changes) -----------
# ============================================================
def make_plots(results, seqs, max_k=10):
    df = pd.DataFrame({"pca1": results["PCA"][:, 0], "pca2": results["PCA"][:, 1], "cluster": results["clusters"]})
    plots = {}
    fig_ab, ax_ab = plt.subplots(figsize=(8, 6)); sns.countplot(x="cluster", data=df, ax=ax_ab, palette="viridis", hue="cluster", legend=False); ax_ab.set_title("Cluster Abundance"); plots["abundance"] = to_base64(fig_ab)
    fig_pca, ax_pca = plt.subplots(figsize=(8, 6)); sns.scatterplot(x="pca1", y="pca2", hue="cluster", data=df, ax=ax_pca, palette="viridis", s=50, alpha=0.8); ax_pca.set_title("PCA of Clusters"); plots["pca"] = to_base64(fig_pca)
    if results["silhouette"] > -1:
        fig_sil, ax_sil = plt.subplots(figsize=(8, 6)); sil = silhouette_samples(results["X"], results["clusters"]); y_lower = 10
        for i in sorted(np.unique(results["clusters"])):
            sil_vals = sil[results["clusters"] == i]; sil_vals.sort(); y_upper = y_lower + len(sil_vals)
            ax_sil.fill_betweenx(np.arange(y_lower, y_upper), 0, sil_vals, alpha=0.7); y_lower = y_upper + 10
        ax_sil.axvline(x=results["silhouette"], color="red", linestyle="--"); ax_sil.set_title("Silhouette Analysis"); ax_sil.set_xlabel("Silhouette coefficient"); ax_sil.set_ylabel("Cluster label"); plots["silhouette"] = to_base64(fig_sil)
    if isinstance(results["model"], KMeans):
        distortions = []; X = results["X"]; k_range = range(2, min(max_k, X.shape[0]))
        if k_range:
            for kk in k_range: km = KMeans(n_clusters=kk, n_init='auto', random_state=42).fit(X); distortions.append(km.inertia_)
            fig_el, ax_el = plt.subplots(figsize=(8, 6)); ax_el.plot(k_range, distortions, "o-"); ax_el.set_title("Elbow Plot for Optimal k"); ax_el.set_xlabel("Number of Clusters (k)"); ax_el.set_ylabel("Distortion"); plots["elbow"] = to_base64(fig_el)
    lengths = [len(s) for s in seqs]; fig_len, ax_len = plt.subplots(figsize=(8, 6)); sns.histplot(lengths, bins=30, kde=True, color="steelblue", ax=ax_len); ax_len.set_title("Sequence Length Distribution"); ax_len.set_xlabel("Length (bp)"); plots["length_dist"] = to_base64(fig_len)
    gcs = [compute_gc_content(s) for s in seqs]; fig_gc, ax_gc = plt.subplots(figsize=(8, 6)); sns.histplot(gcs, bins=30, kde=True, color="darkgreen", ax=ax_gc); ax_gc.set_title("GC Content Distribution"); ax_gc.set_xlabel("GC Content (%)"); plots["gc_dist"] = to_base64(fig_gc)
    return plots

# ============================================================
# -------------------- HTML REPORT (UPDATED) -----------------
# ============================================================

def write_html(results, plots, classes, rep_seqs, args, nseqs):
    """Generate the final, fully responsive HTML report file."""
    if results['silhouette'] >= 0.5:
        silhouette_status, badge_class = "🟢 Good", "good"
    elif results['silhouette'] >= 0.25:
        silhouette_status, badge_class = "🟡 Moderate", "warn"
    else:
        silhouette_status, badge_class = "🔴 Weak/Noisy", "bad"

    rows = ""
    class_map = {lbl: (classes[i], rep_seqs[i]) for i, lbl in enumerate(sorted(c for c in np.unique(results["clusters"]) if c != -1))}

    for lbl in sorted(np.unique(results["clusters"])):
        size = (results["clusters"] == lbl).sum()
        if lbl == -1:
            rows += f"<tr><td data-label='Cluster ID'>Noise</td><td data-label='Size'>{size}</td><td data-label='Taxon'>-</td><td data-label='Status'>-</td><td data-label='Sequence'>-</td></tr>"
        else:
            tax, seq = class_map[lbl]
            novelty = "🧬 Novel?" if "Unclassified" in tax else "✅ Known"
            
            seq_preview_desktop = (seq[:35] + "...") if len(seq) > 35 else seq
            seq_preview_mobile = seq[:4] + "..."

            safe_seq = seq.replace("'", "\\'")
            rows += f"""
            <tr>
                <td data-label="Cluster ID">{lbl}</td>
                <td data-label="Size">{size}</td>
                <td data-label="Predicted Taxon">{tax}</td>
                <td data-label="Status">{novelty}</td>
                <td data-label="Representative Sequence" class="seq-cell">
                    <div class="seq-preview">
                        <span class="desktop-only"><code>{seq_preview_desktop}</code></span>
                        <span class="mobile-only"><code>{seq_preview_mobile}</code></span>
                    </div>
                    <button class="copy-btn" onclick="copySequence(this, '{safe_seq}')">Copy Full</button>
                </td>
            </tr>
            """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>eDNA Biodiversity Report</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            :root {{
                --primary-color: #2c3e50; --secondary-color: #3498db;
                --bg-color: #f8f9fa; --card-bg: #ffffff;
                --text-color: #333; --light-text: #777;
                --border-color: #dee2e6; --shadow: 0 4px 8px rgba(0,0,0,0.07);
            }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 10px; background-color: var(--bg-color); color: var(--text-color); line-height: 1.6; }}
            .container {{ max-width: 1200px; margin: 20px auto; padding: 0 15px; }}
            h1, h2 {{ color: var(--primary-color); }}
            h1 {{ text-align: center; border-bottom: 2px solid var(--secondary-color); padding-bottom: 15px; margin-bottom: 30px; }}
            h2 {{ border-bottom: 1px solid #eee; padding-bottom: 10px; margin-top: 0; }}
            .card {{ background: var(--card-bg); padding: 25px; margin-bottom: 25px; border-radius: 10px; box-shadow: var(--shadow); border: 1px solid var(--border-color); }}
            
            .grid {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 25px; }}
            .grid > .card {{ flex: 0 1 400px; margin: 0; }}

            .summary-item b {{ color: var(--primary-color); }}
            .badge {{ padding: 6px 12px; border-radius: 15px; color: white; font-weight: bold; font-size: 13px; vertical-align: middle; margin-left: 10px; }}
            .good {{ background: #27ae60; }} .warn {{ background: #f39c12; }} .bad {{ background: #e74c3c; }}
            code {{ background: #e9ecef; color: #495057; padding: 4px 6px; border-radius: 5px; font-family: 'SF Mono', 'Courier New', monospace; font-size: 13px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid var(--border-color); }}
            th {{ background: var(--primary-color); color: white; text-align: center; }}
            td:nth-child(1), td:nth-child(2) {{ text-align: center; }}
            .seq-cell {{ display: flex; justify-content: space-between; align-items: center; gap: 15px; }}
            .mobile-only {{ display: none; }}
            .copy-btn {{ padding: 6px 12px; font-size: 12px; cursor: pointer; border: 1px solid #ccc; background-color: #f0f0f0; border-radius: 5px; transition: all 0.2s ease; white-space: nowrap; }}
            .copy-btn:hover {{ background-color: #d8d8d8; border-color: #bbb; }}
            footer {{ margin-top: 40px; text-align: center; color: var(--light-text); font-size: 12px; }}
            
            @media (max-width: 768px) {{
                body {{ padding: 5px; }}
                .container {{ padding: 0 10px; }}
                h1 {{ font-size: 1.5em; }}
                .card {{ padding: 15px; }}
                .grid > .card {{ flex-basis: 100%; }}
                .desktop-only {{ display: none; }}
                .mobile-only {{ display: inline; }}
                table, thead, tbody, th, td, tr {{ display: block; }}
                thead tr {{ position: absolute; top: -9999px; left: -9999px; }}
                tr {{ border: 1px solid var(--border-color); border-radius: 8px; margin-bottom: 15px; }}
                td {{ border: none; border-bottom: 1px solid #eee; position: relative; padding-left: 45%; text-align: right; min-height: 38px; }}
                td:before {{ content: attr(data-label); position: absolute; left: 10px; width: 40%; padding-right: 10px; white-space: nowrap; font-weight: bold; text-align: left; }}
                td.seq-cell {{ padding-top: 8px; padding-bottom: 8px; }}
            }}
        </style>
        <script>
            function copySequence(button, sequence) {{
                if (navigator.clipboard && window.isSecureContext) {{ navigator.clipboard.writeText(sequence); }}
                else {{ const ta = document.createElement('textarea'); ta.value = sequence; ta.style.position = 'absolute'; ta.style.left = '-9999px'; document.body.appendChild(ta); ta.select(); try {{ document.execCommand('copy'); }} catch (err) {{ console.error('Copy failed', err); }} document.body.removeChild(ta); }}
                const originalText = button.innerHTML; button.innerHTML = 'Copied!'; button.style.backgroundColor = '#a0e9a0';
                setTimeout(() => {{ button.innerHTML = originalText; button.style.backgroundColor = ''; }}, 2000);
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>🌍 eDNA Biodiversity Report</h1>
            <div class="card">
                <h2>📄 Run Summary</h2>
                <p class="summary-item"><b>Input File:</b> <code>{args.input}</code></p>
                <p class="summary-item"><b>Total Sequences Analyzed:</b> {nseqs}</p>
                <p class="summary-item"><b>Clustering Algorithm:</b> {args.algo.upper()}</p>
                <p class="summary-item"><b>Clustering Quality:</b> {results['silhouette']:.3f} 
                    <span class="badge {badge_class}">{silhouette_status}</span>
                </p>
            </div>
            <div class="grid">
                <div class="card"><h2>📊 Cluster Abundance</h2><img style="width:100%" src="data:image/png;base64,{plots.get('abundance','')}"/></div>
                <div class="card"><h2>📈 PCA of Clusters</h2><img style="width:100%" src="data:image/png;base64,{plots.get('pca','')}"/></div>
                
                {f'''<div class="card"><h2><span title="From Training Phase">🎓</span> Model Confusion Matrix</h2><img style="width:100%" src="data:image/png;base64,{plots.get('confusion_matrix')}"/></div>''' if plots.get('confusion_matrix') else ''}
                {f'''<div class="card"><h2><span title="From Training Phase">🔑</span> Top K-mer Features</h2><img style="width:100%" src="data:image/png;base64,{plots.get('feature_importance')}"/></div>''' if plots.get('feature_importance') else ''}

                {'<div class="card"><h2>📉 Silhouette Analysis</h2><img style="width:100%" src="data:image/png;base64,' + plots.get('silhouette','') + '"/></div>' if 'silhouette' in plots else ''}
                {'<div class="card"><h2>📉 Elbow Method</h2><img style="width:100%" src="data:image/png;base64,' + plots.get('elbow','') + '"/></div>' if 'elbow' in plots else ''}
                <div class="card"><h2>📏 Sequence Lengths</h2><img style="width:100%" src="data:image/png;base64,{plots['length_dist']}"/></div>
                <div class="card"><h2>🧪 GC Content</h2><img style="width:100%" src="data:image/png;base64,{plots['gc_dist']}"/></div>
            </div>
            <div class="card">
                <h2>🧬 Cluster Details & Classification</h2>
                <table>
                    <thead><tr><th>Cluster ID</th><th>Size</th><th>Predicted Taxon</th><th>Status</th><th>Representative Sequence</th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            <footer><p>Report generated by the Enhanced eDNA Pipeline.</p></footer>
        </div>
    </body>
    </html>
    """
    with open("index.html", "w") as f:
        f.write(html)
    print("✅ Saved responsive report -> index.html")

# ============================================================
# -------------------- MAIN ----------------------------------
# ============================================================

def main():
    p = argparse.ArgumentParser(description="eDNA Clustering & Classification Pipeline")
    p.add_argument("input", help="Input FASTA file")
    p.add_argument("-k", type=int, default=10, help="Number of clusters for K-Means")
    p.add_argument("-a", "--algo", choices=["kmeans", "dbscan"], default="kmeans")
    args = p.parse_args()

    classifier_model, kmer_vectorizer = load_trained_model()
    seqs = load_sequences(args.input)
    results = perform_analysis(seqs, args.k, args.algo, kmer_vectorizer)
    
    rep_seqs = [seqs[i] for i in results["rep_seqs_indices"]]
    X_reps = kmer_vectorizer.transform(rep_seqs)
    classes = classify_clusters(X_reps, classifier_model)

    print("🎨 Generating visualizations...")
    plots = make_plots(results, seqs)
    
    print("🖼️  Looking for training artifacts (confusion matrix, feature importance)...")
    script_dir = pathlib.Path(__file__).parent.resolve()
    plots['confusion_matrix'] = image_to_base64(script_dir / "confusion_matrix.png")
    plots['feature_importance'] = image_to_base64(script_dir / "feature_importance.png")

    print("✍️ Writing final HTML report...")
    write_html(results, plots, classes, rep_seqs, args, len(seqs))

    print("\n🎉 Pipeline finished successfully!")

if __name__ == "__main__":
    main()

