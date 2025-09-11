import numpy as np


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
    with open("app/output/index.html", "w") as f: f.write(html_content)
    print("Saved final report to 'index.html'")
