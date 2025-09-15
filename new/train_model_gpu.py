#!/usr/bin/env python3
"""
train_model_gpu.py (v3.0 - RAPIDS GPU Acceleration)
---------------------------------------------------
This script trains a robust, supervised classifier for eDNA taxonomy
classification, accelerated using NVIDIA RAPIDS on the GPU.

It leverages cuDF and cuML to handle large datasets efficiently,
preventing system freezes and dramatically speeding up training.
"""
import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from Bio import SeqIO
# ### GPU CHANGE ###: Import RAPIDS libraries
import cudf # GPU DataFrame library, replacement for pandas
import cupy as cp # GPU array library, used by cuML
from cuml.ensemble import RandomForestClassifier # GPU RandomForest
from cuml.feature_extraction.text import CountVectorizer # GPU Vectorizer
from cuml.model_selection import train_test_split # GPU train/test split
from cuml.over_sampling import SMOTE # GPU SMOTE

# ### GPU CHANGE ###: Import scikit-learn metrics separately.
# These will run on the CPU after we pull data from the GPU.
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


# ============================================================
# -------------------- HELPER FUNCTIONS ----------------------
# ============================================================
# Note: Helper functions are mostly unchanged, but will now receive
# cuDF DataFrames instead of pandas DataFrames.

def load_sequences(fasta_file):
    """Loads sequences from a FASTA file into a dictionary."""
    print(f"🧬 Loading sequences from '{fasta_file}'...")
    try:
        # This part remains on the CPU as it's I/O bound
        return {record.id: str(record.seq).upper() for record in SeqIO.parse(fasta_file, "fasta")}
    except FileNotFoundError:
        print(f"🔴 ERROR: FASTA file not found at '{fasta_file}'.")
        exit(1)


def load_labels(csv_file):
    """Loads taxonomic labels from a CSV file directly into a GPU DataFrame."""
    print(f"🏷️  Loading labels from '{csv_file}' into GPU memory...")
    try:
        # ### GPU CHANGE ###: Use cudf.read_csv to load data directly onto the GPU
        return cudf.read_csv(csv_file)
    except FileNotFoundError:
        print(f"🔴 ERROR: CSV labels file not found at '{csv_file}'.")
        exit(1)


def plot_confusion_matrix(y_true_gpu, y_pred_gpu, labels, output_path):
    """Generates and saves a heatmap of the confusion matrix."""
    # ### GPU CHANGE ###: Move data from GPU (cuPy array) to CPU (NumPy array) for plotting
    y_true_cpu = y_true_gpu.to_numpy()
    y_pred_cpu = y_pred_gpu.to_numpy()
    
    cm = confusion_matrix(y_true_cpu, y_pred_cpu, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"   - Confusion matrix saved to '{output_path}'")


def plot_feature_importance(model, vectorizer, output_path, top_n=20):
    """Plots the most important k-mers used by the model for classification."""
    # ### GPU CHANGE ###: Get feature importances (already a NumPy array) and names
    importances = model.feature_importances_
    feature_names = vectorizer.get_feature_names_out()
    
    # Use pandas for CPU-side plotting preparation
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances.tolist()})
    top_features = feature_importance_df.sort_values(by='importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
    plt.title(f'Top {top_n} Most Important K-mers', fontsize=16)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('K-mer', fontsize=12)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"   - Feature importance plot saved to '{output_path}'")


# ============================================================
# -------------------- MAIN PIPELINE -------------------------
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GPU-Accelerated eDNA Classifier Training Pipeline")
    parser.add_argument("--fasta", required=True, help="Input FASTA file containing reference sequences.")
    parser.add_argument("--labels", required=True, help="CSV file mapping sequence IDs to taxonomy labels.")
    parser.add_argument("--k", type=int, default=6, help="The size of k-mers to use as features (e.g., 6 for hexamers).")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of the data to hold out for testing.")
    args = parser.parse_args()

    # --- [ STEP 1/8 ] LOAD AND PREPARE DATA ---
    print("\n[ STEP 1/8 ] Loading and Preparing Data (on GPU)...")
    seq_dict = load_sequences(args.fasta) # CPU dict
    labels_df = load_labels(args.labels) # GPU DataFrame

    data = labels_df[labels_df["id"].isin(seq_dict.keys())].copy()
    # ### GPU CHANGE ###: Mapping happens on CPU then result is moved to a GPU Series
    data["sequence"] = cudf.Series(data["id"].to_pandas().map(seq_dict))
    data.dropna(subset=['sequence', 'taxonomy'], inplace=True)
    
    if data.empty:
        print("🔴 ERROR: No matching sequences found between the FASTA and labels file. Exiting.")
        exit(1)
        
    print(f"✅ Found {len(data)} labeled sequences for training on GPU.")

    # --- [ STEP 2/8 ] HANDLE LOW-COUNT CLASSES ---
    print("\n[ STEP 2/8 ] Handling Low-Count Taxonomic Classes...")
    label_counts = data['taxonomy'].value_counts()
    # ### GPU CHANGE ###: Convert cuDF Series to pandas for indexing logic
    labels_to_remove = label_counts[label_counts < 2].index.to_pandas().tolist()
    
    if labels_to_remove:
        print(f"   - Found {len(labels_to_remove)} classes with only 1 member. These will be removed for robust training.")
        print(f"   - Removing: {', '.join(labels_to_remove[:5])}{'...' if len(labels_to_remove) > 5 else ''}")
        data = data[~data['taxonomy'].isin(labels_to_remove)]
        print(f"✅ Data size after filtering: {len(data)} sequences.")
    else:
        print("✅ All classes have at least 2 members. No filtering needed.")
        
    # ### GPU CHANGE ###: Convert labels to categorical integers for cuML
    data['taxonomy'] = data['taxonomy'].astype('category')
    y_cat_codes = data['taxonomy'].cat.codes
    # Create a mapping from codes back to original labels for later
    code_to_label = {i: cat for i, cat in enumerate(data['taxonomy'].cat.categories)}


    # --- [ STEP 3/8 ] K-MER FEATURE EXTRACTION ---
    print("\n[ STEP 3/8 ] Converting DNA Sequences to K-mer Vectors (on GPU)...")
    # ### GPU CHANGE ###: Use cuML's CountVectorizer
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(args.k, args.k))
    X = vectorizer.fit_transform(data["sequence"])
    y = y_cat_codes # Use the integer codes for training
    print(f"✅ Sequences vectorized into a GPU matrix with shape {X.shape}.")

    # --- [ STEP 4/8 ] TRAIN-TEST SPLIT ---
    print("\n[ STEP 4/8 ] Splitting Data into Training and Testing Sets (on GPU)...")
    # ### GPU CHANGE ###: Use cuML's train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"   - Training set size before balancing: {X_train.shape[0]} samples")
    print(f"   - Testing set size:  {X_test.shape[0]} samples")

    # --- [ STEP 5/8 ] BALANCE TRAINING DATA WITH ADAPTIVE SMOTE ---
    print("\n[ STEP 5/8 ] Balancing minority classes in training data using SMOTE (on GPU)...")
    
    min_class_count = y_train.value_counts().min()
    safe_k_neighbors = max(1, min_class_count - 1)
    
    print(f"   - Adapting SMOTE's k_neighbors to {safe_k_neighbors} based on smallest class size ({min_class_count}).")
    # ### GPU CHANGE ###: Use cuML's SMOTE
    smote = SMOTE(random_state=42, k_neighbors=safe_k_neighbors)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"✅ Training set size after SMOTE balancing: {X_train_resampled.shape[0]} samples")
    print("   - Class distribution in new training set:")
    print(y_train_resampled.value_counts().head())

    # --- [ STEP 6/8 ] HYPERPARAMETER TUNING ---
    print("\n[ STEP 6/8 ] Searching for Best Model Parameters (Hyperparameter Tuning on GPU)...")
    param_grid = {
        'n_estimators': [100, 200, 400], # Can often afford more estimators on GPU
        'max_depth': [20, 30, 40], # Deeper trees are faster on GPU
        'min_samples_split': [2, 5]
    }
    # ### GPU CHANGE ###: Use cuML RandomForestClassifier
    # cuML RF has slightly different parameters (e.g., `split_criterion`)
    rf_gpu = RandomForestClassifier(random_state=42)
    # NOTE: GridSearchCV is from scikit-learn but can wrap cuML models.
    # It handles the GPU data transfer automatically.
    grid_search = GridSearchCV(estimator=rf_gpu, param_grid=param_grid, cv=3, verbose=2)
    
    grid_search.fit(X_train_resampled, y_train_resampled)

    best_model = grid_search.best_estimator_
    print(f"✅ Best parameters found: {grid_search.best_params_}")

    # --- [ STEP 7/8 ] MODEL EVALUATION ---
    print("\n[ STEP 7/8 ] Evaluating the Best Model on the original (unbalanced) Test Set...")
    y_pred_codes = best_model.predict(X_test)
    
    # ### GPU CHANGE ###: Convert integer codes back to original string labels
    # First, pull the code arrays from GPU to CPU
    y_test_cpu_codes = y_test.to_numpy()
    y_pred_cpu_codes = y_pred_codes.to_numpy()
    
    # Map codes back to labels using list comprehensions (fast on CPU)
    y_test_labels = [code_to_label[code] for code in y_test_cpu_codes]
    y_pred_labels = [code_to_label[code] for code in y_pred_cpu_codes]
    class_labels = sorted(list(code_to_label.values()))

    report = classification_report(y_test_labels, y_pred_labels, digits=3, labels=class_labels, zero_division=0)
    print("Classification Report:\n", report)

    output_dir = Path(".")
    # ### GPU CHANGE ###: We need to pass the original GPU arrays to the plotting function
    # The function itself will handle the conversion to NumPy
    plot_confusion_matrix(y_test_labels, y_pred_labels, class_labels, output_dir / "confusion_matrix.png")
    plot_feature_importance(best_model, vectorizer, output_dir / "feature_importance.png")


    # --- [ STEP 8/8 ] SAVE ARTIFACTS ---
    print("\n[ STEP 8/8 ] Saving Model, Vectorizer, and Metrics...")
    joblib.dump(best_model, output_dir / "trained_model_gpu.joblib")
    joblib.dump(vectorizer, output_dir / "vectorizer_gpu.joblib")
    
    with open(output_dir / "training_metrics_gpu.txt", "w") as f:
        f.write("eDNA Classifier Training Report (GPU-Accelerated)\n================================================\n\n")
        f.write(f"Best Hyperparameters Found:\n{grid_search.best_params_}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        
    print("✅ Training complete!")
    print(f"💾 Model saved to '{output_dir / 'trained_model_gpu.joblib'}'")
    print(f"💾 Vectorizer saved to '{output_dir / 'vectorizer_gpu.joblib'}'")
    print(f"💾 Metrics saved to '{output_dir / 'training_metrics_gpu.txt'}'")

if __name__ == "__main__":
    main()