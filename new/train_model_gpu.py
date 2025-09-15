#!/usr/bin/env python3
"""
train_model_gpu.py (robust) - GPU-accelerated when available, CPU fallback otherwise.

Notes:
 - cuDF/cuML usage is attempted; if not available we fall back to pandas/scikit-learn.
 - cuML's stable releases typically do NOT include SMOTE. We use imblearn.SMOTE as the
   default and optionally use GPU SMOTE if it's available under cuml.over_sampling.
"""
import argparse
import joblib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO

# Try to import RAPIDS (GPU) libs. If unavailable, fall back to CPU equivalents.
USE_GPU = False
try:
    import cudf               # GPU DataFrame
    import cupy as cp         # GPU arrays
    from cuml.ensemble import RandomForestClassifier as cuRandomForest
    # cuML CountVectorizer lives in cuml.feature_extraction.text in some versions
    from cuml.feature_extraction.text import CountVectorizer as cuCountVectorizer
    from cuml.model_selection import train_test_split as cu_train_test_split
    # NOTE: cuml.over_sampling.SMOTE is often not present in stable releases.
    try:
        from cuml.over_sampling import SMOTE as cuSMOTE
        HAVE_CU_SMOTE = True
    except Exception:
        HAVE_CU_SMOTE = False

    USE_GPU = True
    print("✔️ RAPIDS (cuDF/cuML) imports succeeded — GPU path enabled.")
except Exception as e:
    # GPU libs not available — fall back to CPU stacks
    print("ℹ️ RAPIDS GPU libraries not available or import failed:", str(e))
    print("   Falling back to CPU (pandas / scikit-learn).")
    USE_GPU = False

# CPU libraries (always required)
from sklearn.ensemble import RandomForestClassifier as skRandomForest
from sklearn.feature_extraction.text import CountVectorizer as skCountVectorizer
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# imbalanced-learn SMOTE (CPU). We'll use it by default if GPU SMOTE not available.
try:
    from imblearn.over_sampling import SMOTE as skSMOTE
    HAVE_IMBLEARN = True
except Exception:
    HAVE_IMBLEARN = False

# ---------------- Helper functions ----------------

def load_sequences(fasta_file):
    print(f"🧬 Loading sequences from '{fasta_file}'...")
    try:
        return {record.id: str(record.seq).upper() for record in SeqIO.parse(fasta_file, "fasta")}
    except FileNotFoundError:
        print(f"🔴 ERROR: FASTA file not found at '{fasta_file}'.")
        sys.exit(1)


def load_labels(csv_file):
    print(f"🏷️  Loading labels from '{csv_file}'...")
    if USE_GPU:
        try:
            return cudf.read_csv(csv_file)
        except FileNotFoundError:
            print(f"🔴 ERROR: CSV labels file not found at '{csv_file}'.")
            sys.exit(1)
    else:
        try:
            return pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"🔴 ERROR: CSV labels file not found at '{csv_file}'.")
            sys.exit(1)


def ensure_numpy(x):
    """Convert various types (cuDF Series, cuPy arrays, lists) into NumPy arrays for sklearn plotting."""
    if USE_GPU:
        # cuDF Series
        try:
            import cudf
            if isinstance(x, cudf.Series):
                return x.to_numpy()
        except Exception:
            pass
        # cuPy arrays
        try:
            import cupy as cp
            if isinstance(x, cp.ndarray):
                return cp.asnumpy(x)
        except Exception:
            pass
    # If it's already numpy
    if isinstance(x, np.ndarray):
        return x
    # pandas Series or list
    if isinstance(x, (pd.Series, list, tuple)):
        return np.asarray(list(x))
    # fallback: try to convert
    return np.asarray(x)


def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """Generates and saves a heatmap of the confusion matrix.
    Accepts GPU (cuDF / cuPy) or CPU (lists / numpy / pandas)."""
    y_true_np = ensure_numpy(y_true)
    y_pred_np = ensure_numpy(y_pred)

    cm = confusion_matrix(y_true_np, y_pred_np, labels=labels)
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
    """Plots the most important features / k-mers. Works with cuML or sklearn RF."""
    # Get feature names
    try:
        feature_names = vectorizer.get_feature_names_out()
    except Exception:
        # older CountVectorizer uses get_feature_names
        try:
            feature_names = vectorizer.get_feature_names()
        except Exception:
            feature_names = np.array([f"f{i}" for i in range(model.n_features_in_)]) if hasattr(model, "n_features_in_") else []

    # Get importances (cuML returns numpy, sklearn returns numpy)
    try:
        importances = model.feature_importances_
        importances = np.asarray(importances)
    except Exception:
        importances = np.zeros(len(feature_names))

    # Prepare dataframe with pandas (CPU)
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    top_features = fi_df.sort_values(by='importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title(f'Top {top_n} Most Important K-mers', fontsize=16)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('K-mer', fontsize=12)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"   - Feature importance plot saved to '{output_path}'")


# ---------------- Main pipeline ----------------

def main():
    parser = argparse.ArgumentParser(description="GPU-Accelerated eDNA Classifier Training Pipeline (fallback-enabled)")
    parser.add_argument("--fasta", required=True, help="Input FASTA file containing reference sequences.")
    parser.add_argument("--labels", required=True, help="CSV file mapping sequence IDs to taxonomy labels.")
    parser.add_argument("--k", type=int, default=6, help="k-mer size.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test fraction.")
    args = parser.parse_args()

    print("\n[ STEP 1/8 ] Loading and Preparing Data...")
    seq_dict = load_sequences(args.fasta)
    labels_df = load_labels(args.labels)

    # unify on pandas/cudf index handling
    if USE_GPU:
        # labels_df is cudf
        data = labels_df[labels_df["id"].isin(list(seq_dict.keys()))].copy()
        data["sequence"] = cudf.Series(data["id"].to_pandas().map(seq_dict))
        data.dropna(subset=['sequence', 'taxonomy'], inplace=True)
    else:
        data = labels_df[labels_df["id"].isin(list(seq_dict.keys()))].copy()
        data["sequence"] = data["id"].map(seq_dict)
        data.dropna(subset=['sequence', 'taxonomy'], inplace=True)

    if data.empty:
        print("🔴 ERROR: No matching sequences found between the FASTA and labels file. Exiting.")
        sys.exit(1)
    print(f"✅ Found {len(data)} labeled sequences for training.")

    # Filter low-count classes (remove classes with < 2 members)
    if USE_GPU:
        label_counts = data['taxonomy'].value_counts()
        labels_to_remove = label_counts[label_counts < 2].index.to_pandas().tolist()
        if labels_to_remove:
            data = data[~data['taxonomy'].isin(labels_to_remove)]
    else:
        label_counts = data['taxonomy'].value_counts()
        labels_to_remove = label_counts[label_counts < 2].index.tolist()
        if labels_to_remove:
            data = data[~data['taxonomy'].isin(labels_to_remove)]

    print(f"✅ Data size after filtering: {len(data)} sequences.")

    # Convert taxonomy to categorical codes
    if USE_GPU:
        data['taxonomy'] = data['taxonomy'].astype('category')
        y = data['taxonomy'].cat.codes
        # mapping back
        code_to_label = {i: cat for i, cat in enumerate(data['taxonomy'].cat.categories)}
    else:
        data['taxonomy'] = data['taxonomy'].astype('category')
        y = data['taxonomy'].cat.codes
        code_to_label = {i: cat for i, cat in enumerate(data['taxonomy'].cat.categories)}

    # Vectorize sequences (k-mer)
    print("\n[ STEP 3/8 ] Vectorizing sequences...")
    if USE_GPU:
        vectorizer = cuCountVectorizer(analyzer="char", ngram_range=(args.k, args.k))
        X = vectorizer.fit_transform(data["sequence"])
        print(f"✅ Vectorized to GPU matrix with shape {X.shape}.")
    else:
        vectorizer = skCountVectorizer(analyzer="char", ngram_range=(args.k, args.k))
        X = vectorizer.fit_transform(data["sequence"].astype(str).tolist())
        print(f"✅ Vectorized to CPU matrix with shape {X.shape}.")

    # Train-test split
    print("\n[ STEP 4/8 ] Train-test split...")
    if USE_GPU:
        try:
            X_train, X_test, y_train, y_test = cu_train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)
        except Exception as e:
            # Fall back to CPU split: move to CPU numpy/pandas
            print("⚠️ cuML train_test_split failed; falling back to sklearn split (CPU).", e)
            X_cpu = X.get() if hasattr(X, "get") else X
            y_cpu = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
            X_train, X_test, y_train, y_test = sk_train_test_split(X_cpu, y_cpu, test_size=args.test_size, random_state=42, stratify=y_cpu)
            USE_GPU_LOCAL = False
    else:
        X_train, X_test, y_train, y_test = sk_train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)

    print(f"   - Training samples: {getattr(X_train, 'shape', None)}")
    print(f"   - Testing samples:  {getattr(X_test, 'shape', None)}")

    # Balance with SMOTE
    print("\n[ STEP 5/8 ] Balancing training data with SMOTE (GPU if available else CPU)...")
    if USE_GPU and HAVE_CU_SMOTE:
        smote = cuSMOTE(random_state=42, k_neighbors=max(1, int(y_train.value_counts().min()) - 1))
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    else:
        if not HAVE_IMBLEARN:
            print("🔴 ERROR: imbalanced-learn (imblearn) not installed. Please 'conda install -c conda-forge imbalanced-learn' or 'pip install imbalanced-learn'.")
            sys.exit(1)
        # Convert training data to CPU arrays / sparse if needed
        if USE_GPU:
            # move X_train to CPU
            try:
                X_train_cpu = X_train.get() if hasattr(X_train, "get") else X_train
            except Exception:
                X_train_cpu = X_train
            y_train_cpu = y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)
        else:
            X_train_cpu = X_train
            y_train_cpu = y_train

        smote = skSMOTE(random_state=42, k_neighbors=max(1, int(np.min(np.unique(y_train_cpu, return_counts=True)[1])) - 1))
        X_train_res, y_train_res = smote.fit_resample(X_train_cpu, y_train_cpu)

    print("✅ SMOTE done. New training size:", getattr(X_train_res, "shape", None))

    # Hyperparameter tuning
    print("\n[ STEP 6/8 ] Hyperparameter search...")
    param_grid = {'n_estimators': [100, 200], 'max_depth': [20, 30], 'min_samples_split': [2, 5]}
    if USE_GPU:
        try:
            rf = cuRandomForest(random_state=42)
            # scikit-learn GridSearchCV may work with cuML objects but it can be finicky.
            grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=2)
            grid.fit(X_train_res, y_train_res)
            best_model = grid.best_estimator_
            print("✅ Best params (GPU RF):", grid.best_params_)
        except Exception as e:
            print("⚠️ cuML GridSearch or RF failed; falling back to sklearn RandomForest on CPU.", e)
            rf = skRandomForest(random_state=42)
            grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=2)
            grid.fit(X_train_res, y_train_res)
            best_model = grid.best_estimator_
    else:
        rf = skRandomForest(random_state=42)
        grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=2)
        grid.fit(X_train_res, y_train_res)
        best_model = grid.best_estimator_

    # Evaluation
    print("\n[ STEP 7/8 ] Evaluating model on test set...")
    # Predictions
    if USE_GPU and hasattr(best_model, "predict") and not isinstance(best_model, skRandomForest):
        y_pred = best_model.predict(X_test)
        # bring predictions to CPU
        if hasattr(y_pred, "to_numpy"):
            y_pred_np = y_pred.to_numpy()
        else:
            try:
                y_pred_np = y_pred.get() if hasattr(y_pred, "get") else np.asarray(y_pred)
            except Exception:
                y_pred_np = np.asarray(y_pred)
    else:
        # Ensure test data is CPU-suitable for sklearn
        X_test_cpu = X_test.get() if (USE_GPU and hasattr(X_test, "get")) else X_test
        y_pred_np = best_model.predict(X_test_cpu)

    # y_test to CPU labels
    y_test_np = ensure_numpy(y_test)
    # Map back to original labels
    y_test_labels = [code_to_label[int(c)] for c in np.asarray(y_test_np)]
    y_pred_labels = [code_to_label[int(c)] for c in np.asarray(y_pred_np)]

    class_labels = sorted(list(code_to_label.values()))
    report = classification_report(y_test_labels, y_pred_labels, digits=3, labels=class_labels, zero_division=0)
    print("Classification Report:\n", report)

    output_dir = Path(".")
    plot_confusion_matrix(y_test_labels, y_pred_labels, class_labels, output_dir / "confusion_matrix.png")
    plot_feature_importance(best_model, vectorizer, output_dir / "feature_importance.png")

    # Save artifacts
    print("\n[ STEP 8/8 ] Saving artifacts...")
    joblib.dump(best_model, output_dir / "trained_model_artifact.joblib")
    joblib.dump(vectorizer, output_dir / "vectorizer_artifact.joblib")
    with open(output_dir / "training_metrics.txt", "w") as f:
        f.write("eDNA Classifier Training Report\n================================\n\n")
        f.write(f"Best Hyperparameters:\n{getattr(grid, 'best_params_', 'N/A')}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print("✅ Done. Artifacts saved to the current directory.")


if __name__ == "__main__":
    main()
