#!/usr/bin/env python3
"""
train_model.py (v2.0 - Enhanced)
----------------------------------
This script trains a robust, supervised classifier for eDNA taxonomy classification
by incorporating hyperparameter tuning and generating detailed evaluation metrics.

Key Features:
- **Hyperparameter Tuning:** Uses GridSearchCV to systematically find the optimal
  parameters for the RandomForest model, improving its accuracy.
- **Imbalanced Data Handling:** Uses `class_weight='balanced'` to prevent the model
  from being biased towards majority classes.
- **Detailed Evaluation:** Generates a classification report, a confusion matrix
  heatmap, and a plot of the most important k-mer features.
- **Robust File Paths:** Uses `pathlib` to ensure files are saved and loaded
  reliably, regardless of where the script is run from.

Usage:
    python train_model.py --fasta your_references.fasta --labels your_labels.csv --k 6

Outputs:
    - trained_model.joblib: The best, fully trained classifier.
    - vectorizer.joblib: The k-mer feature encoder fitted on the training data.
    - training_metrics.txt: A detailed report of the model's performance.
    - confusion_matrix.png: A heatmap visualizing model predictions vs. actual labels.
    - feature_importance.png: A plot showing the k-mers most influential in classification.
"""
import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from Bio import SeqIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

# ============================================================
# -------------------- HELPER FUNCTIONS ----------------------
# ============================================================

def load_sequences(fasta_file):
    """
    Loads sequences from a FASTA file into a dictionary.
    This allows for quick lookup of a sequence by its ID.
    Args:
        fasta_file (str): Path to the input FASTA file.
    Returns:
        dict: A dictionary mapping sequence ID to sequence string.
    """
    print(f"🧬 Loading sequences from '{fasta_file}'...")
    try:
        return {record.id: str(record.seq).upper() for record in SeqIO.parse(fasta_file, "fasta")}
    except FileNotFoundError:
        print(f"🔴 ERROR: FASTA file not found at '{fasta_file}'.")
        exit(1)


def load_labels(csv_file):
    """
    Loads taxonomic labels from a CSV file.
    Args:
        csv_file (str): Path to the input CSV file.
    Returns:
        pd.DataFrame: A DataFrame with sequence IDs and their corresponding taxonomy.
    """
    print(f"🏷️  Loading labels from '{csv_file}'...")
    try:
        return pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"🔴 ERROR: CSV labels file not found at '{csv_file}'.")
        exit(1)


def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """
    Generates and saves a heatmap of the confusion matrix.
    A confusion matrix shows where the model is getting confused (e.g.,
    predicting 'Fungi' when the actual label is 'Protista').
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"   - Confusion matrix saved to '{output_path}'")


def plot_feature_importance(model, vectorizer, output_path, top_n=20):
    """
    Plots the most important k-mers used by the model for classification.
    This helps understand what sequence features the model is relying on.
    """
    importances = model.feature_importances_
    feature_names = vectorizer.get_feature_names_out()
    
    # Create a DataFrame of features and their importance scores
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    # Sort by importance and get the top N
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
    parser = argparse.ArgumentParser(description="Enhanced eDNA Classifier Training Pipeline")
    parser.add_argument("--fasta", required=True, help="Input FASTA file containing reference sequences.")
    parser.add_argument("--labels", required=True, help="CSV file mapping sequence IDs to taxonomy labels.")
    parser.add_argument("--k", type=int, default=6, help="The size of k-mers to use as features (e.g., 6 for hexamers).")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of the data to hold out for testing.")
    args = parser.parse_args()

    # --- [ STEP 1/6 ] LOAD AND PREPARE DATA ---
    print("\n[ STEP 1/6 ] Loading and Preparing Data...")
    seq_dict = load_sequences(args.fasta)
    labels_df = load_labels(args.labels)

    # Combine sequences and labels into a single DataFrame
    # This ensures we only work with sequences that have a corresponding label.
    data = labels_df[labels_df["id"].isin(seq_dict.keys())].copy()
    data["sequence"] = data["id"].map(seq_dict)

    # Drop any rows that might have missing sequences or labels
    data.dropna(subset=['sequence', 'taxonomy'], inplace=True)
    
    if data.empty:
        print("🔴 ERROR: No matching sequences found between the FASTA and labels file. Exiting.")
        exit(1)
        
    print(f"✅ Found {len(data)} labeled sequences for training.")

    # --- [ STEP 2/6 ] K-MER FEATURE EXTRACTION ---
    print("\n[ STEP 2/6 ] Converting DNA Sequences to K-mer Vectors...")
    # The CountVectorizer will treat each sequence as a document and count the
    # occurrences of each possible k-mer (character n-grams).
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(args.k, args.k))
    X = vectorizer.fit_transform(data["sequence"])
    y = data["taxonomy"]
    print(f"✅ Sequences vectorized into a matrix with shape {X.shape}.")

    # --- [ STEP 3/6 ] TRAIN-TEST SPLIT ---
    print("\n[ STEP 3/6 ] Splitting Data into Training and Testing Sets...")
    # 'stratify=y' ensures that the proportion of each class (taxonomy) is the
    # same in both the training and testing sets, which is crucial for imbalanced datasets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"   - Training set size: {X_train.shape[0]} samples")
    print(f"   - Testing set size:  {X_test.shape[0]} samples")

    # --- [ STEP 4/6 ] HYPERPARAMETER TUNING WITH GRIDSEARCHCV ---
    print("\n[ STEP 4/6 ] Searching for the Best Model Parameters (Hyperparameter Tuning)...")
    # Define a 'grid' of parameters to test for the RandomForestClassifier.
    # GridSearchCV will train a model for every combination and find the best one
    # using cross-validation. This is much more robust than guessing parameters.
    param_grid = {
        'n_estimators': [100, 200, 300], # Number of trees in the forest
        'max_depth': [10, 30, None],       # Maximum depth of the trees
        'min_samples_split': [2, 5]      # Minimum samples required to split a node
    }
    
    # Initialize the model. `class_weight='balanced'` tells the model to pay more
    # attention to minority classes, preventing bias.
    rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    
    # Set up the search. cv=3 means 3-fold cross-validation.
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # The best model found by the grid search
    best_model = grid_search.best_estimator_
    print(f"✅ Best parameters found: {grid_search.best_params_}")

    # --- [ STEP 5/6 ] MODEL EVALUATION ---
    print("\n[ STEP 5/6 ] Evaluating the Best Model on the Test Set...")
    y_pred = best_model.predict(X_test)
    
    # Get unique class labels for plotting
    class_labels = sorted(y.unique())
    
    report = classification_report(y_test, y_pred, digits=3, labels=class_labels)
    print("Classification Report:\n", report)

    # Generate and save evaluation plots
    output_dir = Path(".") # Save in the current directory
    plot_confusion_matrix(y_test, y_pred, class_labels, output_dir / "confusion_matrix.png")
    plot_feature_importance(best_model, vectorizer, output_dir / "feature_importance.png")

    # --- [ STEP 6/6 ] SAVE ARTIFACTS ---
    print("\n[ STEP 6/6 ] Saving Model, Vectorizer, and Metrics...")
    # Save the trained model and vectorizer for use in the analysis pipeline
    joblib.dump(best_model, output_dir / "trained_model.joblib")
    joblib.dump(vectorizer, output_dir / "vectorizer.joblib")
    
    # Save the detailed metrics to a text file
    with open(output_dir / "training_metrics.txt", "w") as f:
        f.write("eDNA Classifier Training Report\n")
        f.write("=================================\n\n")
        f.write(f"Best Hyperparameters Found:\n{grid_search.best_params_}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        
    print("✅ Training complete!")
    print(f"💾 Model saved to '{output_dir / 'trained_model.joblib'}'")
    print(f"💾 Vectorizer saved to '{output_dir / 'vectorizer.joblib'}'")
    print(f"💾 Metrics saved to '{output_dir / 'training_metrics.txt'}'")

if __name__ == "__main__":
    main()
