
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


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