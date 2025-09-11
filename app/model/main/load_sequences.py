
from Bio import SeqIO

def load_sequences(fasta_file):
    """Loads sequences from a FASTA file."""
    print(f"Loading sequences from {fasta_file}...")
    try:
        records = list(SeqIO.parse(fasta_file, "fasta"))
        sequences = [str(rec.seq) for rec in records]
        print(f"Successfully loaded {len(sequences)} sequences.")
        return sequences
    except FileNotFoundError:
        print(f"Error: The file '{fasta_file}' was not found.")
        exit()