#!/usr/bin/env python3
"""
STEP 2: Create Labels File (Smarter Version)
---------------------------------------------
This script intelligently creates a CSV label file from a FASTA file's
description headers.

This enhanced version ignores common, uninformative words (like "uncultured")
to find the first truly meaningful taxonomic identifier.
"""
import argparse
import pandas as pd
from Bio import SeqIO

def parse_taxonomy(description):
    """
    Intelligently parses the taxonomy from a FASTA description line.

    It first searches for a list of predefined, high-level taxonomic groups.
    If none are found, it finds the first word in the description that is not
    a common, uninformative term.
    """
    # List of high-priority taxa to search for first.
    # We can add more specific terms here as we discover them.
    known_taxa = [
        "Metazoa", "Fungi", "Protista", "Viridiplantae", "Bacteria", "Archaea",
        "Rhodophyta", "Stramenopila", "Alveolata", "Rhizaria", "Amoebozoa",
        "Myxosporea" # This was in your other data, so we'll keep it.
    ]
    
    description_lower = description.lower()
    for taxa in known_taxa:
        if taxa.lower() in description_lower:
            return taxa # Return the properly capitalized version

    # --- NEW, SMARTER FALLBACK LOGIC ---
    
    # List of common, uninformative words to ignore.
    ignore_list = [
        "uncultured", "environmental", "sample", "clone", "sequence",
        "gene", "dna", "partial", "isolate", "eukaryote", "shotgun"
    ]

    # Split the description and get the sequence ID to ignore it.
    parts = description.split()
    seq_id = parts[0]

    # Find the first word that is NOT the ID and NOT in our ignore list.
    for part in parts[1:]: # Start from the second word
        # Clean up the word by removing commas, semicolons, etc.
        cleaned_part = part.strip("(),;.")
        if cleaned_part.lower() not in ignore_list:
            return cleaned_part # This is our likely taxonomy

    # If all else fails, return a specific "NeedsReview" tag.
    return "NeedsReview"


def main():
    parser = argparse.ArgumentParser(description="Create a labels CSV from a FASTA file.")
    parser.add_argument("--fasta", required=True, help="Input FASTA file (the original raw one).")
    parser.add_argument("--output", default="labels.csv", help="Output CSV file name.")
    args = parser.parse_args()

    records = []
    print(f"📄 Parsing '{args.fasta}' with smarter logic to create label file...")
    for record in SeqIO.parse(args.fasta, "fasta"):
        records.append({
            "id": record.id,
            "taxonomy": parse_taxonomy(record.description)
        })

    df = pd.DataFrame(records)
    df.to_csv(args.output, index=False)
    
    print(f"✅ Wrote {len(df)} labeled sequences to '{args.output}'")
    print("\nTaxonomy value counts (you should see fewer 'uncultured' labels):")
    print(df['taxonomy'].value_counts().head(20))

if __name__ == "__main__":
    main()