#!/usr/bin/env python3
"""
STEP 5: Validate a Cluster with BLAST
--------------------------------------
This script takes a single DNA sequence (e.g., the representative sequence
from a cluster in your HTML report) and queries it against the NCBI BLAST
database online.

It fetches the top hits, providing a powerful way to validate your model's
classification and get a species-level identification.

Prerequisite: Biopython must be installed (`pip install biopython`).
"""
import argparse
from Bio.Blast import NCBIWWW, NCBIXML

def run_blast_query(sequence, hitlist_size=5):
    """
    Submits a sequence to NCBI BLAST and returns the parsed results.
    """
    print(f"🔬 Submitting BLASTn query for sequence: '{sequence[:30]}...'")
    print("   (This can take 30-60 seconds, please be patient)...")
    
    try:
        # Use qblast to submit the query
        result_handle = NCBIWWW.qblast(
            program="blastn",      # Nucleotide-to-nucleotide search
            database="nt",         # 'nt' is the comprehensive nucleotide database
            sequence=sequence,
            hitlist_size=hitlist_size
        )
        
        # Parse the XML results
        blast_record = NCBIXML.read(result_handle)
        return blast_record
        
    except Exception as e:
        print(f"🔴 An error occurred during the BLAST query: {e}")
        print("   Please check your internet connection and try again.")
        return None

def display_blast_results(blast_record):
    """
    Formats and prints the top BLAST hits in a readable way.
    """
    if not blast_record or not blast_record.alignments:
        print("\n❌ No significant hits found in the BLAST database for this sequence.")
        return

    print("\n✅ BLAST Results Found! Top hits are:")
    print("-" * 50)
    
    for i, alignment in enumerate(blast_record.alignments):
        # The top hit within an alignment
        hsp = alignment.hsps[0]
        
        # Calculate percent identity
        identity_percent = (hsp.identities / hsp.align_length) * 100
        
        print(f"\nHit #{i+1}:")
        print(f"   Scientific Name: {alignment.title[:80]}...") # Truncate long titles
        print(f"   Identity:        {identity_percent:.2f}% ({hsp.identities}/{hsp.align_length} bases)")
        print(f"   E-value:         {hsp.expect:.2e}")
        print(f"   Description:     The E-value (Expect value) is the number of hits one can 'expect'")
        print(f"                    to see by chance. The lower the E-value, the more significant the match.")
        print("-" * 50)

def main():
    parser = argparse.ArgumentParser(
        description="Validate a DNA sequence against the NCBI BLAST database.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    parser.add_argument(
        "sequence",
        help="The DNA sequence to validate (e.g., 'GATTACAGATTACA...')."
    )
    parser.add_argument(
        "--hits",
        type=int,
        default=3,
        help="Number of top hits to display (default: 3)."
    )
    args = parser.parse_args()

    blast_record = run_blast_query(args.sequence, args.hits)
    display_blast_results(blast_record)

if __name__ == "__main__":
    main()
