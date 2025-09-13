#!/usr/bin/env python3
"""
STEP 1: Prepare Reference Database (Corrected)
----------------------------------------------
Cleans a raw FASTA file by removing duplicates and filtering sequences by length.
This version uses the correct VSEARCH flags for length filtering.

Prerequisite: VSEARCH must be installed.
"""
import argparse
import subprocess
import os
import shutil

def check_vsearch_installed():
    """Checks if vsearch is installed and in the PATH."""
    if not shutil.which("vsearch"):
        print("🔴 ERROR: VSEARCH is not installed or not in your PATH.")
        print("   Please install it. Recommended: 'conda install -c bioconda vsearch'")
        return False
    return True

def run_command(command):
    """Executes a shell command and reports errors."""
    print(f"   Running command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        # vsearch often prints its summary stats to stderr, so we print it on success
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"🔴 COMMAND FAILED: {' '.join(command)}")
        print(f"   ERROR:\n{e.stderr}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Clean and filter a FASTA reference database using VSEARCH.")
    parser.add_argument("--input", required=True, help="Path to the raw input FASTA file.")
    parser.add_argument("--output", required=True, help="Path for the cleaned, output FASTA file.")
    parser.add_argument("--minlen", type=int, default=150, help="Minimum sequence length to keep.")
    parser.add_argument("--maxlen", type=int, default=500, help="Maximum sequence length to keep.")
    args = parser.parse_args()

    if not check_vsearch_installed():
        exit(1)

    print(f"🛠️  Starting reference database preparation for: '{args.input}'")
    temp_derep_file = "temp_unique.fasta"

    # Step 1: Dereplicate (unchanged)
    print("\n[ 1/2 ] Removing duplicate sequences...")
    derep_command = [
        "vsearch",
        "--derep_fulllength", args.input,
        "--output", temp_derep_file,
        "--sizeout"
    ]
    run_command(derep_command)

    # Step 2: Filter by length (CORRECTED FLAGS)
    print("\n[ 2/2 ] Filtering sequences by length...")
    filter_command = [
        "vsearch",
        "--fastx_filter", temp_derep_file,
        "--fastaout", args.output,
        # --- THE FIX IS HERE ---
        "--fastq_minlen", str(args.minlen),  # Changed from --minseqlength
        "--fastq_maxlen", str(args.maxlen)   # Changed from --maxseqlength
    ]
    run_command(filter_command)

    # Clean up intermediate file
    os.remove(temp_derep_file)

    print(f"\n✅ Success! Cleaned database saved to '{args.output}'")


if __name__ == "__main__":
    main()

