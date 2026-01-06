# This file fixes data leakage in validation and test sets by removing sequences that partially overlap with training set sequences.
# Run this script before training/evaluation to create cleaned validation and test sets.
import numpy as np


TRAIN_FILE = "train_data.txt"
VAL_FILE = "validation_data.txt"
TEST_FILE = "test_data.txt"

VAL_OUT = "validation_clean.txt"
TEST_OUT = "test_clean.txt"

# How many tokens to consider for "partial overlap"
OVERLAP_TOKENS = 4  # e.g., first 4 tokens, but you can adjust as needed

# Load sequences from a file
def load_sequences(fname):

    sequences = []
    with open(fname, 'r') as f:
        for line in f:
            tokens = [int(tok) for tok in line.strip().split()]
            sequences.append(tokens)
    return sequences

# Remoeve sequences from target_seqs whose first n_tokens appear in train_seqs
def filter_overlap(train_seqs, target_seqs, n_tokens):
    train_prefixes = set(tuple(seq[:n_tokens]) for seq in train_seqs if len(seq) >= n_tokens)
    filtered = [seq for seq in target_seqs if tuple(seq[:n_tokens]) not in train_prefixes]
    return filtered

# Save sequences to a file
def save_sequences(sequences, fname):
    with open(fname, 'w') as f:
        for seq in sequences:
            f.write(" ".join(map(str, seq)) + "\n")

def main():
    print("Loading datasets...")
    train_seqs = load_sequences(TRAIN_FILE)
    val_seqs = load_sequences(VAL_FILE)
    test_seqs = load_sequences(TEST_FILE)

    print(f"Original counts -> Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}")

    val_clean = filter_overlap(train_seqs, val_seqs, OVERLAP_TOKENS)
    test_clean = filter_overlap(train_seqs, test_seqs, OVERLAP_TOKENS)

    print(f"Filtered counts -> Val: {len(val_clean)}, Test: {len(test_clean)}")

    save_sequences(val_clean, VAL_OUT)
    save_sequences(test_clean, TEST_OUT)

    print(f" Saved cleaned validation to {VAL_OUT} and test to {TEST_OUT}")

if __name__ == "__main__":
    main()
