# This script checks for data leakage in the WCST datasets.
# It verifies that there is no partial sequence overlap between training and validation/test sets,
# Run to check for leakage before training/evaluation.
import torch

TRAIN_FILE = "train_data.txt"
VAL_FILE   = "validation_clean.txt"
TEST_FILE  = "test_clean.txt"
SEP_TOKEN  = 68
CATEGORY_RANGE = range(64, 68)  # C1..C4 tokens

# Load sequences from a file
def load_sequences(fname):

    sequences = []
    with open(fname, 'r') as f:
        for line in f:
            sequences.append([int(t) for t in line.strip().split()])
    return sequences

# Check for partial sequence overlap
def check_partial_overlap(train_seqs, other_seqs, prefix_len=4):

    train_prefixes = {tuple(seq[:prefix_len]) for seq in train_seqs if len(seq) >= prefix_len}
    overlap_count = sum(1 for seq in other_seqs if tuple(seq[:prefix_len]) in train_prefixes)
    return overlap_count

# Check for label leakage
# Input sequence = seq[:-1], target = seq[1:]
# Specifically checks for tokens after SEP matching final category.
def check_label_leakage(seqs):

    leak_count = 0
    for seq in seqs:
        input_seq = seq[:-1]
        target_seq = seq[1:]
        # find SEP in input
        sep_indices = [i for i, t in enumerate(input_seq) if t == SEP_TOKEN]
        if not sep_indices:
            continue
        last_token = target_seq[-1]
        for idx in sep_indices:
            if idx + 1 < len(input_seq):
                if input_seq[idx + 1] == last_token:
                    leak_count += 1
    return leak_count

# check for feature leakage to see if target token appears anywhere in input sequence
def check_feature_leakage(seqs):

    leak_count = 0
    for seq in seqs:
        input_seq = seq[:-1]
        target_seq = seq[1:]
        last_token = target_seq[-1]
        if last_token in input_seq:
            leak_count += 1
    return leak_count

def main():
    print("Loading datasets...")
    train_seqs = load_sequences(TRAIN_FILE)
    val_seqs   = load_sequences(VAL_FILE)
    test_seqs  = load_sequences(TEST_FILE)
    print(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}\n")

    # Partial sequence overlap
    val_overlap = check_partial_overlap(train_seqs, val_seqs)
    test_overlap = check_partial_overlap(train_seqs, test_seqs)
    print(f"Partial sequence overlap (first 4 tokens):")
    print(f"  Train ∩ Val: {val_overlap} sequences")
    print(f"  Train ∩ Test: {test_overlap} sequences\n")

    # Label leakage
    val_label_leak = check_label_leakage(val_seqs)
    test_label_leak = check_label_leakage(test_seqs)
    print(f"Label leakage (target after SEP appears in input):")
    print(f"  Val: {val_label_leak} sequences")
    print(f"  Test: {test_label_leak} sequences\n")

    # Feature leakage
    val_feat_leak = check_feature_leakage(val_seqs)
    test_feat_leak = check_feature_leakage(test_seqs)
    print(f"Feature leakage (target appears anywhere in input):")
    print(f"  Val: {val_feat_leak} sequences")
    print(f"  Test: {test_feat_leak} sequences\n")

    print(" Leakage check complete.")

if __name__ == "__main__":
    main()
