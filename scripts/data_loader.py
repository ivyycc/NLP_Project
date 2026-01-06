# This file loads the WCST dataset for training the next-token prediction model.

import torch
from torch.utils.data import Dataset, DataLoader

class WCST_Dataset(Dataset):
    def __init__(self, data_file):
        self.trials = []
        with open(data_file, 'r') as f:
            for line in f:
                # Read line, split by space, and convert to integers
                tokens = [int(token) for token in line.strip().split()]
                self.trials.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        # The task is next-token prediction.
        # The input is the sequence from the beginning up to the 'sep' token.
        # The target is the full sequence, which the model tries to predict one step ahead.
        full_sequence = self.trials[idx]
        
        # The input is everything except the last token
        # Example: [1, 41, 22, 63, 13, 68, 64] -> Input: [1, 41, 22, 63, 13, 68]
        input_sequence = full_sequence[:-1]
        
        # The target is everything except the first token
        # Example: [1, 41, 22, 63, 13, 68, 64] -> Target: [41, 22, 63, 13, 68, 64]
        # The model sees token `x_i` and tries to predict `x_{i+1}`.
        target_sequence = full_sequence[1:]
        
        return input_sequence, target_sequence

# Example of how to use it for testing that it works, but for the pipeline we have, we dont actually need to run it
if __name__ == "__main__":
    train_dataset = WCST_Dataset("train_data.txt")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Get one batch
    inputs, targets = next(iter(train_loader))
    
    print("--- Data Loader Example ---")
    print(f"Batch of inputs shape: {inputs.shape}")
    print(f"Batch of targets shape: {targets.shape}")
    print("\nExample Input:\n", inputs[0])
    print("\nExample Target:\n", targets[0])