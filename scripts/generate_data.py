# This script generates training, validation, and test datasets for the Wisconsin Card Sorting Test (WCST) task.
# Each dataset consists of multiple trials, where each trial is represented as a sequence of tokens.
# Run this script to create "train_data.txt", "validation_data.txt", and "test_data.txt" before cleaning the dataset

import numpy as np
from wcst import WCST 

#     Each line in the file will be one complete trial sequence:
#      [context (8 tokens)] + [question (3 tokens)] => total 11 tokens

def generate_and_save_data(filename, num_trials, batch_size=32):

    print(f"Generating {num_trials} trials for {filename}...")

    # Initialize the WCST generator
    wcst_generator = WCST(batch_size=batch_size)

    num_batches = int(np.ceil(num_trials / batch_size))

    with open(filename, 'w') as f:
        trial_count = 0
        for i in range(num_batches):
            # The generator yields a context (example) and a target (question)
            context, target = next(wcst_generator.gen_batch())

            # Combine them to form full trials: context + target
            # context shape: (batch_size, 8)
            # target shape:  (batch_size, 3)
            full_trials = np.hstack([context, target])  # -> (batch_size, 11)
            # This is the full sequence for each trial since the task is full sequence next token prediction
            for trial in full_trials:
                if trial_count < num_trials:
                    # Truncate sequence up to category token (stop before EOS or next SEP)
                    tokens = []
                    for t in trial:
                        tokens.append(int(t))
                        # category tokens are indices 64–67
                        if 64 <= int(t) <= 67:
                            break

                    f.write(" ".join(map(str, tokens)) + "\n")
                    trial_count += 1

            # Periodically switch the sorting rule to ensure variety in the data
            if i % 10 == 0:
                wcst_generator.context_switch()

    print(f"Done. Saved {trial_count} trials to {filename}.")



if __name__ == "__main__":
    # Define the number of trials for each dataset
    TRAIN_TRIALS = 50000
    VALID_TRIALS = 2000
    TEST_TRIALS = 5000

    # Generate the datasets
    generate_and_save_data("train_data.txt", TRAIN_TRIALS)
    generate_and_save_data("validation_data.txt", VALID_TRIALS)
    generate_and_save_data("test_data.txt", TEST_TRIALS)