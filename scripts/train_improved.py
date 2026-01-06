# This is the train.py file for training the improved Transformer model with multi-feature token embeddings on the WCST dataset.
# Run this after generating and cleaning the datasets to train the model.
# You will notice the hyperparameters are for the most part the same to train.py for a consistent and fair comparison.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import WCST_Dataset
from model_improved import TransformerWithFeatures   # use the improved model file
import os
import json

# --- 1. Hyperparameters and Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE =  3e-4
BATCH_SIZE = 64
NUM_EPOCHS = 40
VOCAB_SIZE = 70  # 64 cards + 4 categories + sep + eos
D_MODEL = 256
NUM_LAYERS = 4
HEADS = 8
D_FF = 1024      # Forward expansion (4 * 256 = 1024)
DROPOUT = 0.05
MAX_LENGTH = 10  # Max length of a sequence

# Primary setting: train full-sequence next-token prediction (autoregressive)
# Optionally add a small weighted extra objective on the final token (AUX_WEIGHT)
AUX_WEIGHT = 0.0  
TIE_WEIGHTS = False # TokenEmbeddingWithFeatures does not expose token_embedding; hence keep False
BEST_MODEL_PATH = "transformer_wcst_with_features.pth"
METRICS_PATH = "training_metrics_features.json"

# Optimizer choices
USE_ADAMW = True
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 5  # Stop if val accuracy does not improve for 5 epochs
early_stop_counter = 0


def mask_generate(seq_len):
    # Create causal mask (lower triangular)
    mask = torch.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:
                mask[i, j] = 1
            else:
                mask[i, j] = 0
    mask = mask.bool()
    return mask

def _category_stats(preds, labels):
    # preds, labels are token indices (64..67)
    preds_c = preds - 64
    labels_c = labels - 64
    num_classes = 4
    per_class_correct = torch.zeros(num_classes, dtype=torch.long, device=preds.device)
    per_class_total = torch.zeros(num_classes, dtype=torch.long, device=preds.device)
    for c in range(num_classes):
        mask = (labels_c == c)
        per_class_total[c] = mask.sum()
        per_class_correct[c] = ((preds_c == c) & mask).sum()
    return per_class_correct.cpu().tolist(), per_class_total.cpu().tolist()

def main():
    train_losses = []
    val_losses = []
    val_accuracies = []

    print("Loading data...")
    train_dataset = WCST_Dataset("train_data.txt")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    validation_dataset = WCST_Dataset("validation_clean.txt")
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)

    # Sanity check on validation tokens
    sample_inputs, sample_targets = next(iter(validation_loader))
    category_tokens = sample_targets[:, -1]
    uniq, counts = torch.unique(category_tokens, return_counts=True)
    print("Sanity check (validation batch): unique tokens at targets[:, -1]:")
    for u, c in zip(uniq.tolist(), counts.tolist()):
        print(f"  token {u}: {c} samples")
    if not all(64 <= int(u) <= 67 for u in uniq):
        print(" Warning: some targets[:, -1] are not category tokens in [64..67].")

    # --- Model / optimizer / loss ---
    print(f"Using device: {DEVICE}")
    model = TransformerWithFeatures(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        num_heads=HEADS,
        dropout=DROPOUT
    ).to(DEVICE)

    if TIE_WEIGHTS:
        try:
            model.output_layer.weight = model.embedding.token_embedding.weight
            print("Tied output_layer.weight to embedding.token_embedding.weight")
        except Exception as e:
            print("Weight tying unavailable (embedding does not expose token_embedding):", e)

    if USE_ADAMW:
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    criterion = nn.CrossEntropyLoss()  # for classification (used for per-token CE)

    best_val_acc = -1.0
    best_epoch = -1

    # --- Training loop (full-sequence next-token objective) ---
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            seq_len = inputs.size(1)
            mask = mask_generate(seq_len).to(DEVICE)

            outputs = model(inputs, mask, return_attention=False)  # [B, L, V]

            # Full-sequence next-token loss (train objective)
            outputs_reshaped = outputs.reshape(-1, outputs.shape[2])    # [B*L, V]
            targets_reshaped = targets.reshape(-1)                      # [B*L]
            loss_full = criterion(outputs_reshaped, targets_reshaped)

            # Diagnostic / optional extra final-token loss
            logits_last = outputs[:, -1, :]      # [B, V]
            labels_last = targets[:, -1]         # [B]
            loss_last = criterion(logits_last, labels_last)

            loss = loss_full + AUX_WEIGHT * loss_last

            optimizer.zero_grad()
            loss.backward()

            # Small gradient debug for the first batch of epoch
            if batch_idx == 0:
                grad_samples = []
                for i, p in enumerate(model.parameters()):
                    if p.grad is None:
                        grad_samples.append(float('nan'))
                    else:
                        grad_samples.append(p.grad.norm().item())
                    if i >= 9:
                        break
                print(f"Epoch {epoch+1} batch 0 sample grad norms: {grad_samples}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx}/{len(train_loader)}], Loss(full+aux): {loss.item():.4f}")

        # Validation (compute same full-sequence CE for consistency; also report final-token metrics)
        model.eval()
        total_val_loss_full = 0.0   # full-sequence CE (same as training objective)
        total_val_loss_last = 0.0   # final-token CE (diagnostic)
        correct_predictions = 0
        total_predictions = 0
        per_class_correct = [0, 0, 0, 0]
        per_class_total = [0, 0, 0, 0]
        mean_category_probs = torch.zeros(4, device=DEVICE)

        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                seq_len = inputs.size(1)
                mask = mask_generate(seq_len).to(DEVICE)
                outputs = model(inputs, mask, return_attention=False)

                # Full-sequence CE (aligned with training)
                outputs_reshaped = outputs.reshape(-1, outputs.shape[2])
                targets_reshaped = targets.reshape(-1)
                val_loss_full = criterion(outputs_reshaped, targets_reshaped)
                total_val_loss_full += val_loss_full.item()

                # Final-token CE (diagnostic) and accuracy
                logits_last = outputs[:, -1, :]
                labels_last = targets[:, -1]
                val_loss_last = criterion(logits_last, labels_last)
                total_val_loss_last += val_loss_last.item()

                preds = logits_last.argmax(dim=1)
                correct_predictions += (preds == labels_last).sum().item()
                total_predictions += preds.size(0)

                pc_corr, pc_total = _category_stats(preds, labels_last)
                per_class_correct = [sum(x) for x in zip(per_class_correct, pc_corr)]
                per_class_total = [sum(x) for x in zip(per_class_total, pc_total)]

                probs = torch.softmax(logits_last, dim=1)
                cat_probs = probs[:, 64:68].mean(dim=0)  # mean across batch
                mean_category_probs += cat_probs

        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_loss_full = total_val_loss_full / len(validation_loader) if len(validation_loader) > 0 else 0
        avg_val_loss_last = total_val_loss_last / len(validation_loader) if len(validation_loader) > 0 else 0
        val_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        mean_category_probs = (mean_category_probs / len(validation_loader)).cpu().tolist()

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]  Train Loss(full+aux): {avg_train_loss:.4f}  Val Loss(full): {avg_val_loss_full:.4f}  Val Loss(category-only): {avg_val_loss_last:.4f}  Val Acc(category): {val_accuracy:.4f}")
        print(f"  Mean predicted prob for category tokens [C1..C4]: {mean_category_probs}")
        print(f"  Per-class correct/total: {list(zip(per_class_correct, per_class_total))}")

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss_full)
        val_accuracies.append(val_accuracy)

        # Save best model by validation category accuracy (task metric)
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"   New best validation accuracy. Saved best model to {BEST_MODEL_PATH} (epoch {best_epoch}, val_acc={best_val_acc:.4f})")
            early_stop_counter = 0  # Reset patience counter when we improve
        else:
            early_stop_counter += 1
            print(f" No improvement. Early stopping counter: {early_stop_counter}/{EARLY_STOPPING_PATIENCE}")

        # Check for early stopping
        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}. No improvement for {EARLY_STOPPING_PATIENCE} epochs.")
            break

    print("Training complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    # --- Final test evaluation (load best checkpoint if available) ---
    if os.path.exists(BEST_MODEL_PATH):
        print(f"Loading best model from {BEST_MODEL_PATH} for final test evaluation...")
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
    else:
        print("No best model file found, using current model for testing.")

    def evaluate_test_set(model, criterion):
        test_dataset = WCST_Dataset("test_clean.txt")
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                seq_len = inputs.size(1)
                mask = mask_generate(seq_len).to(DEVICE)
                outputs = model(inputs, mask)

                # Full-sequence CE (same as training objective)
                outputs_reshaped = outputs.reshape(-1, outputs.shape[2])
                targets_reshaped = targets.reshape(-1)
                loss = criterion(outputs_reshaped, targets_reshaped)
                total_loss += loss.item()

                # Test accuracy (last token)
                logits_last = outputs[:, -1, :]
                labels_last = targets[:, -1]
                preds = logits_last.argmax(dim=1)
                correct += (preds == labels_last).sum().item()
                total += preds.size(0)

        test_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
        test_acc = correct / total if total > 0 else 0
        print(f"\n{'='*50}")
        print("FINAL TEST SET EVALUATION")
        print(f"Test Loss (full-sequence CE): {test_loss:.4f}")
        print(f"Test Accuracy (category): {test_acc:.4f}")
        print(f"{'='*50}\n")
        return test_loss, test_acc

    test_loss, test_acc = evaluate_test_set(model, criterion)
    print("Testing complete.")

    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_val_acc,
        'best_epoch': best_epoch,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'hyperparameters': {
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'd_model': D_MODEL,
            'num_layers': NUM_LAYERS,
            'num_heads': HEADS,
            'd_ff': D_FF,
            'dropout': DROPOUT,
            'aux_weight': AUX_WEIGHT,
            'optimizer': 'AdamW' if USE_ADAMW else 'Adam',
            'weight_decay': WEIGHT_DECAY
        }
    }

    # Write metrics for compare_models.py to read
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f" Saved training metrics to {METRICS_PATH}")

    # also save current model state 
    torch.save(model.state_dict(), "transformer_wcst_with_features_last.pth")
    print("Saved final model snapshot as transformer_wcst_with_features_last.pth")


if __name__ == "__main__":
    main()