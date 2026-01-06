#!/usr/bin/env python3
"""

This runs the same evaluation pipeline for two models:
 - Baseline Transformer (model.py)
 - Feature-based TransformerWithFeatures (model_improved.py)

This script performs:
 - load checkpoints & metrics JSONs
 - embedding analysis (PCA + silhouette)
 - attention visualization for a sample sequence
 - CIRCUIT ANALYSIS (head specialization identification)
 - ablation (no-pos, random embeddings) on each model
 - failure analysis (confusion matrix / per-class accuracy) on test set
 - something similar to in-context learning probing
 - saves plots under evaluation_plots/{baseline,features}/

How to use it:
  run [python evaluate_both.py] in the terminal
"""
import os
import json
import argparse
from collections import Counter
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix

from data_loader import WCST_Dataset
from torch.utils.data import DataLoader

from model import Transformer
from model_improved import TransformerWithFeatures

from visualizations import plot_tsne_umap_som

# reuse training config constants where applicable
from train import DEVICE, VOCAB_SIZE, D_MODEL, D_FF, NUM_LAYERS, HEADS, DROPOUT

# local utilities
PLOT_ROOT = "evaluation_plots"
os.makedirs(PLOT_ROOT, exist_ok=True)
sns.set_palette("husl")
plt.style.use("default")


def mask_generate(seq_len):
    m = torch.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:
                m[i, j] = 1
    return m.bool()


def save_plot(fig, filename, subdir):
    out_dir = os.path.join(PLOT_ROOT, subdir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved plot: {path}")
    return path


def load_checkpoint_for(model_cls, path):
    model = model_cls(vocab_size=VOCAB_SIZE, d_model=D_MODEL, d_ff=D_FF,
                      num_layers=NUM_LAYERS, num_heads=HEADS, dropout=DROPOUT).to(DEVICE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint missing: {path}")
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint {path}")
    return model


"""
Produce [64, d_model] numpy array representing the card token embeddings
in the way the model uses them 
"""
def extract_card_embeddings_consistent(model):

    emb_module = getattr(model, "embedding", None)
    if emb_module is None:
        raise RuntimeError("Model has no embedding attribute")

    with torch.no_grad():
        # baseline case: TokenEmbedding with token_embedding.weight
        if hasattr(emb_module, "token_embedding"):
            weight = emb_module.token_embedding.weight[:64].detach().cpu()
            scale = getattr(emb_module, "scaling", None)
            if scale is not None:
                try:
                    weight = weight * float(scale)
                except Exception:
                    pass
            return weight.numpy()

        # feature-based case: reconstruct from color/shape/quantity tables
        if all(hasattr(emb_module, a) for a in ("color_embed", "shape_embed", "quantity_embed")):
            c_w = emb_module.color_embed.weight.detach().cpu()
            s_w = emb_module.shape_embed.weight.detach().cpu()
            q_w = emb_module.quantity_embed.weight.detach().cpu()
            parts = []
            for idx in range(64):
                c = idx // 16
                s = (idx % 16) // 4
                q = idx % 4
                vec = torch.cat([c_w[c], s_w[s], q_w[q]], dim=-1)
                parts.append(vec.unsqueeze(0))
            emb = torch.cat(parts, dim=0)
            scale = getattr(emb_module, "scaling", None)
            if scale is not None:
                try:
                    emb = emb * float(scale)
                except Exception:
                    pass
            return emb.numpy()

        # in case, we can fallback: call embedding forward on tokens
        tokens = torch.arange(64, device=DEVICE).unsqueeze(1)
        out = emb_module(tokens)
        if out.dim() == 3:
            emb = out[:, 0, :].detach().cpu().numpy()
            return emb
        elif out.dim() == 2:
            return out.detach().cpu().numpy()
        else:
            raise RuntimeError(f"Unexpected embedding output shape: {out.shape}")


# This function performs embedding analysis and plotting PCA
def embedding_analysis_and_plots(model, name):
    emb = extract_card_embeddings_consistent(model)  # [64, d]
    colors = np.array([i // 16 for i in range(64)])
    shapes = np.array([(i % 16) // 4 for i in range(64)])
    quantities = np.array([i % 4 for i in range(64)])

    # PCA
    pca = PCA(n_components=2)
    emb2 = pca.fit_transform(emb)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # color
    for c in range(4):
        mask = colors == c
        axes[0].scatter(emb2[mask, 0], emb2[mask, 1], label=f"color{c}", alpha=0.7)
    axes[0].set_title(f"{name} Embeddings by Color")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # shape
    for s in range(4):
        mask = shapes == s
        axes[1].scatter(emb2[mask, 0], emb2[mask, 1], label=f"shape{s}", alpha=0.7)
    axes[1].set_title(f"{name} Embeddings by Shape")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # quantity
    for q in range(4):
        mask = quantities == q
        axes[2].scatter(emb2[mask, 0], emb2[mask, 1], label=f"qty{q}", alpha=0.7)
    axes[2].set_title(f"{name} Embeddings by Quantity")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    fig_path = save_plot(fig, "embeddings_pca.png", name)

    # silhouette scores
    color_s = silhouette_score(emb, colors)
    shape_s = silhouette_score(emb, shapes)
    qty_s = silhouette_score(emb, quantities)

    # Save numeric summary
    summary = {
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "silhouette": {
            "color": float(color_s),
            "shape": float(shape_s),
            "quantity": float(qty_s)
        }
    }
    with open(os.path.join(PLOT_ROOT, name, "embedding_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{name} silhouette scores: color={color_s:.3f}, shape={shape_s:.3f}, qty={qty_s:.3f}")

    # emb is the numpy array (64, d)
    labels_dict = {
        "color": colors,
        "shape": shapes,
        "quantity": quantities
    }
    out_dir = os.path.join(PLOT_ROOT, name)
    plot_tsne_umap_som(emb, labels_dict, out_dir, prefix="emb")
    return summary

# This function visualizes attention maps for a single sample and prints a per-head analysis
def visualize_attention_for_sample(model, name, sample_sequence):
    model.eval()
    token_names = []
    for token in sample_sequence:
        t = int(token.item())
        if t < 64:
            token_names.append(f"Card{t}")
        elif t == 68:
            token_names.append("SEP")
        elif t == 69:
            token_names.append("EOS")
        else:
            token_names.append(f"Cat{t-64}")

    with torch.no_grad():
        seq_len = sample_sequence.size(0)
        mask = mask_generate(seq_len).to(DEVICE)
        out, attn = model(sample_sequence.unsqueeze(0).to(DEVICE), mask=mask, return_attention=True)

    layers = [0, len(attn)-1]
    for layer_idx in layers:
        layer_att = attn[layer_idx][0].cpu().numpy()  # [heads, L, L]
        heads = layer_att.shape[0]
        cols = min(4, heads)
        rows = int(np.ceil(heads / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        axes = np.array(axes).reshape(-1)
        for h in range(heads):
            im = axes[h].imshow(layer_att[h], cmap="viridis", vmin=0, vmax=1, aspect="auto")
            axes[h].set_title(f"Layer{layer_idx} Head{h}")
            axes[h].set_xticks(range(len(token_names)))
            axes[h].set_yticks(range(len(token_names)))
            axes[h].set_xticklabels(token_names, rotation=45, ha="right")
            axes[h].set_yticklabels(token_names)
            plt.colorbar(im, ax=axes[h], shrink=0.6)
        # remove excess axes
        for ax in axes[heads:]:
            fig.delaxes(ax)
        fig.suptitle(f"{name} Attention Layer {layer_idx}")
        save_plot(fig, f"attention_layer_{layer_idx}.png", name)

        # CIRCUIT ANALYSIS: Print where each head attends when predicting the category
        print(f"\n{name} - Layer {layer_idx} Attention Analysis:")
        for head_idx in range(heads):
            att_map = layer_att[head_idx]
            # Find which token gets the most attention when predicting the category
            category_token_idx = len(token_names) - 1  # Last token
            attention_to_category = att_map[category_token_idx]
            most_attended_token_idx = np.argmax(attention_to_category)
            print(f"  Head {head_idx}: When predicting category, attends most to '{token_names[most_attended_token_idx]}'")

# This function performs circuit analysis by averaging attention patterns over many samples
def head_attention_stats(model, dataset, name, n_samples=1000, max_batches=None):

    print(f"\n{'='*60}")
    print(f"CIRCUIT ANALYSIS: {name} - Averaged Attention Patterns")
    print(f"{'='*60}")
    
    model.eval()
    loader = DataLoader(dataset, batch_size=64)
    accum = None
    counts = 0
    
    with torch.no_grad():
        for b_idx, (inp, tgt) in enumerate(loader):
            if counts >= n_samples:
                break
            inp = inp.to(DEVICE)
            mask = mask_generate(inp.size(1)).to(DEVICE)
            out, attn = model(inp, mask=mask, return_attention=True)
            
            # Convert to numpy and sum over batch
            for layer_idx, layer_att in enumerate(attn):
                # shape: [B, H, L, L]
                layer_np = layer_att.cpu().numpy().sum(axis=0)  # sum over batch -> [H, L, L]
                if accum is None:
                    accum = [None] * len(attn)
                if accum[layer_idx] is None:
                    accum[layer_idx] = layer_np
                else:
                    accum[layer_idx] += layer_np
            counts += inp.size(0)
            if max_batches and b_idx >= max_batches:
                break
    
    # Normalize
    for i in range(len(accum)):
        accum[i] = accum[i] / counts
    
    # Create circuit_probes subdirectory
    circuit_dir = os.path.join(PLOT_ROOT, name, "circuit_probes")
    os.makedirs(circuit_dir, exist_ok=True)
    
    # Save plots for each layer
    for layer_idx, arr in enumerate(accum):
        heads = arr.shape[0]
        rows = int(np.ceil(heads / 4))
        fig, axes = plt.subplots(rows, 4, figsize=(16, 3*rows))
        axes = np.array(axes).reshape(-1)
        
        for h in range(heads):
            im = axes[h].imshow(arr[h], cmap='viridis', vmin=0, vmax=arr.max())
            axes[h].set_title(f"Layer{layer_idx} Head{h}", fontsize=10)
            axes[h].set_xlabel("Attending to position")
            axes[h].set_ylabel("Attending from position")
            plt.colorbar(im, ax=axes[h], shrink=0.6)
        
        for ax in axes[heads:]:
            fig.delaxes(ax)
        
        plt.suptitle(f"{name} - Average Attention Patterns Layer {layer_idx}", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(circuit_dir, f"avg_attention_layer_{layer_idx}.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved: {save_path}")
    
    # IDENTIFY SPECIALIZED HEADS
    identify_specialized_heads(accum, name)
    
    return [arr.tolist() for arr in accum]

# This function identifies specialized heads based on averaged attention patterns
def identify_specialized_heads(avg_attention_stats, name):

    print(f"\n{'='*60}")
    print(f"HEAD SPECIALIZATION ANALYSIS: {name}")
    print(f"{'='*60}")
    
    # Sequence structure: [cat1, cat2, cat3, cat4, trial, SEP, ...]
    CATEGORY_POSITIONS = [0, 1, 2, 3]
    TRIAL_POSITION = 4
    SEP_POSITION = 5
    
    specializations = []
    
    for layer_idx, layer_attn in enumerate(avg_attention_stats):
        num_heads = layer_attn.shape[0]
        seq_len = layer_attn.shape[1]
        
        print(f"\n Layer {layer_idx}:")
        
        for head_idx in range(num_heads):
            # Attention from last position (where prediction happens)
            attn_from_last = layer_attn[head_idx, -1, :]  # [L]
            
            # Calculate attention to different regions
            attn_to_categories = attn_from_last[CATEGORY_POSITIONS].mean() if seq_len > max(CATEGORY_POSITIONS) else 0
            attn_to_trial = attn_from_last[TRIAL_POSITION] if TRIAL_POSITION < seq_len else 0
            attn_to_sep = attn_from_last[SEP_POSITION] if SEP_POSITION < seq_len else 0
            
            # Classify specialization
            max_attn = max(attn_to_categories, attn_to_trial, attn_to_sep)
            
            if max_attn == attn_to_categories and attn_to_categories > 0.3:
                spec_type = " CATEGORY-MATCHING"
                desc = f"Attends to category cards (avg: {attn_to_categories:.3f})"
            elif max_attn == attn_to_trial and attn_to_trial > 0.3:
                spec_type = " TRIAL-FOCUSED"
                desc = f"Attends to trial card ({attn_to_trial:.3f})"
            elif max_attn == attn_to_sep and attn_to_sep > 0.3:
                spec_type = " SEPARATOR"
                desc = f"Attends to SEP token ({attn_to_sep:.3f})"
            else:
                spec_type = " DISTRIBUTED"
                desc = f"No clear focus (cat:{attn_to_categories:.2f}, trial:{attn_to_trial:.2f})"
            
            specializations.append({
                'layer': layer_idx,
                'head': head_idx,
                'type': spec_type,
                'description': desc
            })
            
            # Print heads with strong specialization
            if max_attn > 0.3:
                print(f"  Head {head_idx}: {spec_type} - {desc}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {name} Head Specializations")
    print(f"{'='*60}")
    
    spec_types = {}
    for spec in specializations:
        spec_type = spec['type']
        if spec_type not in spec_types:
            spec_types[spec_type] = []
        spec_types[spec_type].append(f"L{spec['layer']}H{spec['head']}")
    
    for spec_type, heads in spec_types.items():
        if len(heads) > 0:
            print(f"\n{spec_type}:")
            print(f"  {', '.join(heads[:10])}")
            if len(heads) > 10:
                print(f"  ... and {len(heads)-10} more")
    
    return specializations

# This function performs an approximate ablation study on the model...
# We evaluate the model under 3 conditions: The full model, no positional encodings, and random embeddings to see how performance degrades/improves
def ablation(model, name, validation_loader):
    def eval_acc(m):
        m.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inp, tgt in validation_loader:
                inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
                mask = mask_generate(inp.size(1)).to(DEVICE)
                out = m(inp, mask)
                preds = out[:, -1, :].argmax(dim=1)
                correct += (preds == tgt[:, -1]).sum().item()
                total += preds.size(0)
        return correct/total if total>0 else 0.0

    baseline = eval_acc(model)
    results = {"Full Model": baseline}
    print(f"{name} baseline accuracy: {baseline:.4f}")

    # no positional enc 
    emb_mod = getattr(model, "embedding", None)
    if emb_mod is not None and hasattr(emb_mod, "positional_encoding"):
        orig_forward = emb_mod.forward

        def forward_no_pos(self, input_x):
            # mimic TokenEmbedding.forward but skip pos enc
            if hasattr(self, "token_embedding"):
                x = self.token_embedding(input_x) * getattr(self, "scaling", 1.0)
            else:
                return orig_forward(input_x)
            x = self.dropout(x)
            return x

        import types
        emb_mod.forward = types.MethodType(forward_no_pos, emb_mod)
        acc_no_pos = eval_acc(model)
        results["NoPos"] = acc_no_pos
        print(f"  Without positional encoding: {acc_no_pos:.4f} (drop {baseline - acc_no_pos:.4f})")
        emb_mod.forward = orig_forward
    else:
        results["NoPos"] = float("nan")
        print("  No positional-encoding monkeypatch available for this embedding type; skipping NoPos test.")

    # random embeddings: try to perturb token_embedding weights if available
    if emb_mod is not None and hasattr(emb_mod, "token_embedding"):
        orig_w = emb_mod.token_embedding.weight.data.clone()
        emb_mod.token_embedding.weight.data = torch.randn_like(orig_w)
        acc_rand = eval_acc(model)
        results["RandomEmb"] = acc_rand
        print(f"  With random embeddings: {acc_rand:.4f} (drop {baseline - acc_rand:.4f})")
        emb_mod.token_embedding.weight.data = orig_w
    else:
        results["RandomEmb"] = float("nan")
        print("  No token_embedding available to randomize; skipping RandomEmb test.")

    # save a small bar plot
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(results.keys())
    vals = [results[k] if not np.isnan(results[k]) else 0.0 for k in names]
    bars = ax.bar(names, vals, color=["#2ECC71", "#3498DB", "#E74C3C"])
    ax.set_ylim(0, max(vals)*1.1 if len(vals)>0 else 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Ablation - {name}")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}", ha="center")
    save_plot(fig, "ablation.png", name)
    return results

# This function prints confusion matrix and per-class accuracy on test set
def analyze_failures(model, name, test_loader):
    model.eval()
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for inp, tgt in test_loader:
            inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
            out = model(inp, mask_generate(inp.size(1)).to(DEVICE))
            preds = out[:, -1, :].argmax(dim=1).cpu().numpy()
            targets = tgt[:, -1].cpu().numpy()
            preds_all.extend(preds.tolist())
            targets_all.extend(targets.tolist())
    preds = np.array(preds_all)
    targets = np.array(targets_all)
    # distribution
    counter_pred = Counter(preds.tolist())
    counter_tgt = Counter(targets.tolist())
    print(f"{name} prediction distribution (token ids): {counter_pred}")
    # confusion on category indices (0-3)
    pred_c = preds - 64
    tgt_c = targets - 64
    cm = confusion_matrix(tgt_c, pred_c, labels=[0,1,2,3])
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["C0","C1","C2","C3"], yticklabels=["C0","C1","C2","C3"])
    ax.set_title(f"{name} Confusion Matrix (test)")
    save_plot(fig, "confusion_matrix.png", name)
    per_class_acc = {}
    overall = (preds == targets).mean()
    for c in range(4):
        mask = tgt_c == c
        per_class_acc[c] = float((pred_c[mask] == c).mean()) if mask.sum()>0 else float("nan")
    print(f"{name} overall acc: {overall:.4f}")
    return {"confusion": cm.tolist(), "per_class_acc": per_class_acc, "overall": float(overall)}


# This function tests the model on single trials only (no multi-trial concatenation).
# Essentially what we did was to see if the model can do in-context learning by just seeing more examples in the validation set...
# It isnt a full in-context learning test, but it gives some idea of how well the model can generalize to more examples.
def in_context_probe(model, name, validation_dataset):

    model.eval()
    
    sample_sizes = [50, 100, 200, 300, 500]
    accuracies = []
    
    for size in sample_sizes:
        correct = 0
        total = 0
        
        for i in range(min(size, len(validation_dataset))):
            inp, tgt = validation_dataset[i]
            
            with torch.no_grad():
                inp_batch = inp.unsqueeze(0).to(DEVICE)
                mask = mask_generate(inp.size(0)).to(DEVICE)
                out = model(inp_batch, mask)
                pred = out[0, -1, :].argmax().item()
                
                if pred == tgt[-1].item():
                    correct += 1
                total += 1
        
        acc = correct / total if total > 0 else 0.0
        accuracies.append(acc)
        print(f"{name} evaluation on {size} samples: {acc:.3f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(sample_sizes, accuracies, marker='o')
    ax.set_xlabel("Number of validation examples")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Consistency Check - {name}")
    ax.axhline(y=accuracies[0], color='gray', linestyle='--', alpha=0.5)
    save_plot(fig, "consistency_check.png", name)
    
    return {
        "sample_sizes": sample_sizes, 
        "accuracies": accuracies,
        "note": "Consistency check on single trials (model not trained for multi-trial context)"
    }

# This function loads training metrics from JSON and plots training/validation loss and accuracy curves
def plot_training_metrics(metrics_path, name):
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return None
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    # simple plotting: train/val loss and val accuracy
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
    epochs = range(1, len(metrics.get("train_losses", [])) + 1)
    ax1.plot(epochs, metrics.get("train_losses", []), label="train")
    ax1.plot(epochs, metrics.get("val_losses", []), label="val")
    ax1.set_title(f"{name} Loss Curves")
    ax1.legend(); ax1.grid(True)
    ax2.plot(epochs, metrics.get("val_accuracies", []), marker='o')
    ax2.axhline(0.25, color='gray', linestyle='--', label='chance')
    ax2.set_title(f"{name} Val Accuracy")
    ax2.legend(); ax2.grid(True)
    save_plot(fig, "training_curves.png", name)
    return metrics

# This function runs the full evaluation using all the above functions for a given model and its checkpoint
# So we can compare the two models 
def evaluate_model_suite(model_cls, checkpoint, metrics_json, dataset_files_prefix, name):

    model = load_checkpoint_for(model_cls, checkpoint)
    
    # Data loaders
    val_ds = WCST_Dataset(f"{dataset_files_prefix}_validation.txt") if os.path.exists(f"{dataset_files_prefix}_validation.txt") else WCST_Dataset("validation_clean.txt")
    test_ds = WCST_Dataset(f"{dataset_files_prefix}_test.txt") if os.path.exists(f"{dataset_files_prefix}_test.txt") else WCST_Dataset("test_clean.txt")
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    os.makedirs(os.path.join(PLOT_ROOT, name), exist_ok=True)

    # 1. Embedding analysis
    emb_summary = embedding_analysis_and_plots(model, name)

    # 2. Training metrics
    metrics = plot_training_metrics(metrics_json, name)

    # 3. Attention visualization 
    sample_inputs, _ = next(iter(val_loader))
    sample_seq = sample_inputs[0].cpu()
    try:
        visualize_attention_for_sample(model, name, sample_seq)
    except Exception as e:
        print(f"Attention visualization failed for {name}: {e}")

    # 4. CIRCUIT ANALYSIS 
    try:
        circuit_analysis = head_attention_stats(model, val_ds, name, n_samples=1000)
    except Exception as e:
        print(f"Circuit analysis failed for {name}: {e}")
        circuit_analysis = None

    # 5. Ablation
    ab = ablation(model, name, val_loader)

    # 6. Failure analysis
    fail = analyze_failures(model, name, test_loader)

    # 7. In-context probe
    ctx = in_context_probe(model, name, val_ds)

    return {
        "name": name,
        "emb_summary": emb_summary,
        "metrics": metrics,
        "circuit_analysis": circuit_analysis,  
        "ablation": ab,
        "failure": fail,
        "in_context": ctx
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-checkpoint", default="transformer_wcst_best.pth")
    parser.add_argument("--baseline-metrics", default="training_metrics.json")
    parser.add_argument("--features-checkpoint", default="transformer_wcst_with_features.pth")
    parser.add_argument("--features-metrics", default="training_metrics_features.json")
    args = parser.parse_args()

    print("🔬 Comparing Baseline vs Feature-based models")
    results = {}

    # Baseline
    try:
        results['baseline'] = evaluate_model_suite(
            Transformer,
            args.baseline_checkpoint,
            args.baseline_metrics,
            "train",
            "Baseline"
        )
    except Exception as e:
        print(f"Baseline evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    # Feature model
    try:
        results['features'] = evaluate_model_suite(
            TransformerWithFeatures,
            args.features_checkpoint,
            args.features_metrics,
            "train",
            "Features"
        )
    except Exception as e:
        print(f"Features evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    # Summarize comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON SUMMARY".center(70))
    print("="*70)
    
    for k in ['baseline', 'features']:
        if k in results and results[k] is not None:
            r = results[k]
            sil = r['emb_summary']['silhouette']
            fail = r['failure']
            
            print(f"\n{'─'*70}")
            print(f"  MODEL: {r['name'].upper()}")
            print(f"{'─'*70}")
            print(f"   Test Accuracy: {fail['overall']:.1%}")
            print(f"   Silhouette Scores:")
            print(f"     - Color:    {sil['color']:+.3f}")
            print(f"     - Shape:    {sil['shape']:+.3f}")
            print(f"     - Quantity: {sil['quantity']:+.3f}")
            print(f"     - Average:  {np.mean([sil['color'], sil['shape'], sil['quantity']]):+.3f}")
            
            if 'per_class_acc' in fail:
                print(f"  Per-Class Accuracy:")
                for c in range(4):
                    if c in fail['per_class_acc'] and not np.isnan(fail['per_class_acc'][c]):
                        print(f"     - Category {c}: {fail['per_class_acc'][c]:.1%}")
    
    # Compare the two models
    if 'baseline' in results and 'features' in results:
        baseline_acc = results['baseline']['failure']['overall']
        features_acc = results['features']['failure']['overall']
        improvement = features_acc - baseline_acc
        relative_improvement = (features_acc / baseline_acc - 1) * 100 if baseline_acc > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"IMPROVEMENT ANALYSIS".center(70))
        print(f"{'='*70}")
        print(f"  Absolute Improvement: {improvement:+.1%}")
        print(f"  Relative Improvement: {relative_improvement:+.1f}%")
        print(f"  Performance Ratio: {features_acc/baseline_acc:.2f}x" if baseline_acc > 0 else "  Performance Ratio: N/A")
        
        baseline_sil = results['baseline']['emb_summary']['silhouette']
        features_sil = results['features']['emb_summary']['silhouette']
        
        print(f"\n  Embedding Quality Improvement:")
        for attr in ['color', 'shape', 'quantity']:
            b = baseline_sil[attr]
            f = features_sil[attr]
            diff = f - b
            print(f"     {attr.capitalize():8s}: {b:+.3f} → {f:+.3f} (Δ = {diff:+.3f})")
    
    # Save combined results
    with open(os.path.join(PLOT_ROOT, "comparison_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f" EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f" Results saved to: {os.path.join(PLOT_ROOT, 'comparison_summary.json')}")
    print(f" All plots saved under: {os.path.abspath(PLOT_ROOT)}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()