# WCST Sorting Task with Transformer Architecture
### done by:  Keren Chetty (2548549),  Kgetja Bruce Mphekgwane (2593733) and Ivy Chepkwony (2431951)

## Overview
This project implements two **Transformer-based models** to perform the **Wisconsin Card Sorting Test (WCST)** as a next-token prediction task. The goal is to study how a Transformer can learn rule-based reasoning from synthetic card sequences and predict the correct category or card in the sequence. We compare a baseline model against a Feature-enhanced model.

The project includes:
- Training both **encoder-only Transformers** from scratch.
- Evaluating the models via embeddings, attention analysis, and in-context learning experiments.
- Conducting an ablation study to explore the impact of architectural choices.
- Generating plots and summaries for interpretability.

---

## 1. Project Structure
.
├── data_loader.py          # Dataset and DataLoader for WCST sequences
├── leakage_checker.py      # script to check for data leakage
├── fix_leakage.py          # script to fix any data leakage detected
├── model.py                # First Transformer encoder model definition
├── model_improved.py       # Second/improved Transformer encoder model definition       
├── train.py                # Training script for baseline model
├── train_improved.py       # Training script for second model
├── evaluation_2.py         # Evaluation and interpretability analyses
├── requirements.txt        # Python dependencies
├── train_data.txt          # Training sequences
├── validation_data.txt     # Validation sequences
├── test_data.txt           # Test sequences
├── validation_clean.txt    # Cleaned Validation sequences
├── test_clean.txt          # Cleaned Test sequences
├── evaluation_plots/       # Generated plots from evaluation
│   ├── Baseline/
│   │   ├── embeddings_pca.png
│   │   ├── emb_*_tsne.png
│   │   ├── emb_*_umap.png
│   │   ├── emb_*_som.png
│   │   ├── attention_layer_*.png
│   │   ├── circuit_probes/
│   │   │   └── avg_attention_layer_*.png
│   │   ├── ablation.png
│   │   ├── confusion_matrix.png
│   │   ├── consistency_check.png
│   │   └── training_curves.png
│   ├── Features/
│   │   └── (same structure as Baseline)
│   └── comparison_summary.json


---
## 3. Requirements

Python 3.9+ and the following packages:

- torch
- numpy
- matplotlib
- seaborn
- scikit-learn
- umap-learn
- minisom

Or install all dependencies via:

```bash
pip install -r requirements.txt
```

## 4. Main scripts to run

```bash
python generate_data.py
python leakage_checker.py
python fix_leakage.py
python train.py
python train_improved.py
python evaluation_2.py

```

## 5. Outputs

Plots: Saved in evaluation_plots/, organized by baseline or feature-modified models.
Summary JSON: comparison_summary.json contains key evaluation metrics across model variants.
Model Weights: transformer_wcst.pth for future inference or analysis.
