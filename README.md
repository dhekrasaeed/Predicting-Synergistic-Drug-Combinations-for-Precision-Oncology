# 🧬 MMDGNN: Multimodal Molecular Drug Graph Neural Network

### Predicting Synergistic Drug Combinations for Precision Oncology

[![Paper](https://img.shields.io/badge/Paper-BMC%20Bioinformatics-blue)](https://doi.org/10.1186/s12859-025-XXXXX)
[![Dataset](https://img.shields.io/badge/Dataset-ALMANAC%20%7C%20O%E2%80%99Neil%20%7C%20CLOUD%20%7C%20FORCINA-green)](https://databrowser.nci.nih.gov/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**Authors:**
Dhekra Saeed, Huanlai Xing, and Li Feng
School of Computer Science, Southwest Jiaotong University, Chengdu, China

---

## 📘 Overview

**MMDGNN** (Multimodal Molecular Drug Graph Neural Network) is a **scalable, data-centric deep learning framework** for predicting **cancer drug combination responses**.
Unlike traditional graph neural networks that assume molecular *homophily*, MMDGNN captures both **homophilic and heterophilic molecular relationships**, allowing more accurate modeling of chemical interactions between drugs.

This framework integrates **molecular fingerprints**, **SMILES strings**, and **cancer cell line features** within an **adaptive GNN** and **distributed training architecture**—enabling large-scale, high-throughput synergy prediction.

---

## 🚀 Key Features

* **Multimodal Representation Learning:** Combines fingerprints and SMILES-based molecular graphs.
* **Non-Homophilic GNN Design:** Captures both similar and contrasting atomic interactions.
* **Data-Centric Distributed Training:** Supports multi-GPU parallelization for large-scale datasets.
* **Enhanced Predictive Performance:** Outperforms models like DeepSynergy, PRODeepSyn, and MGAE-DC.
* **Multi-Metric Evaluation:** Supports Loewe, Bliss, HSA, and ZIP synergy scores.

---

## 🧠 Methodology

### Architecture

* Three **Graph Attention Network (GAT)** layers with attention-based message passing.
* **Global pooling strategies** (mean, max, sum) to aggregate molecular representations.
* Integration of **CCLE** features and **molecular fingerprints** via dense layers.
* Final synergy prediction through a fully connected regression network.

### Distributed Training

Implements a **Data-Centric Distributed Training (DCDT)** framework that parallelizes GNN training across multiple GPUs, improving speed and scalability for large datasets.

---

## 🧩 Model Architecture Diagram

Below is a conceptual overview of the MMDGNN pipeline. You can generate this figure using tools like **Graphviz** or **Matplotlib** if not included:

```text
┌────────────────────────────┐
│   Drug A (SMILES Graph)    │
└─────────────┬──────────────┘
              │  GAT Layers
┌─────────────▼──────────────┐
│   Molecular Embedding A     │
└─────────────┬──────────────┘
              │
              │
┌─────────────▼──────────────┐
│   Drug B (SMILES Graph)    │
└─────────────┬──────────────┘
              │  GAT Layers
┌─────────────▼──────────────┐
│   Molecular Embedding B     │
└─────────────┬──────────────┘
              │
        ┌─────▼─────┐
        │ Fusion +  │
        │ Fingerprint│
        │  + CCLE   │
        └─────┬─────┘
              │
     ┌────────▼────────┐
     │  Fully Connected │
     │ Regression Layer │
     └────────┬────────┘
              │
              ▼
       Synergy Prediction
```

---

## 📈 Training Performance Visualization

You can monitor model convergence and performance using **TensorBoard** or matplotlib.

### Example: Plotting Loss Curve

```python
import matplotlib.pyplot as plt

epochs = list(range(1, 501))
train_loss = [...]  # training loss per epoch
val_loss = [...]    # validation loss per epoch

plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Training Performance of MMDGNN')
plt.legend()
plt.grid(True)
plt.show()
```

📊 Example plot:

* The training curve should show smooth convergence after ~300 epochs.
* Validation loss stabilizes early, demonstrating effective generalization.

---

## 📊 Datasets

| Dataset     | Drug Pairs | Cell Lines | Source                                                                                       |
| ----------- | ---------- | ---------- | -------------------------------------------------------------------------------------------- |
| **O’Neil**  | 154,596    | 60         | [O’Neil et al., 2016](https://mct.aacrjournals.org/content/15/6/1155)                        |
| **ALMANAC** | 22,737     | 39         | [Holbeck et al., 2017](https://cancerres.aacrjournals.org/content/77/13/3564)                |
| **CLOUD**   | 29,278     | 1          | [Zhang et al., 2023](https://www.nature.com/articles/s42003-023-04741-5)                     |
| **FORCINA** | 757        | 1          | [Forcina et al., 2017](https://www.cell.com/cell-systems/fulltext/S2405-4712%2817%2930441-5) |

---

## 📈 Results

| Dataset | Metric | MGAE-DC                                       | **MMDGNN (Ours)** |
| ------- | ------ | --------------------------------------------- | ----------------- |
| Oneil   | Bliss  | MSE 17.36 → **16.18**, PCC 0.84 → **0.85**    |                   |
| Oneil   | Loewe  | MSE 162.21 → **152.78**, PCC 0.83 → **0.84**  |                   |
| Cloud   | ZIP    | MSE 323.43 → **317.85**, PCC 0.31 → **0.33**  |                   |
| ALMANAC | All    | Consistent improvement across synergy metrics |                   |

MMDGNN achieved **6.8% lower MSE** and **1.2% higher correlation** than MGAE-DC on average.

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/MMDGNN.git
cd MMDGNN
pip install -r requirements.txt
```

### Requirements

```
Python >= 3.8
PyTorch >= 1.12
DGL >= 1.1
RDKit >= 2023.3.1
scikit-learn
numpy
pandas
```

---

## 🏃 Usage

### Data Preprocessing

```bash
python preprocess_data.py --dataset almanac
```

### Training

```bash
python train.py --dataset almanac --epochs 500 --batch_size 32 --gpus 4
```

### Evaluation

```bash
python evaluate.py --dataset oneil --metric bliss
```

---

## 📦 Repository Structure

```
|--- data/               # Processed datasets
|--- models/             # GNN model definitions
|--- configs/            # Config files for experiments
|--- tools/              # Training/testing scripts
|--- results/            # Evaluation results
|--- saved_models/       # Pretrained models
|--- README.md
|--- LICENSE
```

---

## 🔬 Citation

If you use this work, please cite:

```bibtex
@article{saeed2025mmdgnn,
  title={A Scalable Multimodal Graph Neural Network for Drug Combination Response Prediction},
  author={Saeed, Dhekra and Xing, Huanlai and Feng, Li},
  journal={BMC Bioinformatics},
  year={2025}
}
```

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 📌 Links

* 📄 Paper: [DOI link](https://doi.org/10.1186/s12859-025-XXXXX) *(update once available)*
* 📊 Dataset: [NCI ALMANAC](https://databrowser.nci.nih.gov/)
* 🧠 Related Project: [MsGKD Repository](https://github.com/<your-username>/MsGKD)
