# 🧬 MMDGNN: Multimodal Molecular Drug Graph Neural Network
Predicting Synergistic Drug Combinations for Precision Oncology
📘 Overview

MMDGNN (Multimodal Molecular Drug Graph Neural Network) is a scalable, data-centric deep learning framework for predicting cancer drug combination responses.
Unlike traditional graph neural networks that assume molecular homophily, MMDGNN captures both homophilic and heterophilic molecular relationships, allowing more accurate modeling of chemical interactions between drugs.

This framework integrates molecular fingerprints, SMILES strings, and cancer cell line features within an adaptive GNN and distributed training architecture—enabling large-scale, high-throughput synergy prediction.

🚀 Key Features

Multimodal Representation Learning: Combines fingerprints and SMILES-based molecular graphs.

Non-Homophilic GNN Design: Models both similar and contrasting atomic interactions.

Data-Centric Distributed Training: Scalable GPU-based parallel training for large biomedical datasets.

Enhanced Predictive Performance: Outperforms baseline models such as DeepSynergy, PRODeepSyn, and MGAE-DC.

Applicable to Multiple Synergy Metrics: Supports Loewe, Bliss, HSA, and ZIP synergy evaluation.

🧠 Methodology
Architecture

Three Graph Attention Network (GAT) layers with attention-based message passing.

Integration of global pooling strategies (mean, max, sum) for molecular representation.

Incorporation of CCLE (Cancer Cell Line Encyclopedia) and molecular fingerprint features.

Two-layer fully connected network for final synergy score prediction.

Distributed Training

Implements a Data-Centric Distributed Training (DCDT) algorithm to parallelize learning across GPUs, improving computational efficiency while maintaining model consistency.

📊 Datasets

MMDGNN was trained and evaluated on the following public benchmark datasets:

Dataset	Drug Pairs	Cell Lines	Source
O’Neil	154,596	60	O’Neil et al., 2016
ALMANAC	22,737	39	Holbeck et al., 2017
CLOUD	29,278	1	Zhang et al., 2023
FORCINA	757	1	Forcina et al., 2017
📈 Performance
Dataset	Metric	Baseline (MGAE-DC)	MMDGNN (Ours)
Oneil	Bliss	MSE 17.36 → 16.18, PCC 0.84 → 0.85
Oneil	Loewe	MSE 162.21 → 152.78, PCC 0.83 → 0.84
Cloud	ZIP	MSE 323.43 → 317.85, PCC 0.31 → 0.33
ALMANAC	All	Consistent improvement across all synergy metrics

MMDGNN achieved 6.8% lower MSE and 1.2% higher correlation compared to MGAE-DC on average.

⚙️ Requirements
Python >= 3.8
PyTorch >= 1.12
RDKit >= 2023.3.1
DGL >= 1.1
scikit-learn
numpy
pandas
🧬 Installation
git clone https://github.com/<your-username>/MMDGNN.git
cd MMDGNN
pip install -r requirements.txt
💡 Usage
1. Preprocess the dataset:
python preprocess_data.py --dataset almanac
2. Train the model:
python train.py --dataset almanac --epochs 500 --batch_size 32 --gpus 4
3. Evaluate performance:
python evaluate.py --dataset oneil --metric bliss
📚 Citation

If you use this repository in your research, please cite:

@article{Dhekra2025MMDGNN,
  title={A Scalable Multimodal Graph Neural Network for Drug Combination Response Prediction},
  author={Dhekra Saeed and Huanlai Xing and Li Feng},
  journal={BMC Bioinformatics},
  year={2025}
}
🤝 Contributing

Pull requests and collaborations are welcome!
Please open an issue first to discuss proposed changes.

📜 License

This project is licensed under the MIT License.
