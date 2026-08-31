# Federated Learning for Cybersecurity Threat Detection

A comprehensive federated learning system for collaborative cybersecurity threat detection across multiple organizations while preserving data privacy.  
This implementation supports three major network intrusion datasets: **CICIDS2017**, **UNSW-NB15**, and **Bot-IoT**.

---

## 🚀 Overview

This project implements a **professional-grade federated learning framework** that enables multiple organizations  
(financial institutions, tech companies, healthcare networks, government agencies, and educational institutions)  
to collaboratively train AI models for cybersecurity threat detection **without sharing raw data**.

---

## ✅ Key Features

- **Multi-Dataset Support:** CICIDS2017, UNSW-NB15, and Bot-IoT  
- **Differential Privacy:** Configurable ε and δ parameters  
- **Organization Modeling:** Realistic profiles for organization types  
- **Intelligent Aggregation:** Performance & data-quality weighted model fusion  
- **Professional Visualizations:** Publication-ready insights  
- **Comprehensive Metrics:** Accuracy, Precision, Recall, F1, AUC-ROC

---

## 🧠 Architecture

### Neural Network Design

| Layer Type   | Details                           |
|--------------|-----------------------------------|
| Input        | 50 features (post-selection)      |
| Hidden       | 256 → 128 → 64 (ReLU + BatchNorm) |
| Regularization | Dropout (0.3, 0.3, 0.2)         |
| Output       | Binary Classification (Benign vs Attack) |

---

## 🏢 Organization Profiles

| Organization       | Privacy Level | Data Quality | Attack Preferences                         |
|--------------------|--------------|--------------|--------------------------------------------|
| Financial Bank     | High         | 95%          | DDoS, Web Attack, Infiltration             |
| Tech Corporation   | Medium       | 90%          | PortScan, Bot, Web Attack                  |
| Healthcare Network | High         | 85%          | Infiltration, DoS Hulk, DDoS               |
| Government Agency  | Maximum      | 88%          | PortScan, Infiltration, Bot                |
| University Network | Low          | 80%          | DoS Hulk, Web Attack, DDoS                 |

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/federated-cybersecurity.git
cd federated-cybersecurity

# Install dependencies
pip install -r requirements.txt
```

---

## 📦 Requirements

```
numpy>=1.21.0
pandas>=1.3.0
torch>=1.9.0
kagglehub>=0.1.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.3.0
scikit-learn>=0.24.0
scipy>=1.7.0
```

---

## ▶️ Usage

```bash
# Running CICIDS2017 Experiment
python CICIDS-2017.py

# Running UNSW-NB15 Experiment
python UNSW.py
```

---

## ⚙️ Configuration

```python
config = FederatedConfig(
    n_organizations=5,
    global_rounds=20,
    local_epochs=3,
    learning_rate=0.001,
    batch_size=128,
    feature_selection_k=50,
    model_hidden_dims=[256, 128, 64],
    dp_epsilon=1.0,
    dp_delta=1e-5,
    random_seed=42
)
```

---

## 📊 Results

### Performance Across Datasets

| Dataset     | Final Accuracy | F1-Score | AUC-ROC | Training Time |
|-------------|----------------|----------|---------|---------------|
| CICIDS2017  | 90.3%          | 0.896    | 0.945   | 450s          |
| UNSW-NB15   | 89.7%          | 0.883    | 0.932   | 420s          |
| Bot-IoT     | 92.1%          | 0.915    | 0.958   | ~380s         |

### Organization Performance Gains

| Organization       | Accuracy Gain |
|--------------------|---------------|
| Financial Bank     | +15.8%        |
| Tech Corporation   | +18.2%        |
| Healthcare Network | +16.3%        |
| Government Agency  | +11.0%        |
| University Network | +18.9%        |

---

## 🔒 Privacy-Utility Tradeoff

| ε Value | Accuracy Loss | Privacy Strength |
|---------|---------------|------------------|
| 0.5     | ~2.5%         | Strong           |
| 1.0     | ~1.8% ✅ Recommended | Medium     |
| 2.0     | ~1.2%         | Weak             |

---

## 📂 Output Files

```
results_[experiment_name]_[timestamp]/
 ├── figures/
 │   ├── attack_distribution_analysis.png
 │   ├── organization_data_distribution.png
 │   ├── federated_training_convergence.png
 │   ├── confusion_matrices_analysis.png
 │   ├── roc_curves_comparison.png
 │   └── comprehensive_summary_report.png
 ├── comprehensive_experiment_results.json
 └── comprehensive_research_report.md
```

---

## 📚 Dataset References

- **CICIDS2017** – Canadian Institute for Cybersecurity  
- **UNSW-NB15** – University of New South Wales  
- **Bot-IoT** – IoT Network Intrusion Dataset

---

## 🔐 Privacy Mechanisms

- **Gradient Clipping (L2)**  
- **Gaussian Noise Addition**  
- **Cumulative Privacy Tracking (ε, δ)**

---

## 🧪 Research Contributions

✅ Realistic Organizational Modeling  
✅ Modern Threat Coverage  
✅ Privacy-Preserving Collaboration  
✅ Multi-Dataset Validation  
✅ Publication-Ready Visuals

---

## 📜 Citation

```bibtex
@software{federated_cybersecurity_2024,
  title={Federated Learning for Cybersecurity Threat Detection},
  author={Cybersecurity Research Team},
  year={2024},
  url={https://github.com/yourusername/federated-cybersecurity}
}
```

---

## 🤝 Contributing

1. Fork the repository  
2. Create a feature branch: `git checkout -b feature/improvement`  
3. Commit changes: `git commit -am 'Add feature'`  
4. Push: `git push origin feature/improvement`  
5. Create a Pull Request

---

## 🆘 Troubleshooting

| Issue | Solution |
|--------|---------|
| Kaggle Download Fails | `pip install --upgrade kagglehub` |
| Out of Memory | Reduce `batch_size` or `n_organizations` |
| Convergence Issues | Adjust `learning_rate` or increase `global_rounds` |

---

## 📬 Contact

- **Email:** mank8837@vandals.uidaho.edu  
- **Issues:** GitHub Issues Page

---

## 🙏 Acknowledgments

- Canadian Institute for Cybersecurity  
- University of New South Wales  
- Bot-IoT Dataset Contributors  
- Kaggle Hosting Infrastructure

---
