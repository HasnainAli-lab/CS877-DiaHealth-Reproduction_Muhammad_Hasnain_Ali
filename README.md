# CS877 Project: Diabetes Risk Prediction via DiaHealth Dataset (Reproduction & Novelty)

**Author:** Muhammad Hasnain Ali   
**Institution:** School of Electrical Engineering and Computer Science (SEECS), NUST Islamabad  
**Course:** CS877 – Artificial Intelligence & Machine Learning  
**Target Paper:** *Machine Learning-Based Diabetes Risk Prediction via DiaHealth Dataset with Explainable AI and Streamlit Deployment* (*Future Internet*, 2026, 18(6), 331, DOI: `10.3390/fi18060331`)

---

## Overview

This repository contains the complete reproduction code, leakage-free evaluation benchmarks, cost-sensitive threshold optimization, clinical noise/missing-value robustness tests, and Streamlit web application prototype for the DiaHealth diabetes risk prediction study.

### Key Findings
* **Headline Reproduction:** Reproduced the paper's tuned Random Forest accuracy of **94.80%** (Table 5), achieving **95.93%** on seed 42 and **95.42% ± 0.45%** across 15 random seeds (`0–14`).
* **Data Leakage Audit:** Applying SMOTE prior to the 80:20 train-test split leaks synthetic interpolations of test instances into training, inflating minority-class F1 from **37.09%** (`36.1% ± 4.6%` over 15 seeds) to **95.93%**.
* **Implemented Novelty:**
  1. **Leakage-Free Stratified Evaluation:** Strict train-only SMOTE/class-weighting evaluated on unseen real patients.
  2. **Out-of-Fold Cost-Sensitive Threshold Tuning:** Optimizing the probability cutoff inside training folds (`mean threshold = 0.22` vs. default `0.50`) boosts real-patient diabetic recall from **18.3% to 44.2%** and F1-score from **28.8% to 40.3%**.
  3. **Clinical Sensor Robustness Stress-Test:** Evaluates model stability under Gaussian measurement noise (up to `1.0x` std) and missing vital signs (up to `50%` missing, median-imputed).

---

## Repository Structure

```text
├── CS877_Colab_Notebook.ipynb   # Full reproduction, figures, novelty experiments, and Deliverable 2 log script
├── app.py                       # Interactive Streamlit clinical risk prototype
├── requirements.txt             # Python dependencies
├── .gitignore                   # Standard Python/Jupyter ignores
└── README.md                    # Project documentation and setup instructions
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Reproduction & Novelty Notebook
1. Download the **DiaHealth** dataset (`Diabetes_Final_Data_V2.csv`, Mendeley Data DOI: `10.17632/7m7555vgrn.1`).
2. Open `CS877_Colab_Notebook.ipynb` in Google Colab or Jupyter Notebook and run all cells sequentially (`Steps 0–6`).
3. Running `Step 3` generates `model.pkl` and the diagnostic figures (`figures/`).

### 3. Launch the Streamlit Prototype
Once `model.pkl` is generated in the project root:
```bash
streamlit run app.py
```
