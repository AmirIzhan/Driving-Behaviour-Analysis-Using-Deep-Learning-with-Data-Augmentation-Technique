# Driving Behaviour Analysis Using Deep Learning with Data Augmentation Technique


Hi everyone!! 

This is my Final Year Project :)

Hybrid Conv1D-BiLSTM with CTGAN augmentation for driving behaviour classification (Normal,
Aggressive, Drowsy)  | UAH-DriveSet 
Uses CTGAN to generate synthetic samples and fix class imbalance.

**80.4% window-level accuracy** — outperforms 8 baseline models.

---

## Project Structure
```
driving-behaviour-analysis/
├── Augmentation/
│   └── train_ctgan.py          # CTGAN synthetic data generation
├── Baselines/
│   ├── train_svm_rf_adaboost_gb.py
│   ├── train_knn_logreg.py
│   ├── train_lstm.py
│   ├── train_bilstm.py
│   └── train_cnn_bigru.py
├── Data/
│   └── README.md               # Dataset download instructions
├── Model/
│   └── train.py                # Proposed hybrid model
├── Preprocessing/
│   ├── txt_to_csv.py
│   ├── interpolate.py
│   ├── feature_drop.py
│   ├── preprocessing_cleaning.py
│   └── split_data.py
├── .gitignore
├── requirements.txt
└── README.md
```
---

## Results

| Model                        | Accuracy |
|------------------------------|----------|
| SVM (Linear)                 | 41.49%   |
| AdaBoost                     | 29.14%   |
| Random Forest                | 38.90%   |
| Gradient Boosting            | 37.97%   |
| K-Nearest Neighbours         | 36.04%   |
| LSTM (stacked)               | 45.32%   |
| CNN-BiGRU                    | 48.31%   |
| Proposed CNN-BiLSTM          | 75.60%   |
| Proposed CNN-BiLSTM + CTGAN  | 80.40%   |

---

## Setup

git clone https://github.com/AmirIzhan/Driving-Behaviour-Analysis-Using-Deep-Learning-with-Data-Augmentation-Technique.git
cd driving-behaviour-analysis
pip install -r requirement.txt

Download the UAH-DriveSet dataset — see data/README.md for instructions.

## How to Run

### Step 1 — Preprocessing
```bash
python Preprocessing/txt_to_csv.py
python Preprocessing/interpolate.py
python Preprocessing/feature_drop.py
python Preprocessing/preprocessing_cleaning.py
python Preprocessing/split_data.py
```

### Step 2 — Augmentation
```bash
python Augmentation/train_ctgan.py
```

### Step 3 — Train Proposed Model
```bash
python Model/train.py
```

### Step 4 — Train Baselines
```bash
python Baselines/train_svm_rf_adaboost_gb.py
python Baselines/train_knn_logreg.py
python Baselines/train_lstm.py
python Baselines/train_bilstm.py
python Baselines/train_cnn_bigru.py
```

---

## Model Architecture

Dual-input hybrid model:
```
Branch 1 — Sequence (raw time-series):
  Input(timesteps=45, features)
  → LayerNormalization
  → SpatialDropout1D(0.10)
  → Conv1D(64, kernel_size=5, padding=same, relu)
  → BatchNormalization → Dropout(0.10)
  → BiLSTM(64, return_sequences=True,  dropout=0.15) → Dropout(0.15)
  → BiLSTM(32, return_sequences=False, dropout=0.10) → Dropout(0.10)

Branch 2 — Statistical features:
  Input(n_stats)
  → LayerNormalization
  → Dense(64, relu, l2=1e-4) → Dropout(0.10)

Fusion:
  Concatenate([Branch1, Branch2])
  → Dense(96, relu, l2=1e-4) → Dropout(0.15)
  → Dense(48, relu, l2=5e-5) → Dropout(0.10)
  → Dense(3,  softmax)
```

| Setting | Value |
|---|---|
| Loss | Sparse Categorical Crossentropy |
| Optimiser | AdamW (lr=3e-4, weight_decay=1e-4) |
| Class weight | Balanced per fold |
| Validation | GroupKFold (k=5) by trip_id |
| Window length | 45 |
| Step size | 15 |
| Purity threshold | 0.80 |

## Dataset

UAH-DriveSet — 6 drivers, accelerometer + GPS, 3 behaviour classes.
See Data/README.md for download instructions.

---

## Report

The full project report is available in the `Report/` folder.

[📄 View Full Report](Report/FYP%20REPORT.pdf)

The report covers:
- Literature review and comparison of existing driving behaviour systems
- Full data pipeline design and justification
- Model architecture design decisions
- CTGAN augmentation methodology
- Complete experimental results and analysis



---
## Author

Amir Izhan Bin Bekri | 2121507 | IIUM | January 2026
