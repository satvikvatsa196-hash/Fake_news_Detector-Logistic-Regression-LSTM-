# 🔍 Fake News Detection System

A complete machine learning project for detecting misinformation using NLP, TF-IDF embeddings, and Logistic Regression — with a fully interactive Streamlit dashboard.

---

## 📁 Project Structure

```
fake_news_detector/
│
├── app.py                      ← Main Streamlit application (6 pages)
├── requirements.txt
│
├── data/
│   ├── __init__.py
│   └── generate_dataset.py     ← Synthetic real/fake news corpus generator
│
├── utils/
│   ├── __init__.py
│   ├── features.py             ← TF-IDF + linguistic feature engineering
│   └── train.py                ← Pipeline training, evaluation, persistence
│
└── models/                     ← Auto-created; stores lr_pipeline.pkl
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 📖 App Pages

| Page | What you'll learn |
|------|-------------------|
| 🏠 Overview | Dataset preview, label distribution, text length EDA |
| 🔬 Train Model | Train LR pipeline, see live metrics & CV scores |
| 🔍 Live Detector | Paste any article → real-time classification + feature breakdown |
| 📊 Model Evaluation | Confusion matrix, ROC curve, Precision-Recall curve, CV plot |
| 🧠 Feature Analysis | Top LR weights, linguistic feature distributions, TF-IDF demo |
| 📚 LSTM Theory | LSTM gates, architecture diagram, Keras implementation reference |

---

## 🧪 ML Pipeline

```
Raw Text
  │
  ├─ Clean: lowercase, remove HTML/URLs
  │
  ├─ TF-IDF word n-grams (1-2), max 30k features
  ├─ TF-IDF char n-grams (2-4), max 10k features     ─── FeatureUnion
  └─ Linguistic features (14 hand-crafted signals)
                │
    Logistic Regression (C=1.0, L2, balanced classes)
                │
           P(fake) ∈ [0, 1]
```

### Linguistic features extracted:
- Exclamation & question mark counts
- ALL-CAPS ratio & all-caps word count
- Average sentence & word length
- Vocabulary diversity (type/token ratio)
- Credibility term score (peer-reviewed, published, study…)
- Sensationalism term score (SHOCKING, deep state, whistleblower…)
- Number count, URL count, text length

---

## 📊 Expected Results

| Metric | Value |
|--------|-------|
| Accuracy | ~85–89% |
| Precision | ~86–90% |
| Recall | ~83–88% |
| F1 Score | ~84–89% |
| AUC-ROC | ~0.91–0.95 |

---

## 🔧 Using Real Data

Replace the synthetic dataset with the Kaggle Fake/Real News dataset:

```python
# Download from:
# https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

import pandas as pd
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")
fake["label"] = 1; real["label"] = 0
df = pd.concat([fake, real]).sample(frac=1).reset_index(drop=True)
df["full_text"] = df["title"] + " " + df["text"]
```

Then pass `df` to `train_and_evaluate(df)`.

---

## 📚 Key Concepts

### TF-IDF
```
TF(t,d)  = count(t in d) / total_words(d)
IDF(t)   = log(N / df(t)) + 1
TFIDF    = TF × IDF
```

### Logistic Regression
```
logit = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
P(fake) = σ(logit) = 1 / (1 + e^{-logit})
```

### LSTM Gates
```
Forget:  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input:   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Cell:    c_t = f_t ⊙ c_{t-1} + i_t ⊙ tanh(W_c · [h_{t-1}, x_t])
Output:  h_t = σ(W_o · [h_{t-1}, x_t]) ⊙ tanh(c_t)
```
