# 📧 LogiSpamDetector - Spam Mail Prediction using Logistic Regression

A supervised machine learning project that classifies email/SMS messages as **Spam** or **Ham (Not Spam)** using Logistic Regression with TF-IDF feature extraction. The project covers the complete ML pipeline - from exploratory data analysis and preprocessing to model training, evaluation, and a live predictive system.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Dataset Description](#dataset-description)
4. [Project Workflow](#project-workflow)
5. [Tech Stack](#tech-stack)
6. [Project Structure](#project-structure)
7. [Getting Started](#getting-started)
8. [Model Performance](#model-performance)
9. [Sample Predictions](#sample-predictions)
10. [Key Insights](#key-insights)
11. [Author](#author)
12. [Contributing](#contributing)

---

## Project Overview

LogiSpamDetector is a text classification project that detects whether an email or SMS message is spam or legitimate (ham). It uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert raw message text into numerical feature vectors, and trains a **Logistic Regression** classifier on top of those features.

The project demonstrates a clean end-to-end machine learning workflow in Python using `scikit-learn`, `pandas`, `matplotlib`, and `seaborn` - suitable as a portfolio project or learning resource for NLP-based classification tasks.

---

## Features

- Full **EDA (Exploratory Data Analysis)** on the SMS spam dataset
- Null value detection, duplicate handling, and class distribution analysis
- **Bar chart** and **pie chart** visualization of spam vs. ham distribution
- **Label Encoding**: spam → 0, ham → 1
- **TF-IDF Vectorization** with English stop word removal and lowercasing
- **80/20 train-test split** with `random_state=3` for reproducibility
- Logistic Regression model training on TF-IDF feature vectors
- Accuracy evaluation on both training and test sets
- **Predictive system** for classifying new, custom email/SMS messages

---

## Dataset Description

| File | Description | Columns |
|---|---|---|
| `mail_data.csv` | Labeled SMS/email messages | `Category` (spam / ham), `Message` (raw text) |

**Dataset stats (after deduplication):**

- Total records: 5,572 (raw) → 5,157 (after removing 415 duplicates)
- Ham messages: 4,516
- Spam messages: 641
- Class distribution: ~87.6% ham, ~12.4% spam

**Column descriptions:**

- `Category` — The target label: `ham` (legitimate) or `spam` (unwanted). Encoded as `1` and `0` respectively for model training.
- `Message` — The raw text content of the SMS/email. Serves as the sole input feature for classification after TF-IDF transformation.

---

## Project Workflow

### 🟢 Step 1 — Data Loading & Overview
Load the dataset, inspect shape, data types, column names, null values, and unique class labels.

```python
df = pd.read_csv("mail_data.csv")
df.shape        # (5572, 2)
df.info()
df.isnull().sum()
df['Category'].unique()   # ['ham', 'spam']
```

---

### 🔵 Step 2 — Exploratory Data Analysis (EDA)

**Univariate Analysis — Category Column:**
- Count of spam vs. ham with `value_counts()`
- Bar chart visualizing class frequency
- Pie chart showing percentage distribution (~87.6% ham, ~12.4% spam)

---

### 🟡 Step 3 — Data Preprocessing

**Duplicate Handling:**
- 415 duplicate rows detected and removed (`drop_duplicates()`)
- Post-deduplication size: 5,157 rows

**Feature Engineering:**
- Strip and lowercase the `Category` column
- **Label Encoding**: `spam` → `0`, `ham` → `1`
- Split into feature matrix `X` (Message) and target vector `Y` (Category)

---

### 🟠 Step 4 — Train-Test Split

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
# X.shape: (5157,) → X_train: (4125,), X_test: (1032,)
```

---

### 🔴 Step 5 — Feature Extraction (TF-IDF)

Convert raw text into numerical vectors using `TfidfVectorizer`:

```python
feature_extraction = TfidfVectorizer(
    min_df=1,
    stop_words='english',
    lowercase=True
)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
```

- `fit_transform` on training data - learns vocabulary and transforms
- `transform` only on test data - prevents data leakage

---

### 🟣 Step 6 — Model Training

```python
model = LogisticRegression()
model.fit(X_train_features, Y_train)
```

---

### ⚫ Step 7 — Model Evaluation

```python
train_prediction = model.predict(X_train_features)
training_accuracy = accuracy_score(Y_train, train_prediction)

test_prediction = model.predict(X_test_features)
testing_accuracy = accuracy_score(Y_test, test_prediction)
```

---

### ⚪ Step 8 — Predictive System

Classify any new raw email/SMS text by transforming it through the same TF-IDF vectorizer and running model inference.

---

## Tech Stack

- **Language**: Python 3.10
- **Libraries**:
  - `pandas` — data loading, manipulation, and cleaning
  - `numpy` — numerical operations
  - `matplotlib` & `seaborn` — data visualization
  - `scikit-learn` — TF-IDF vectorization, Logistic Regression, train-test split, accuracy scoring
- **Environment**: Jupyter Notebook

---

## Project Structure

```
LogiSpamDetector/
│
├── README.md
│
└── Logistic Regression Project - Spam Mail Prediction/
    ├── Logistic Regression Project - Classifying Email as Spam or Not Spam.ipynb
    └── mail_data.csv
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/LogiSpamDetector.git
cd LogiSpamDetector
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3. Launch the notebook

```bash
jupyter notebook "Logistic Regression Project - Classifying Email as Spam or Not Spam.ipynb"
```

### 4. Run all cells in order

The notebook is fully self-contained. Run cells top to bottom to reproduce the full pipeline.

---

## Model Performance

| Split | Accuracy |
|---|---|
| Training Set | ~96.19% |
| Test Set | ~95.45% |

Training and test accuracy are close to each other and both near 96%, indicating the model is **neither overfitting nor underfitting** - a well-generalized classifier.

---

## Sample Predictions

**Ham (Legitimate) example:**
```
Input: "I've been searching for the right words to thank you for this breather.
        I promise I won't take your help for granted..."
Prediction: Ham mail ✅
```

**Spam example:**
```
Input: "You have WON $10,000 in our lucky draw. Click the link below to claim
        your prize NOW. Hurry! Offer valid for 24 hours only!"
Prediction: Spam mail 🚫
```

---

## Key Insights

- The dataset is **imbalanced** (~88% ham, ~12% spam), yet Logistic Regression with TF-IDF performs well without resampling
- **415 duplicate messages** (7.4% of the dataset) were cleaned before modeling — a critical preprocessing step
- TF-IDF effectively downweights common words and highlights distinguishing spam vocabulary (prize, win, free, click, etc.)
- Training and test accuracy being close (~96%) confirms the model generalizes well to unseen messages
- The predictive system can be easily extended into a Flask/FastAPI web app for real-time spam detection

---

## Author

Built as a Python machine learning portfolio project to demonstrate end-to-end NLP classification using Logistic Regression, TF-IDF feature extraction, and the scikit-learn ecosystem.

---

🤝 Contributing

Contributions are welcome! If you'd like to improve this project, feel free to fork the repository and submit a pull request 🚀

---

> ⭐ If you found this project useful, consider starring the repository!
