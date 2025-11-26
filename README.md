# SMS Spam Detection -- AI/ML Project

This project focuses on detecting spam vs. ham (non-spam) SMS messages
using Machine Learning (ML) and Natural Language Processing (NLP). It
includes model training, evaluation, and a Flask-based web app for
real‑time SMS classification.

## 📌 Project Overview

-   Analyze and preprocess SMS text messages\
-   Convert text into TF‑IDF numerical vectors\
-   Train and compare multiple ML models:
    -   Logistic Regression\
    -   Linear Regression\
    -   Random Forest\
    -   Artificial Neural Network (ANN)\
-   Evaluate model performance\
-   Build a simple web interface\
-   Deployable Flask backend

------------------------------------------------------------------------

## 📊 Dataset Information

**Dataset:** SMS Spam Collection Dataset\
**Source:** Kaggle\
**URL:**
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

### Dataset Details

-   **Total messages:** 5,574\
-   **Labels:**
    -   `ham` -- normal SMS\
    -   `spam` -- fraudulent / promotional\
-   **Columns:**
    -   `label` -- spam/ham\
    -   `message` -- SMS text\
-   **Note:** Dataset is moderately imbalanced and widely used in NLP
    research.

------------------------------------------------------------------------

## 🛠 Technologies Used

-   Python\
-   NumPy\
-   Pandas\
-   Scikit‑learn\
-   NLTK\
-   Flask\
-   HTML / CSS\
-   Jupyter Notebook / Google Colab

------------------------------------------------------------------------

## 📁 Files Included

  File / Folder            Description
  ------------------------ -------------------------------------------------------
  `app.py`                 Flask backend for loading model & predicting spam/ham
  `spam_model.pkl`         Saved ML model
  `tfidf_vectorizer.pkl`   TF‑IDF vectorizer
  `templates/index.html`   Web UI for input
  `static/style.css`       UI styling
  `.vscode/`               VS Code configs
  `README.md`              Documentation

### 📂 Project Structure

    AL_ML_PROJECT/
    │── app.py
    │── spam_model.pkl
    │── tfidf_vectorizer.pkl
    │── static/
    │   └── style.css
    │── templates/
    │   └── index.html
    └── README.md

------------------------------------------------------------------------

## 🔬 Model Comparison (ANN vs Logistic Regression)

**Sample Input:**\
"Congratulations! You have won a free lottery ticket. Click the link to
claim your prize."

**Results:**\
- **Logistic Regression:** Detected as *Spam*\
- **ANN:** Detected as *Spam* with higher confidence

**Conclusion:**\
ANN performs better due to learning non‑linear text patterns.

------------------------------------------------------------------------

## ▶ How to Run This Project

### **Step 1: Install Dependencies**

    pip install flask sklearn pandas numpy nltk

### **Step 2: Run Flask App**

    python app.py

### **Step 3: Open Browser**

Visit:

    http://127.0.0.1:5000

### **Step 4: Test SMS Messages**

The app returns: - **Spam** - **Ham (Not Spam)**

------------------------------------------------------------------------

## 🚀 Future Improvements

-   Add deep learning models (LSTM/GRU)\
-   Enhanced UI/UX\
-   Deploy on Render/Railway/AWS\
-   Multi-language spam detection\
-   Analyze sender metadata

------------------------------------------------------------------------

## 🙏 Acknowledgements

-   Kaggle: SMS Spam Collection Dataset\
-   UCI Machine Learning Repository\
-   Scikit‑learn & NLTK\
-   Flask Framework
