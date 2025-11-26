SMS Spam Detection Using Multiple Machine Learning Models

This project is an AI/ML-based SMS Spam Detection System that classifies SMS messages as Spam or Not Spam (Ham).
The project includes training, evaluating, and comparing multiple machine learning algorithms to identify the best-performing model for real-world spam filtering.

🗃️ Dataset: SMS Spam Collection Dataset

The dataset used is the SMS Spam Collection Dataset, originally from the UCI Machine Learning Repository, and available via Kaggle.

Dataset Details

Total Messages: 5,574

Categories:

ham → legitimate messages

spam → unsolicited / promotional / phishing SMS

Columns:

label – spam/ham

message – content of the SMS

Class Distribution: imbalanced (ham > spam)

Sources:

Ham messages from volunteers' personal SMS collections

Spam messages from spam reports, forums, and online archives

Common Use Case: research in NLP, spam filtering, text classification

🚀 Features of This Project

✔️ Trains multiple algorithms and compares them
✔️ Uses TF-IDF for text vectorization
✔️ Includes complete preprocessing pipeline (cleaning, tokenization, stopwords removal)
✔️ Exports the best-performing model as:

spam_model.pkl

tfidf_vectorizer.pkl
✔️ Can be integrated into:

Flask / Django Web App

FastAPI backend

Desktop application

Mobile app API

🧠 Machine Learning Models Used

The following algorithms were trained and evaluated:

Logistic Regression

Linear Regression (adapted for classification analysis)

Random Forest Classifier

Artificial Neural Network (ANN)

Naive Bayes (optional)

Support Vector Machine (optional)

Each model was compared using:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix
Downloading Dataset in Google Colab
# Install dependencies
!pip install kaggle pandas numpy scikit-learn nltk --quiet

# Upload kaggle.json
from google.colab import files
uploaded = files.upload()

# Set Kaggle credentials
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download dataset
!kaggle datasets download -d uciml/sms-spam-collection-dataset

# Unzip
!unzip -q sms-spam-collection-dataset.zip -d sms_spam_data

🧼 Data Preprocessing Steps

Convert all text to lowercase

Remove URLs, punctuation, symbols

Remove extra spaces

Remove stopwords

Apply TF-IDF vectorization

🏗️ Project Structure
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── ann_model.pkl
│   └── comparison_results.csv
├── spam_model.pkl
├── tfidf_vectorizer.pkl
├── app.py (optional web interface)
├── notebook.ipynb (training code)
├── README.md
└── requirements.txt

📌 How to Use the Model
import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

msg = ["Congratulations! You have won a prize."]
msg_tfidf = vectorizer.transform(msg)

prediction = model.predict(msg_tfidf)
print(prediction)


Output:
['spam']

🔧 Technologies Used

Python

Scikit-learn

Pandas / NumPy

NLTK

Joblib

Kaggle API

📈 Future Improvements

Integrate deep learning (LSTM, GRU models)

Build full web interface

Deploy on Render/Heroku

Add live API endpoint

🙌 Acknowledgements

UCI Machine Learning Repository

Kaggle Dataset Contributors

scikit-learn & NLTK communities
