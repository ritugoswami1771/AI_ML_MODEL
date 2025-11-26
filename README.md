SMS Spam Detection – AI/ML Project

This project focuses on detecting spam vs. ham (non-spam) SMS messages using Machine Learning (ML) and Natural Language Processing (NLP) techniques.
A trained model and a simple Flask-based web interface are included to allow users to test SMS messages in real-time.

1. Project Overview

Analyse SMS text messages
Clean and preprocess unstructured text data
Convert text into numerical representation using TF-IDF Vectorizer

Train and compare multiple ML models:
• Logistic Regression
• Linear Regression (for comparison)
• Random Forest
• Artificial Neural Network (ANN)

Evaluate model performance
Build a simple web interface for testing
Deployable Flask backend

2. Dataset Information

This project uses the publicly available SMS Spam Collection Dataset.
Dataset URL (Kaggle):
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Dataset Details:
Total messages: 5574
Labels:
• ham – legitimate SMS
• spam – fraudulent / promotional SMS
Columns:
• label – ham/spam
• message – SMS text
Dataset is moderately imbalanced (more ham than spam)
Widely used benchmark for spam detection and NLP research

3. Technologies Used
Python
NumPy
Pandas
Scikit-learn
NLTK
Flask
HTML / CSS
Jupyter Notebook or Google Colab

4. Files Included

File / Folder	Description
• app.py	Flask backend that loads the model & vectorizer and predicts spam/ham
• spam_model.pkl	Saved best performing ML model
• tfidf_vectorizer.pkl	TF-IDF vectorizer used to preprocess text
• templates/index.html	Frontend webpage for SMS input
• static/style.css	Styling for the webpage
• .vscode/	VS Code workspace configuration
• README.txt	Documentation file


5. Model Comparison (ANN vs Logistic Regression)
A sample SMS was used to compare model performance:

Sample Input:
"Congratulations! You have won a free lottery ticket. Click the link to claim your prize."
Results:
Logistic Regression: Detected as Spam (moderate confidence)
ANN: Detected as Spam with higher accuracy and confidence
Conclusion:
ANN performs better due to its ability to learn non-linear patterns in text data.

6. How to Run This Project
Step 1: Install requirements
pip install flask sklearn pandas numpy nltk
Step 2: Run the Flask application
python app.py
Step 3: Open Browser
Go to:
http://127.0.0.1:5000
Step 4: Enter SMS text and get prediction
The interface will show either:
“Spam”
“Ham” (not spam)

7. Future Improvements

Add deep learning (LSTM/GRU models)
Build a full UI/UX enhanced frontend
Deploy on Render / Railway / AWS
Add multi-language spam detection
Add SMS sender metadata analysis

8. Acknowledgements
Kaggle Dataset: SMS Spam Collection
UCI Machine Learning Repository
Scikit-learn and NLTK communities
Flask framework
