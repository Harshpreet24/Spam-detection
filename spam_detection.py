# ============================================================
# TASK 1 - SPAM EMAIL DETECTION USING MACHINE LEARNING
# OutriX AI Internship
# ============================================================

# ============================================================
# STEP 1 - IMPORT LIBRARIES
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
import warnings
warnings.filterwarnings('ignore')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

# Download NLTK data (run once)
nltk.download('stopwords')
nltk.download('punkt')

print("✅ Libraries imported successfully!")


# ============================================================
# STEP 2 - LOAD DATASET
# ============================================================
# Download dataset from:
# https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
# Save the file as 'spam.csv' in the same folder as this script

df = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only the useful columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print("\n✅ Dataset loaded!")
print(f"Shape: {df.shape}")
print(df.head())
print("\nClass distribution:")
print(df['label'].value_counts())


# ============================================================
# STEP 3 - VISUALIZE CLASS DISTRIBUTION
# ============================================================
plt.figure(figsize=(6, 4))
df['label'].value_counts().plot(kind='bar', color=['steelblue', 'salmon'])
plt.title('Spam vs Ham Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()
print("✅ Class distribution chart saved!")


# ============================================================
# STEP 4 - TEXT PREPROCESSING
# ============================================================
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 3. Tokenize (split into words)
    words = text.split()
    # 4. Remove stopwords and apply stemming
    words = [ps.stem(word) for word in words if word not in stop_words]
    # 5. Join back into string
    return ' '.join(words)

df['cleaned_message'] = df['message'].apply(preprocess_text)

print("\n✅ Text preprocessing done!")
print("\nOriginal vs Cleaned:")
print("Original :", df['message'][0])
print("Cleaned  :", df['cleaned_message'][0])


# ============================================================
# STEP 5 - ENCODE LABELS
# ============================================================
# Convert 'spam' -> 1, 'ham' -> 0
df['label_encoded'] = df['label'].map({'spam': 1, 'ham': 0})
print("\n✅ Labels encoded! (spam=1, ham=0)")


# ============================================================
# STEP 6 - FEATURE EXTRACTION (TF-IDF)
# ============================================================
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df['cleaned_message'])
y = df['label_encoded']

print(f"\n✅ TF-IDF done! Feature matrix shape: {X.shape}")


# ============================================================
# STEP 7 - TRAIN/TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✅ Data split done!")
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")


# ============================================================
# STEP 8 - TRAIN MODEL 1: NAIVE BAYES
# ============================================================
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)

nb_accuracy = accuracy_score(y_test, nb_predictions)
print(f"\n✅ Naive Bayes Accuracy: {nb_accuracy * 100:.2f}%")
print("\nNaive Bayes Classification Report:")
print(classification_report(y_test, nb_predictions, target_names=['Ham', 'Spam']))


# ============================================================
# STEP 9 - TRAIN MODEL 2: SVM
# ============================================================
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"\n✅ SVM Accuracy: {svm_accuracy * 100:.2f}%")
print("\nSVM Classification Report:")
print(classification_report(y_test, svm_predictions, target_names=['Ham', 'Spam']))


# ============================================================
# STEP 10 - CONFUSION MATRIX VISUALIZATION
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Naive Bayes confusion matrix
cm_nb = confusion_matrix(y_test, nb_predictions)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
axes[0].set_title(f'Naive Bayes\nAccuracy: {nb_accuracy*100:.2f}%')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# SVM confusion matrix
cm_svm = confusion_matrix(y_test, svm_predictions)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
axes[1].set_title(f'SVM\nAccuracy: {svm_accuracy*100:.2f}%')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.show()
print("✅ Confusion matrix chart saved!")


# ============================================================
# STEP 11 - TEST WITH YOUR OWN EMAIL
# ============================================================
def predict_email(email_text, model=svm_model):
    cleaned = preprocess_text(email_text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    result = "🚨 SPAM" if prediction == 1 else "✅ HAM (Not Spam)"
    print(f"\nEmail: {email_text}")
    print(f"Prediction: {result}")
    return result

# Test examples
predict_email("Congratulations! You won a free iPhone. Click here to claim now!")
predict_email("Hey, are we still meeting tomorrow at 5pm?")
predict_email("URGENT: Your bank account has been compromised. Verify now!")
predict_email("Can you send me the notes from today's lecture?")


# ============================================================
# STEP 12 - MODEL COMPARISON BAR CHART
# ============================================================
models = ['Naive Bayes', 'SVM']
accuracies = [nb_accuracy * 100, svm_accuracy * 100]

plt.figure(figsize=(6, 4))
bars = plt.bar(models, accuracies, color=['steelblue', 'coral'], width=0.4)
plt.ylim(90, 100)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.5,
             f'{acc:.2f}%', ha='center', va='top', color='white', fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()
print("✅ Model comparison chart saved!")

print("\n🎉 TASK 1 COMPLETE! All outputs saved.")
print("Files generated:")
print("  - class_distribution.png")
print("  - confusion_matrices.png")
print("  - model_comparison.png")