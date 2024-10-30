import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# NLTK download (only need to run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocess function (No need for custom tokenizer)
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in tokens])  # Returning preprocessed text as a string

def main():
    # Step 1: Load the dataset
    df = pd.read_csv('dataset/amazon.csv')
    print(df.head())  # Inspect the dataset

    # Step 2: Prepare features and labels
    X = df['Text']  # Adjust to your dataset's column name for reviews
    y = df['label']  # Adjust to your dataset's column name for sentiments

    # Step 3: Apply text preprocessing before vectorizing
    X_preprocessed = X.apply(preprocess_text)

    # Step 4: Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42, stratify=y)

    # Step 5: Feature extraction using TF-IDF without custom tokenizer
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Step 6: Train Naive Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vectorized, y_train)
    nb_predictions = nb_model.predict(X_test_vectorized)

    # Step 7: Train Logistic Regression model
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_vectorized, y_train)
    lr_predictions = lr_model.predict(X_test_vectorized)

    # Step 8: Evaluate Naive Bayes
    print("\nNaive Bayes Classification Report:")
    print(classification_report(y_test, nb_predictions))

    nb_accuracy = accuracy_score(y_test, nb_predictions)
    print(f'Naive Bayes Accuracy: {nb_accuracy:.2f}')

    print("Naive Bayes Confusion Matrix:")
    print(confusion_matrix(y_test, nb_predictions))

    # Step 9: Evaluate Logistic Regression
    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_test, lr_predictions))

    lr_accuracy = accuracy_score(y_test, lr_predictions)
    print(f'Logistic Regression Accuracy: {lr_accuracy:.2f}')

    print("Logistic Regression Confusion Matrix:")
    print(confusion_matrix(y_test, lr_predictions))

    # Step 10: Save both models and the vectorizer
    joblib.dump(nb_model, 'sentiment_nb_model.pkl')
    joblib.dump(lr_model, 'sentiment_lr_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')  # Corrected the vectorizer save

    print("\nModels and vectorizer saved successfully!")

if __name__ == '__main__':
    main()
