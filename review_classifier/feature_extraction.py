import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def load_data(file_path):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def extract_features_tfidf(df):
    """Extract features using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words='english')
    X_tfidf = vectorizer.fit_transform(df['Text'])
    return X_tfidf, vectorizer

def extract_features_bow(df):
    """Extract features using Bag of Words."""
    vectorizer = CountVectorizer(stop_words='english')
    X_bow = vectorizer.fit_transform(df['Text'])
    return X_bow, vectorizer

def main():
    file_path = 'dataset/amazon.csv'
    
    # Load data
    df = load_data(file_path)
    
    # Extract features using TF-IDF
    X_tfidf, tfidf_vectorizer = extract_features_tfidf(df)
    print(f"TF-IDF feature shape: {X_tfidf.shape}")

    # Extract features using Bag of Words
    X_bow, bow_vectorizer = extract_features_bow(df)
    print(f"Bag of Words feature shape: {X_bow.shape}")

if __name__ == "__main__":
    main()
