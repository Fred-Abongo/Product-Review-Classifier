import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string

# Download NLTK data (run only once if needed)
nltk.download('punkt')  # For tokenization
nltk.download('stopwords')  # For stopwords
nltk.download('wordnet')  # For lemmatization

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to clean the text
def preprocess_text(text):
    text = str(text).lower()  # Convert to lowercase and ensure it's a string
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize tokens
    return ' '.join(lemmatized_tokens)

# Load your CSV dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Apply the preprocessing function to the 'Text' column of your dataset
def preprocess_dataset(file_path):
    df = load_data(file_path)
    df['cleaned_review'] = df['Text'].apply(preprocess_text)
    return df

# Save the cleaned dataset to a new CSV file
def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)

# Example usage
if __name__ == "__main__":
    dataset_path = './dataset/amazon.csv'
    output_path = './dataset/preprocessed_reviews.csv'

    cleaned_df = preprocess_dataset(dataset_path)
    save_cleaned_data(cleaned_df, output_path)

    # Show cleaned dataset
    print(cleaned_df[['Text', 'cleaned_review']].head())
