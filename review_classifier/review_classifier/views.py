from django.http import JsonResponse
import joblib
import os
import json
from django.views.decorators.csrf import csrf_exempt
import nltk
from preprocessing import preprocess_text  # Ensure this file exists
from django.shortcuts import render

# Get the current directory of the file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and vectorizer with corrected paths
model_nb = joblib.load(os.path.join(BASE_DIR, '../sentiment_nb_model.pkl'))  # Naive Bayes model
model_lr = joblib.load(os.path.join(BASE_DIR, '../sentiment_lr_model.pkl'))  # Logistic Regression model
vectorizer_path = os.path.join(BASE_DIR, '../tfidf_vectorizer.pkl')
vectorizer = joblib.load(vectorizer_path)  # Ensure to use joblib.load consistently

# NLTK setup (ensure required packages are installed)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

@csrf_exempt
def predict_sentiment(request):
    if request.method == 'POST':
        # Get the user input from the request
        body_unicode = request.body.decode('utf-8')
        body_data = json.loads(body_unicode)
        user_input = body_data['review_text']

        # Preprocess the input text
        processed_text = preprocess_text(user_input)

        # Vectorize the input
        vectorized_input = vectorizer.transform([processed_text]) 

        # Predict the sentiment using both models
        prediction_nb = model_nb.predict(vectorized_input)[0]
        prediction_lr = model_lr.predict(vectorized_input)[0]

        # Prepare response data
        response_data = {
            'naive_bayes_prediction': 'Positive' if prediction_nb == 1 else 'Negative',
            'logistic_regression_prediction': 'Positive' if prediction_lr == 1 else 'Negative'
        }

        # Return a JSON response
        return JsonResponse(response_data)

    return JsonResponse({'error': 'Invalid request method, only POST is allowed.'})

