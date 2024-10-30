import os
import json
import joblib
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

# Get the current directory of the file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained models and vectorizer with corrected paths
model_nb = joblib.load(os.path.join(BASE_DIR, '../sentiment_nb_model.pkl'))  # Naive Bayes model
model_lr = joblib.load(os.path.join(BASE_DIR, '../sentiment_lr_model.pkl'))  # Logistic Regression model
vectorizer = joblib.load(os.path.join(BASE_DIR, '../tfidf_vectorizer.pkl'))  # Vectorizer

def preprocess_text(text):
    """
    Preprocesses the input text: lowercasing, stripping, etc.
    """
    return text.lower().strip()  # Add further preprocessing as necessary

def home(request):
    """
    Renders the home page with the input form.
    """
    return render(request, 'review_classifier/home.html')

def index(request):
    return render(request, 'home.html')

@csrf_exempt
def predict_sentiment(request):
    """
    A view to handle sentiment prediction.
    Accepts both JSON and form data.
    """
    if request.method == 'POST':
        try:
            # Check if the request is JSON
            if request.content_type == 'application/json':
                body_unicode = request.body.decode('utf-8')
                body_data = json.loads(body_unicode)
                user_input = body_data.get('review_text', '')
            else:
                # Handle form data for non-JSON requests
                user_input = request.POST.get('review_text', '')

            if not user_input:
                return JsonResponse({'error': 'No review text provided'}, status=400)

            # Preprocess and vectorize the input
            processed_text = preprocess_text(user_input)
            vectorized_input = vectorizer.transform([processed_text])

            # Make predictions using both models
            prediction_nb = model_nb.predict(vectorized_input)[0]
            prediction_lr = model_lr.predict(vectorized_input)[0]

            response_data = {
                'naive_bayes_prediction': 'Positive' if prediction_nb == 1 else 'Negative',
                'logistic_regression_prediction': 'Positive' if prediction_lr == 1 else 'Negative'
            }

            # Return predictions based on request type
            if request.content_type == 'application/json':
                return JsonResponse(response_data)
            else:
                # Pass predictions to the result template
                return render(request, 'review_classifier/result.html', response_data)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON format'}, status=400)

    # Redirect to home page for GET requests
    return render(request, 'review_classifier/home.html')
