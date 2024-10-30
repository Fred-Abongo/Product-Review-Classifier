from django.test import TestCase, Client
from django.urls import reverse
import json

class ReviewClassifierViewTests(TestCase):
    def setUp(self):
        # Set up a test client to interact with views
        self.client = Client()
        self.home_url = reverse('home')
        self.predict_url = reverse('predict_sentiment')

    def test_home_view(self):
        """
        Test that the home view loads successfully.
        """
        response = self.client.get(self.home_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'review_classifier/home.html')

    def test_predict_sentiment_view_post_json(self):
        """
        Test POST request with JSON data to predict sentiment.
        """
        data = {'review_text': 'This product is amazing!'}
        response = self.client.post(
            self.predict_url,
            json.dumps(data),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn('naive_bayes_prediction', response.json())
        self.assertIn('logistic_regression_prediction', response.json())

    def test_predict_sentiment_view_post_form(self):
        """
        Test POST request with form data to predict sentiment.
        """
        data = {'review_text': 'This product is terrible!'}
        response = self.client.post(self.predict_url, data)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'review_classifier/result.html')
        self.assertContains(response, 'Positive', status_code=200)
        self.assertContains(response, 'Negative', status_code=200)

    def test_predict_sentiment_view_no_data(self):
        """
        Test POST request with no review text provided.
        """
        response = self.client.post(self.predict_url, json.dumps({}), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()['error'], 'No review text provided')
