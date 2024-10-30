from django.db import models

class Review(models.Model):
    id = models.AutoField(primary_key=True)  # Auto-generated ID
    review_text = models.TextField()  # Field for the review text
    created_at = models.DateTimeField(auto_now_add=True)  # Timestamp when the review was created
    updated_at = models.DateTimeField(auto_now=True)  # Timestamp for when the review was last updated
    sentiment = models.CharField(max_length=10, choices=[('Positive', 'Positive'), ('Negative', 'Negative')])  # Sentiment classification

    def __str__(self):
        return f"{self.review_text[:50]} - {self.sentiment}" 
