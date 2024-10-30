from django.contrib import admin
from .models import Review  # Import the Review model

@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = ('id', 'review_text', 'sentiment', 'created_at')  # Adjust based on your model fields
    search_fields = ('review_text', 'sentiment')  # Allows searching by these fields
    list_filter = ('sentiment', 'created_at')  # Allows filtering by these fields
    ordering = ('-created_at',)  # Orders by creation date, newest first

    def get_queryset(self, request):
        # Customizing the queryset if needed
        return super().get_queryset(request)
