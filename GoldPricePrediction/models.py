from django.db import models

class GoldPrice(models.Model):
    date = models.DateField(primary_key=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)

    def __str__(self):
        return f"Gold Price on {self.date}: ${self.price}"

class NewsArticle(models.Model):
    title = models.CharField(max_length=255)
    url = models.URLField()
    url_to_image = models.URLField(null=True, blank=True)  # Optional image field
    published_at = models.DateField()  # Store as DateTimeField
    
    def __str__(self):
        return self.title