from django.db import models


class Article(models.Model):
    title = models.CharField(max_length=255)
    url = models.URLField()
    text = models.TextField()  # Ensure the field name matches the dataset

    def __str__(self):
        return self.title
