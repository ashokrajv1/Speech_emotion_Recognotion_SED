from django.db import models

# Create your models here.

class Audio(models.Model):
    audio_file = models.FileField(null=True, blank=True)
    #transcript = models.TextField(null=True, blank=True)