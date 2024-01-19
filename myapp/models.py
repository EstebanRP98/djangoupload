from django.db import models

# Create your models here.


class venture(models.Model):
    venture_id = models.CharField(max_length=10)
    name = models.CharField(max_length=100)
    path = models.FilePathField()
    prompt_template = models.TextField()


class Documento(models.Model):
    archivo = models.FileField(upload_to='archivo/')
