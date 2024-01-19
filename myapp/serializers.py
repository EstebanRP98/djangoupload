from rest_framework import serializers
from .models import Archivo


class ArchivoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Archivo
        fields = ['archivo']
