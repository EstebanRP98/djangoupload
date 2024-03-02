from django.http import JsonResponse
from .decorators import firebase_auth_required
from django.views.decorators.csrf import csrf_exempt
import json
from .mongodb_connector import MongoDBConnector
from django.core.files.storage import FileSystemStorage
from PyPDF2 import PdfReader
import io
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import pickle
import os
import secrets
import string
import openai
import numpy as np
import tiktoken
from datetime import datetime, timedelta
from dateutil import parser
from decouple import config
from .forms import DocumentoForm
import locale
import requests

mongodb_connector = MongoDBConnector()
os.environ["OPENAI_API_KEY"] = config('OPENAI_API_KEY')
openai.api_key = config('OPENAI_API_KEY')
token_whatsapp = config('TOKEN_WHATSAPP')
chat_history = []


@csrf_exempt
def verify_token(request):
    if request.method == 'GET':
        try:
            token = request.GET.get('hub.verify_token')
            challenge = request.GET.get('hub.challenge')
            access_token = "myaccestoken"
            if token == access_token:
                return challenge
            else:
                return JsonResponse({'message': 'Error Access'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'message': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'message': 'Invalid request'}, status=400)


@csrf_exempt
def received_message(request):
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            entry = body["entry"][0]
            changes = entry["changes"][0]
            value = changes["value"]
            message = value["messages"][0]
            text = message["text"]
            question_user = text["body"]
            number = message["from"]

            print("este es el texto: ", question_user)

            body_answer = plantilla_mensaje(question_user, number)
            send_message = whatsappService(body_answer)

            if send_message:
                return JsonResponse({'message': "Mensaje enviado correctamente"})
            else:
                return JsonResponse({'message': 'Error al enviar el mensaje'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'message': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'message': 'Invalid request'}, status=400)


def whatsappService(body):
    try:
        token = token_whatsapp
        api_url = "https://graph.facebook.com/v18.0/196839136856334/messages"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        response = requests.post(api_url,
                                 data=json.dumps(body),
                                 headers=headers
                                 )
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def plantilla_mensaje(text, number):
    body = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": number,
        "type": "text",
        "text": {
            "body": "esta es la respuesta a la pregunta: "+text
        }
    }
    return body
