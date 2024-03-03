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
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
import argparse
from dataclasses import dataclass
from langchain_community.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
import spacy
from spacy.lang.es.examples import sentences
from spacy.training import Example
import random
from pathlib import Path
import os
import shutil
from langchain.document_loaders import DirectoryLoader
import PyPDF2  # o puedes usar PyPDF2
from sentence_transformers import SentenceTransformer
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

CHROMA_PATH = "chroma"
DATA_PATH = "/chat/data/pdf"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
Recuerda eres un chatbot, se cordial y saluda

{context}

---

Answer the question based on the above context: {question}
"""

mongodb_connector = MongoDBConnector()
os.environ["OPENAI_API_KEY"] = config('OPENAI_API_KEY')
openai.api_key = config('OPENAI_API_KEY')
chat_history = []
nlp = spacy.load("es_core_news_sm")


def dividir_en_segmentos(texto, longitud_segmento=500):
    palabras = texto.split()
    segmentos = [' '.join(palabras[i:i + longitud_segmento])
                 for i in range(0, len(palabras), longitud_segmento)]
    return segmentos


def encontrar_segmentos_relevantes(prompt_embedding, texto, n_segmentos=3):
    segmentos = dividir_en_segmentos(texto)
    segmentos_embeddings = np.array(
        [obtener_embeddings(segmento) for segmento in segmentos])

    # Calculando la similitud entre el prompt y cada segmento
    similitudes = cosine_similarity(
        prompt_embedding.reshape(1, -1), segmentos_embeddings)

    # Encontrando los índices de los n_segmentos más relevantes
    indices_relevantes = np.argsort(similitudes[0])[::-1][:n_segmentos]

    # Recuperando los segmentos más relevantes
    segmentos_relevantes = [segmentos[i] for i in indices_relevantes]
    return " ".join(segmentos_relevantes)


def chatbot_recomendador(pdf_path, prompt):
    texto_pdf = extraer_texto(pdf_path)
    client = OpenAI()
    # Obtener embeddings del prompt
    prompt_embedding = obtener_embeddings(prompt)

    # Encontrar los segmentos más relevantes del texto
    texto_relevante = encontrar_segmentos_relevantes(
        prompt_embedding, texto_pdf)

    # Pasar el texto relevante junto con el prompt al modelo GPT
    respuesta = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            # Contexto relevante como mensaje del sistema
            {"role": "system", "content": texto_relevante},
            # La pregunta o solicitud del usuario
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )

    # Accediendo a la respuesta
    texto_respuesta = respuesta.choices[0].message.content.strip()
    return texto_respuesta


def extraer_texto(pdf_path):
    texto = ""
    with open(pdf_path, "rb") as f:
        pdf = PyPDF2.PdfReader(f)
        for pagina in range(len(pdf.pages)):
            texto += pdf.pages[pagina].extract_text()
    return texto


modelo_embedding = SentenceTransformer('all-MiniLM-L6-v2')


def obtener_embeddings(texto):
    return modelo_embedding.encode(texto)


def chatbot_gpt(prompt, archivos_embeddings):
    # Aquí puedes integrar lógica para usar embeddings y determinar la relevancia con el prompt
    # Por simplificación, solo estamos pasando el prompt a GPT-3.5 directamente
    respuesta = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=150
    )
    return respuesta.choices[0].text.strip()


def message_chatbot(mensaje, numero):

    user_id = numero
    name = "EMPRE-info"
    query_text = mensaje

    # Prepare the DB.
    pdf_path = "/home/ubuntu/djangoupload/myapp/chat/pdf/EMPRE-info.pdf"
    # pdf_path = r"C:\Users\david\Documents\Python\ed_empre\myapp\chat\pdf\EMPRE-info.pdf"
    prompt = mensaje
    respuesta_chatbot = chatbot_recomendador(pdf_path, prompt)
    print(respuesta_chatbot)

    if respuesta_chatbot:
        return respuesta_chatbot
    else:
        return "lo siento por el momento no te puedo ayudar."


@csrf_exempt
def message_intention(request):
    if request.method == 'POST':
        # Carga el cuerpo de la solicitud JSON
        data = json.loads(request.body)
        user_message = data.get('message', '')

        # Prepare the DB.
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=CHROMA_PATH,
                    embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(user_message, k=3)
        if len(results) == 0 or results[0][1] < 0.7:
            print(f"Unable to find matching results.")
            return
        print("resultados list")
        print(len(results))
        context_text = "\n\n---\n\n".join(
            [doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(
            context=context_text, question=user_message)
        print(prompt)

        model = ChatOpenAI()
        response_text = model.predict(prompt)

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        print(formatted_response)
        return JsonResponse({"intent": response_text, })

    return JsonResponse({"error": "Invalid request"}, status=400)


@csrf_exempt
def message_products(request):
    if request.method == 'POST':
        # Carga el cuerpo de la solicitud JSON
        data = json.loads(request.body)
        user_message = data.get('message', '')
        current_directory = Path.cwd()
        output_dir = Path(current_directory) / "modelo-train"
        nlp = spacy.load(output_dir)

        # Procesamos el texto
        doc = nlp(user_message)

        # Extraemos las entidades
        # Inicializamos una lista para almacenar las entidades
        entities = []

        # Extraemos las entidades y las agregamos a la lista
        for ent in doc.ents:
            entities.append({"text": ent.text, "label": ent.label_})

        # Devolvemos las entidades como una respuesta JSON
        return JsonResponse({"entities": entities})

    return JsonResponse({"error": "Invalid request"}, status=400)


TRAIN_DATA = [
    ("Quiero 2 pizzas y 3 refrescos", {"entities": [
     (7, 8, "CANTIDAD"), (9, 15, "ITEM"), (19, 20, "CANTIDAD"), (21, 29, "ITEM")]}),
    ("Me gustaría una hamburguesa y dos cervezas", {"entities": [
     (12, 15, "CANTIDAD"), (16, 27, "ITEM"), (32, 35, "CANTIDAD"), (36, 44, "ITEM")]}),
    ("Ponme 4 tacos y 1 soda", {"entities": [
     (6, 7, "CANTIDAD"), (8, 13, "ITEM"), (17, 18, "CANTIDAD"), (19, 23, "ITEM")]}),
    ("Deseo 3 empanadas y una jarra de agua", {"entities": [
     (6, 7, "CANTIDAD"), (8, 17, "ITEM"), (22, 25, "CANTIDAD"), (26, 41, "ITEM")]}),
    ("Añade al pedido 5 sándwiches y 2 tés", {"entities": [
     (16, 17, "CANTIDAD"), (18, 28, "ITEM"), (33, 34, "CANTIDAD"), (35, 38, "ITEM")]}),
    ("Quisiera pedir una pizza margarita y dos cocacolas", {
        "entities": [(18, 21, "CANTIDAD"), (22, 36, "ITEM"), (41, 44, "CANTIDAD"), (45, 54, "ITEM")]
    }),
    ("Necesito 3 ensaladas y 1 limonada", {"entities": [
     (10, 11, "CANTIDAD"), (12, 21, "ITEM"), (25, 26, "CANTIDAD"), (27, 35, "ITEM")]}),
    ("Por favor, tráeme 2 porciones de pastel y 4 cafés", {"entities": [
     (18, 19, "CANTIDAD"), (20, 33, "ITEM"), (38, 39, "CANTIDAD"), (40, 45, "ITEM")]}),
    ("Voy a querer 5 hamburguesas con papas fritas", {"entities": [
     (12, 13, "CANTIDAD"), (14, 25, "ITEM"), (29, 30, "CANTIDAD"), (31, 42, "ITEM")]}),
    ("Agrega 2 platos de pasta y 3 botellas de vino", {"entities": [
     (7, 8, "CANTIDAD"), (9, 20, "ITEM"), (24, 25, "CANTIDAD"), (26, 42, "ITEM")]}),
    ("Quiero ordenar 4 helados y 1 batido de frutas", {"entities": [
     (13, 14, "CANTIDAD"), (15, 21, "ITEM"), (25, 26, "CANTIDAD"), (27, 44, "ITEM")]}),
    ("Hazme 3 burritos y 2 jugos naturales", {"entities": [
     (5, 6, "CANTIDAD"), (7, 14, "ITEM"), (18, 19, "CANTIDAD"), (20, 33, "ITEM")]}),
    ("Voy a pedir 6 porciones de sushi y 1 té helado", {"entities": [
     (11, 12, "CANTIDAD"), (13, 27, "ITEM"), (31, 32, "CANTIDAD"), (33, 42, "ITEM")]}),
    ("Quiero 4 bolsas de papas fritas y 3 botellas de jugo", {"entities": [
     (7, 8, "CANTIDAD"), (9, 28, "ITEM"), (32, 33, "CANTIDAD"), (34, 46, "ITEM")]}),
    ("Añade a mi pedido 5 tacos al pastor y 2 aguas frescas", {"entities": [
     (17, 18, "CANTIDAD"), (19, 34, "ITEM"), (39, 40, "CANTIDAD"), (41, 54, "ITEM")]}),
    ("Quisiera dos hamburguesas y tres cocacolas", {"entities": [
     (10, 12, "CANTIDAD"), (13, 23, "ITEM"), (28, 29, "CANTIDAD"), (30, 39, "ITEM")]}),
    ("Dame una hamburguesa con queso y una cocacola grande", {"entities": [
     (5, 6, "CANTIDAD"), (7, 24, "ITEM"), (29, 30, "CANTIDAD"), (31, 46, "ITEM")]}),
    ("Pido tres hamburguesas dobles y dos cocacolas light", {"entities": [
     (5, 6, "CANTIDAD"), (7, 20, "ITEM"), (28, 29, "CANTIDAD"), (30, 46, "ITEM")]}),
    # Puedes añadir más ejemplos aquí según sea necesario
]


@csrf_exempt
def entrenar_modelo_spicy(request):
    if request.method == 'POST':
        # Carga el cuerpo de la solicitud JSON
        data = json.loads(request.body)
        nlp = spacy.blank("es")  # Usar "es" para español

        # Crea un nuevo pipeline 'ner' y lo añade al pipeline si no existe
        if "ner" not in nlp.pipe_names:
            ner = nlp.create_pipe("ner")
            nlp.add_pipe("ner")
        else:
            ner = nlp.get_pipe("ner")

        # Añade las nuevas etiquetas al entity recognizer
        ner.add_label("ITEM")
        ner.add_label("CANTIDAD")

        # Deshabilita otros componentes del pipeline para entrenar solo NER
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

        # Entrenamiento del modelo
        with nlp.disable_pipes(*other_pipes):  # Entrena solo NER
            optimizer = nlp.begin_training()
            for itn in range(10):  # 10 iteraciones
                random.shuffle(TRAIN_DATA)
                losses = {}
                for text, annotations in TRAIN_DATA:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    nlp.update([example], drop=0.5,
                               losses=losses, sgd=optimizer)
                print(losses)
        current_directory = Path.cwd()
        output_dir = Path(current_directory) / "modelo-train"
        nlp.to_disk(output_dir)
        print(f"Modelo guardado en: {output_dir}")
        return JsonResponse({"intent": "ok", })

    return JsonResponse({"error": "Invalid request"}, status=400)


def word_to_number(word):
    # Mapea las palabras a sus equivalentes numéricos
    numbers = {
        "uno": 1,
        "dos": 2,
        "tres": 3
        # Puedes agregar más mapeos aquí según sea necesario
    }
    return numbers.get(word, None)
