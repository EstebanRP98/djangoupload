from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .mongodb_connector import MongoDBConnector
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
import pickle
import os
import openai
from decouple import config
import requests
from . import views

mongodb_connector = MongoDBConnector()
os.environ["OPENAI_API_KEY"] = config('OPENAI_API_KEY')
openai.api_key = config('OPENAI_API_KEY')
token_whatsapp = config('TOKEN_WHATSAPP')
chat_history = []


@csrf_exempt
def received_message(request):
    if request.method == 'GET':
        try:
            token = request.GET.get('hub.verify_token')
            challenge = request.GET.get('hub.challenge')
            access_token = "myaccestoken"
            if token == access_token:
                return HttpResponse(challenge)  # Modifica esta línea
            else:
                return JsonResponse({'message': 'Error Access'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'message': 'Invalid JSON'}, status=400)
    elif request.method == 'POST':
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
            # answer = viewsAi.message_chatbot(question_user, number)
            answer = 'Hola'
            body_answer = plantilla_mensaje(answer, number)
            send_message = whatsappService(body_answer)
            if send_message:
                return JsonResponse({'message': "Mensaje enviado correctamente"})
            else:
                return JsonResponse({'message': 'Error al enviar el mensaje'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'message': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'message': 'Invalid request'}, status=400)


def message_chatbot(mensaje, numero):

    user_id = numero
    name = "EMPRE-info"
    query = mensaje

    views.agregar_mensaje("user", query, user_id)
    chat_history = views.obtener_conversacion(user_id)
    collection = mongodb_connector.get_collection('ventures')

    venture = collection.find_one({'name': name})

    with open(venture['path'], 'rb') as f:
        pdfsearch = pickle.load(f)

    if 'prompt_template' in venture and venture['prompt_template']:
        venture_prompt_value = venture['prompt_template']
        venture_promt = f"""{venture_prompt_value}"""
    else:
        venture_promt = """Eres un robot de preguntas y respuestas, Utiliza los siguientes elementos de contexto para responder a la pregunta del final. Responde siempre los nombres de los emprendimientos con el simbolo @.
Si no sabes la respuesta, solo responde 'Lo siento, no lo sé', no intentes inventarte una respuesta. Responde de forma corta y precisa."""

    prompt_template = venture_promt+"""

    {context}

    Question: {question}"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo"),
                                                  retriever=pdfsearch.as_retriever(
        search_kwargs={"k": 6}),
        combine_docs_chain_kwargs={
        "prompt": PROMPT},
        condense_question_llm=ChatOpenAI(
        temperature=0, model='gpt-3.5-turbo'),
        return_source_documents=False,)
    result = chain(
        {"question": query, 'chat_history': chat_history}, return_only_outputs=True)
    print(chat_history)
    views.agregar_mensaje("assistant", result["answer"], user_id)

    if venture:
        return result["answer"]
    else:
        return "lo siento por el momento no te puedo ayudar."


def whatsappService(body):
    try:
        token = token_whatsapp
        api_url = "https://graph.facebook.com/v18.0/252147934645534/messages"
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
            "body": text
        }
    }
    return body
