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

# chat
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool

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

# chat


def generate_data(request):
    if request.method == 'GET':
        # pdf_path_info = "/home/ubuntu/djangoupload/myapp/chat/pdf/Info.pdf"
        # pdf_path_suport = "/home/ubuntu/djangoupload/myapp/chat/pdf/Sleep_2.pdf"
        pdf_path_info = r"C:\Users\david\Documents\Python\ed_empre\myapp\chat\pdf\Info.pdf"
        pdf_path_suport = r"C:\Users\david\Documents\Python\ed_empre\myapp\chat\pdf\Sleep_2.pdf"
        loader = PyPDFLoader(pdf_path_info)
        kb_data = loader.load_and_split()
        loader = PyPDFLoader(pdf_path_suport)
        medium_data_split = loader.load_and_split()

        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002", chunk_size=1)

        sales_data = kb_data
        sales_store = Chroma.from_documents(
            sales_data, embeddings, persist_directory="./db/sales"
        )

        support_data = medium_data_split
        support_store = Chroma.from_documents(
            support_data, embeddings, persist_directory="./db/info"
        )
        return JsonResponse({'message': "ok"})
    else:
        return JsonResponse({'message': 'Invalid request'}, status=400)


@csrf_exempt
def chat_agent(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        query = data.get('question')
        data = generate_agent(query)
        return JsonResponse({'message': data['output']})
    else:
        return JsonResponse({'message': 'Invalid request'}, status=400)


def generate_agent(message):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", chunk_size=1)
    sales_store = Chroma(persist_directory="./sales",
                         embedding_function=embeddings)
    support_store = Chroma(persist_directory="./info",
                           embedding_function=embeddings)

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=openai.api_key,
        max_tokens=512,
    )
    sales_template = """Como bot de marketing de Sleep Haven, tu objetivo es proporcionar información precisa y útil sobre Sleep Haven,
    es una empresa de colchones de primera calidad que proporciona a sus clientes la experiencia de descanso más cómoda y confortable posible. Ofrecemos una gama de colchones, almohadas y accesorios de cama de alta calidad diseñados para satisfacer las necesidades únicas de nuestros clientes.
    Debes responder a las preguntas de los usuarios basándote en el contexto proporcionado y evitar inventar respuestas.
    Si no conoces la respuesta, simplemente di que no la conoces.
    Recuerde proporcionar información relevante sobre las características, ventajas y casos de uso de Sleep Haven.

    {context}

    Question: {question}"""
    SALES_PROMPT = PromptTemplate(
        template=sales_template, input_variables=["context", "question"]
    )
    sales_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=sales_store.as_retriever(),
        chain_type_kwargs={"prompt": SALES_PROMPT},
    )
    support_template = """
    Como Bot de Marketing de Sleep haven,
    Si saluda debes saludar muy respetuosamente, si se despide dile que puede visitar la pagina para comprar si tiene alguna duda.
    tu objetivo es detectar las siguientes intenciones, y dar como respuesta sus acciones:
    si el usuario quiere agendar una cita para comprar, respondele de forma muy contenta y amigable y enviale este link:
    - www.deroapp.com/agendar
    si el usuario quiere comprar directamente enviale el link de la pagina que es la siguiente:
    - www.deroappec.com/shop

    {context}

    Question: {question}"""

    SUPPORT_PROMPT = PromptTemplate(
        template=support_template, input_variables=["context", "question"]
    )

    support_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=support_store.as_retriever(),
        chain_type_kwargs={"prompt": SUPPORT_PROMPT},
    )

    meet_template = """Como bot de marketing de Sleep Haven, tu objetivo es saludar con el usuario,
    presentarte como Luis Borja que le puedes ayudar en cualquier pregunta que tenga.

    {context}

    Question: {question}"""
    MEET_PROMPT = PromptTemplate(
        template=meet_template, input_variables=["context", "question"]
    )
    meet_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=support_store.as_retriever(),
        chain_type_kwargs={"prompt": MEET_PROMPT},
    )

    # the zero-shot-react-description agent will use the "description" string to select tools
    tools = [
        Tool(
            name="sales",
            func=sales_qa.run,
            description="""útil para cuando un usuario está interesado en información diversa de los productos de Sleep haven."""
        ),
        Tool(
            name="info",
            func=support_qa.run,
            description="""útil para cuando el usuario quiere iniciar la conversacion, agendar o comprar."""
        ),
        Tool(
            name="general_qa",
            func=meet_qa.run,
            description="Útil para responder preguntas generales o específicas proporcionando respuestas detalladas y contextuales como Luis Borja, el bot de marketing."
        )
    ]

    agent = initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=True)

    data = agent.invoke(
        {"input": message},
        return_only_outputs=True,
    )

    return data
