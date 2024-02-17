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
from datetime import datetime, timedelta
from dateutil import parser
from decouple import config
from .forms import DocumentoForm
import locale

mongodb_connector = MongoDBConnector()
os.environ["OPENAI_API_KEY"] = config('OPENAI_API_KEY')
chat_history = []


@csrf_exempt
def save_or_update_venture(request, venture_id=None):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            name = data.get('name')
            path = data.get('path')
            prompt_template = data.get('prompt_template')  # Nuevo campo

            collection = mongodb_connector.get_collection('ventures')

            venture_data = {'name': name, 'path': path}

            # Agregar el campo prompt_template si se proporciona
            if prompt_template:
                venture_data['prompt_template'] = prompt_template

            if venture_id and prompt_template:
                venture = collection.find_one({'venture_id': venture_id})
                venture['prompt_template'] = prompt_template
                # Actualizar el objeto venture si se proporciona un ID
                collection.update_one({'venture_id': venture_id}, {
                                      '$set': venture})
                return JsonResponse({'message': 'Venture updated successfully', 'venture_id': str(venture_id)})
            else:
                # Crear un nuevo objeto venture
                venture_id = collection.insert_one(venture_data).inserted_id
                return JsonResponse({'message': 'Venture saved successfully', 'venture_id': str(venture_id)})
        except json.JSONDecodeError:
            return JsonResponse({'message': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'message': 'Invalid request'}, status=400)


@csrf_exempt
def get_chat_by_user_id(request):
    if request.method == 'GET':
        user_id = request.GET.get('user_id')
        if user_id:
            collection_chat = mongodb_connector.get_collection('chat')
            # Buscar el chat del usuario por su user_id
            chat_user = collection_chat.find_one({'user_id': user_id})

            if chat_user:
                # Obtener la lista de mensajes del chat
                chat_history = chat_user.get('chat_history', [])

                # Ordenar los mensajes del chat del más antiguo al más actual
                chat_history.sort(key=lambda x: x.get('timestamp', ''))

                # Limitar a los últimos 10 mensajes
                chat_history = chat_history[-20:]

                # Convierte el ObjectId a una cadena antes de serializarlo
                chat_user['_id'] = str(chat_user['_id'])

                # Actualiza la lista de mensajes en el documento del usuario
                chat_user['chat_history'] = chat_history
                return JsonResponse(chat_user, status=200)
            else:
                return JsonResponse({'message': 'El usuario no tiene un chat'}, status=404)
        else:
            return JsonResponse({'message': 'ID de usuario no proporcionado'}, status=400)
    else:
        return JsonResponse({'message': 'Invalid request'}, status=400)


@csrf_exempt
def get_venture_by_name(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_id = data.get('user_id')
        name = data.get('name')
        query = data.get('question')

        agregar_mensaje("user", query, user_id)
        chat_history = obtener_conversacion(user_id)
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
        agregar_mensaje("assistant", result["answer"], user_id)

        if venture:
            return JsonResponse({'name': venture['name'], 'answer': result["answer"]})
        else:
            return JsonResponse({'message': 'Venture not found'}, status=404)
    else:
        return JsonResponse({'message': 'Invalid request'}, status=400)


@csrf_exempt
def upload_pdf(request):
    if request.method == 'POST' and request.FILES['file']:
        pdf_file = request.FILES['file']
        filename_without_extension, _ = os.path.splitext(pdf_file.name)
        doc_reader = PdfReader(pdf_file)
        # read data from the file and put them into a variable called raw_text
        raw_text = ''
        for i, page in enumerate(doc_reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
        # Splitting up the text into smaller chunks for indexing
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,  # striding over the text
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)
        # Download embeddings from OpenAI
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)
        ruta_completa = os.path.join(
            os.getcwd()+"/media", filename_without_extension + ".pkl")
        with open(ruta_completa, 'wb') as f:
            pickle.dump(docsearch, f)
        id_generado = generar_id_aleatorio()
        # Guardar en MongoDB
        mongodb_connector = MongoDBConnector()
        collection = mongodb_connector.get_collection('ventures')
        venture_id = collection.insert_one(
            {'name': filename_without_extension,
             'path': ruta_completa,
             'venture_id': id_generado
             }).inserted_id

        return JsonResponse({'message': 'PDF uploaded and venture saved successfully', 'venture_id': str(venture_id)})
    else:
        return JsonResponse({'message': 'Invalid request'}, status=400)


def generar_id_aleatorio():
    # Definir los caracteres válidos para el ID (números y letras)
    caracteres_validos = string.ascii_letters + string.digits

    # Generar un ID aleatorio con la longitud especificada
    id_aleatorio = ''.join(secrets.choice(caracteres_validos)
                           for _ in range(10))

    return id_aleatorio


def agregar_mensaje(role_chat, message, user_id):
    collection_chat = mongodb_connector.get_collection('chat')
    # Buscar al usuario por su user_id
    usuario = collection_chat.find_one({'user_id': user_id})

    if not usuario:
        # Si el usuario no existe, crear uno nuevo con su user_id
        usuario = {'user_id': user_id, 'chat_history': []}

    role = role_chat  # El rol del mensaje (user, assistant, etc.)
    content = message  # El contenido del mensaje
    timestamp = datetime.now()  # Fecha y hora actual

    if role and content:
        # Crear un mensaje con el rol y el contenido
        mensaje = {'role': role, 'message': content, 'timestamp': timestamp}

        # Agregar el mensaje al historial de chat del usuario
        usuario['chat_history'].append(mensaje)

        # Actualizar o insertar el usuario en la colección
        collection_chat.update_one({'user_id': user_id}, {
                                   '$set': usuario}, upsert=True)

        return JsonResponse({'message': 'Mensaje agregado con éxito'})
    else:
        return JsonResponse({'message': 'Datos de mensaje incompletos'}, status=400)


def obtener_conversacion(user_id):
    collection_chat = mongodb_connector.get_collection('chat')
    # Buscar al usuario por su user_id
    usuario = collection_chat.find_one({'user_id': user_id})

    if usuario:
        # Obtener el historial de chat del usuario
        chat_history = usuario.get('chat_history', [])

        # Ordenar los mensajes por timestamp (de más antiguo a más reciente)
        chat_history.sort(key=lambda x: x.get('timestamp', datetime.min))

        # Crear una lista de tuplas con mensajes del usuario y del asistente
        conversacion = []
        user_message = None

        for mensaje in chat_history[:20]:
            role = mensaje.get('role')
            content = mensaje.get('message')

            if role == 'user':
                user_message = content
            elif role == 'assistant' and user_message is not None:
                conversacion.insert(0, (user_message, content))
                user_message = None

        return conversacion  # Retorna el array conversacion
    else:
        return []  # Retorna una lista vacía si el usuario no existe


################################################################# time_venture#############################
@csrf_exempt
def save_time_venture(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            venture_id = data.get('venture_id')
            day = data.get('day')
            time_ranges = data.get('time_ranges')

            if venture_id and day:
                collection_time_venture = mongodb_connector.get_collection(
                    'time_venture')
                # Check if a document for the company already exists
                existing_document = collection_time_venture.find_one(
                    {'venture_id': venture_id, 'day': day})

                if existing_document:
                    # Si time_ranges es null, elimina el documento
                    if time_ranges is None or (isinstance(time_ranges, list) and not time_ranges):
                        collection_time_venture.delete_one(
                            {'venture_id': venture_id, 'day': day})
                        return JsonResponse({'message': 'Time venture document deleted successfully'})
                    else:
                        # Si time_ranges no es null, actualiza los time_ranges
                        collection_time_venture.update_one(
                            {'venture_id': venture_id, 'day': day},
                            {'$set': {'time_ranges': time_ranges}}
                        )
                        return JsonResponse({'message': 'Time ranges updated successfully'})
                else:
                    # Si no existe, crea un nuevo documento time_venture
                    if time_ranges is not None and (not isinstance(time_ranges, list) or time_ranges):
                        time_venture = {
                            'venture_id': venture_id,
                            'day': day,
                            'time_ranges': time_ranges
                        }
                        result = collection_time_venture.insert_one(
                            time_venture)
                        return JsonResponse({'message': 'time_venture document saved successfully', 'document_id': str(result.inserted_id)})
                    else:
                        return JsonResponse({'message': 'time_ranges is required for new documents'}, status=400)
            else:
                return JsonResponse({'message': 'Incomplete data'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'message': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'message': 'Invalid request'}, status=400)


@csrf_exempt
def get_time_venture_by_company_name(request, venture_id):
    if request.method == 'GET':
        try:
            # Obtener el parámetro venture_id, que siempre está presente
            # Verificar si se proporcionó venture_id
            if not venture_id:
                return JsonResponse({'message': 'El parámetro venture_id es requerido'}, status=400)

            # Obtener el parámetro day, que es opcional
            day = request.GET.get('day')

            # Crear una conexión a la base de datos MongoDB
            collection_time_venture = mongodb_connector.get_collection(
                'time_venture')

            # Crear el filtro de búsqueda basado en los parámetros proporcionados
            query_filter = {'venture_id': venture_id}
            if day:
                query_filter['day'] = day

            # Buscar time_ventures con el filtro
            cursor = collection_time_venture.find(query_filter)

            # Convertir los documentos de MongoDB a una lista
            time_ventures = list(cursor)
            for time_venture in time_ventures:
                time_venture['_id'] = str(time_venture['_id'])

            return JsonResponse(time_ventures, safe=False)

        except Exception as e:
            return JsonResponse({'message': str(e)}, status=500)
    else:
        return JsonResponse({'message': 'Invalid request'}, status=400)


@csrf_exempt
def save_schedule_venture(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            schedule_id = data.get('schedule_id')
            start_time = data.get('start_time')
            end_time = data.get('end_time')
            status = data.get('status')
            venture_id = data.get('venture_id')
            client = data.get('client')
            services = data.get('services')
            visible = data.get('visible')
            title = data.get('title')
            description = data.get('description')
            color = data.get('color')
            location = data.get('location')
            alert = data.get('alert')

            if schedule_id and start_time and status and venture_id and client:
                # Crear una conexión a la base de datos MongoDB
                collection_schedule_venture = mongodb_connector.get_collection(
                    'schedule_venture')

                # Verificar si ya existe un documento para el schedule_id
                existing_document = collection_schedule_venture.find_one(
                    {'schedule_id': schedule_id})

                # Crear el documento schedule_venture
                schedule_venture = {
                    'title': title,
                    'visible': visible,
                    'description': description,
                    'color': color,
                    'location': location,
                    'alert': alert,
                    'schedule_id': schedule_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'status': status,
                    'venture_id': venture_id,
                    'services': services,
                    'client': client
                }

                overlap = check_time_overlap(
                    venture_id, start_time, end_time)

                if existing_document:
                    # Si existe, actualizar el documento
                    if overlap and (start_time != existing_document['start_time'] or end_time != existing_document['end_time']):
                        return JsonResponse({'message': 'La cita ya ha sido seleccionada por otro usuario'}, status=400)
                    collection_schedule_venture.update_one(
                        {'schedule_id': schedule_id},
                        {'$set': schedule_venture}
                    )
                    return JsonResponse({'message': 'Documento schedule_venture actualizado con éxito'})
                else:
                    if overlap:
                        return JsonResponse({'message': 'La cita ya ha sido seleccionada por otro usuario'}, status=400)
                    # Si no existe, insertar un nuevo documento
                    result = collection_schedule_venture.insert_one(
                        schedule_venture)
                    return JsonResponse({'message': 'Documento schedule_venture guardado con éxito', 'document_id': str(result.inserted_id)})
            else:
                return JsonResponse({'message': 'Datos incompletos'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'message': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'message': str(e)}, status=500)
    else:
        return JsonResponse({'message': 'Invalid request'}, status=400)


def get_allow_time(request):
    if request.method == 'GET':
        try:
            # Obtener los query parameters
            venture_id = request.GET.get('venture_id')
            day = request.GET.get('day')
            # Convertir a entero (default 0)
            time_service = int(request.GET.get('time_service', 0))

            output_format = '%Y-%m-%d %H:%M'

            # Convertir la fecha de string a un objeto datetime
            day_date = datetime.strptime(day, '%d/%m/%Y')

            try:
                # Intentar configurar el locale a español
                locale.setlocale(locale.LC_TIME, 'spanish')
            except locale.Error:
                # Si falla, usar un método alternativo
                days = ['LUNES', 'MARTES', 'MIÉRCOLES',
                        'JUEVES', 'VIERNES', 'SÁBADO', 'DOMINGO']
                day_name = days[day_date.weekday()]
            else:
                # Si el locale se configura con éxito
                day_name = day_date.strftime('%A').upper()

            if not venture_id or not day:
                return JsonResponse({'message': 'Venture ID y día son parámetros requeridos'}, status=400)

            collection_time_venture = mongodb_connector.get_collection(
                'time_venture')
            time_venture = collection_time_venture.find_one(
                {'venture_id': venture_id, 'day': day_name})

            if not time_venture:
                return JsonResponse({'message': 'No hay citas para hoy'}, status=404)

            # Obtener las horas de trabajo del venture
            working_hours = time_venture.get('time_ranges', [])

            # Calcular la fecha y hora inicial y final del día
            day_datetime = parser.parse(day)

            # Inicio y fin del día en formato UTC para la consulta
            day_start_utc = day_datetime.strftime(output_format)
            day_end_utc = (day_datetime.replace(
                hour=23, minute=59, second=0)).strftime(output_format)

            collection_schedule_venture = mongodb_connector.get_collection(
                'schedule_venture')
            # Consulta a MongoDB ajustada
            scheduled_appointments = list(collection_schedule_venture.find({
                'venture_id': venture_id,
                'start_time': {'$gte': day_start_utc, '$lt': day_end_utc}
            }))

            # Calcular las horas disponibles
            available_hours = []
            for working_hour in working_hours:
                start_time = datetime.strptime(
                    working_hour['start_time'], '%H:%M').time()
                end_time = datetime.strptime(
                    working_hour['end_time'], '%H:%M').time()

                # Ajustar current_time al mismo día que day_datetime
                current_time = day_datetime.replace(
                    hour=start_time.hour, minute=start_time.minute)
                end_datetime = day_datetime.replace(
                    hour=end_time.hour, minute=end_time.minute)

                # Generar intervalos de tiempo dentro del rango de trabajo
                while current_time + timedelta(minutes=time_service) <= end_datetime:
                    # Verificar si la hora está disponible
                    is_available = True
                    for appointment in scheduled_appointments:
                        appointment_start = make_naive(parser.parse(
                            appointment['start_time']))
                        appointment_end = make_naive(parser.parse(appointment.get(
                            'end_time', appointment['start_time'])))

                        # Verificar solapamiento
                        if current_time < appointment_end and current_time + timedelta(minutes=time_service) > appointment_start:
                            is_available = False
                            break

                    # Agregar la hora a la lista de horas disponibles si es válida
                    if is_available:
                        available_hour = {
                            "start_time": current_time.strftime('%Y-%m-%d %H:%M'),
                            "end_time": (current_time + timedelta(minutes=time_service)).strftime('%Y-%m-%d %H:%M')
                        }
                        available_hours.append(available_hour)

                    current_time += timedelta(minutes=time_service)

            return JsonResponse({'available_hours': available_hours})
        except Exception as e:
            return JsonResponse({'message': str(e)}, status=500)
    else:
        return JsonResponse({'message': 'Invalid request'}, status=400)


def make_naive(dt):
    """Convierte un datetime offset-aware en naive (sin zona horaria)."""
    if dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None:
        return dt.replace(tzinfo=None)
    return dt


@csrf_exempt
def get_schedule_venture(request):
    if request.method == 'GET':
        try:
            # Obtener los parámetros de consulta
            # Fecha en formato YYYY-MM-DD
            date_filter = request.GET.get('date', '')
            # Estado de la cita (opcional)
            status_filter = request.GET.get('status', '')
            venture_id_filter = request.GET.get(
                'venture_id', '')  # ID del venture (opcional)
            output_format = '%Y-%m-%d %H:%M'

            # Crear una conexión a la base de datos MongoDB
            collection_schedule_venture = mongodb_connector.get_collection(
                'schedule_venture')
            filter_dict = {}
            # Definir un filtro inicial basado en la fecha
            if date_filter:
                day_datetime = parser.parse(date_filter)
                # Inicio y fin del día en formato UTC para la consulta
                # Inicio y fin del día en formato UTC para la consulta
                day_start_utc = day_datetime.strftime(output_format)
                day_end_utc = (day_datetime.replace(
                    hour=23, minute=59, second=0)).strftime(output_format)

                filter_dict = {'start_time': {
                    '$gte': day_start_utc, '$lt': day_end_utc}}
            # Agregar filtros adicionales si se proporcionan
            if status_filter:
                filter_dict['status'] = status_filter
            if venture_id_filter:
                filter_dict['venture_id'] = venture_id_filter

            # Buscar las citas que coinciden con los filtros
            results = list(collection_schedule_venture.find(filter_dict))

            for result in results:
                result['_id'] = str(result['_id'])

            return JsonResponse({'events': results})
        except Exception as e:
            return JsonResponse({'message': str(e)}, status=500)
    else:
        return JsonResponse({'message': 'Invalid request'}, status=400)


@csrf_exempt
def subir_documento(request):
    if request.method == 'POST':
        print("ingreso")
        form = DocumentoForm(request.POST, request.FILES)

        if form.is_valid():
            documento = form.save()
            # Generar la URL del archivo almacenado en S3
            url_archivo = documento.archivo.url
            return JsonResponse({
                'mensaje': 'Subida exitosa',
                'url_archivo': url_archivo
            }, status=201)
    else:
        form = DocumentoForm()
    return JsonResponse({'mensaje': 'Método no permitideishon'}, status=405)


def check_time_overlap(venture_id, start_time_str, end_time_str):
    try:
        # Validar que los parámetros necesarios estén presentes
        if not venture_id or not start_time_str or not end_time_str:
            raise ValueError(
                "Venture ID, start time y end time son parámetros requeridos")

        # Convertir los strings de tiempo a objetos datetime
        start_time = parser.parse(start_time_str)
        end_time = parser.parse(end_time_str)

        collection_schedule_venture = mongodb_connector.get_collection(
            'schedule_venture')
        # Consulta a MongoDB para obtener citas programadas
        scheduled_appointments = list(collection_schedule_venture.find({
            'venture_id': venture_id,
            'start_time': {'$lt': end_time_str},
            'end_time': {'$gt': start_time_str}
        }))

        # Verificar si hay algún solapamiento
        for appointment in scheduled_appointments:
            appointment_start = parser.parse(appointment['start_time'])
            appointment_end = parser.parse(appointment.get(
                'end_time', appointment['start_time']))

            if start_time < appointment_end and end_time > appointment_start:
                # Se encontró un solapamiento
                return True

        # No se encontró solapamiento
        return False
    except Exception as e:
        raise Exception(f"Error al verificar el solapamiento: {str(e)}")


@firebase_auth_required
def health_check(request):
    # Aquí puedes añadir lógica para verificar cosas como la conexión a la base de datos, etc.
    # Por ahora, simplemente devolvemos un estado "OK".
    return JsonResponse({"status": "OK"})
