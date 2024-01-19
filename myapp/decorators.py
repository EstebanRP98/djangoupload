from functools import wraps
from firebase_admin import auth
from django.http import JsonResponse


def firebase_auth_required(view_func):
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        # Intenta obtener el token del encabezado HTTP
        token = request.META.get('HTTP_AUTHORIZATION')

        # Si no está en el encabezado, intenta obtenerlo de los parámetros de consulta
        if not token:
            token = request.GET.get('Authorization')

        # Si aún no hay token, devuelve un error
        if not token:
            return JsonResponse({'message': 'No token provided'}, status=401)

        try:
            # Quita el prefijo "Bearer " si está presente
            token = token.replace("Bearer ", "")
            # Verifica el token con Firebase
            decoded_token = auth.verify_id_token(token)
            request.user = decoded_token
            return view_func(request, *args, **kwargs)
        except auth.InvalidIdTokenError:
            return JsonResponse({'message': 'Invalid token'}, status=403)

    return _wrapped_view
