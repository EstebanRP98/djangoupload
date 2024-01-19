from django.urls import path, include
from django.conf import settings
from . import views
from django.conf.urls.static import static

urlpatterns = [
    path('ventures/<venture_id>/',
         views.save_or_update_venture, name='update_venture'),
    path('get_venture/', views.get_venture_by_name, name='get_venture_by_name'),
    path('upload_pdf/', views.upload_pdf, name='upload_pdf'),
    path('get_chat/', views.get_chat_by_user_id, name='get_chat_by_user_id'),
    path('save-time-venture/', views.save_time_venture, name='save_time_venture'),
    path('get-time-venture/<str:venture_id>/',
         views.get_time_venture_by_company_name, name='get_time_venture_by_company_name'),
    path('save-schedule-venture/', views.save_schedule_venture,
         name='save_schedule_venture'),
    path('get-allow-time/', views.get_allow_time, name='get_allow_time'),
    path('schedule-venture/', views.get_schedule_venture,
         name='get_schedule_venture'),
    path('upload/', views.subir_documento, name='subir_documento'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
