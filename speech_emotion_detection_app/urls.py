from django.urls import path
from django.conf.urls import url
from .import views

from django.conf import settings
from django.views.static import serve

urlpatterns = [
    path('', views.Upload_audio.as_view(), name = 'index'),
    path('view/', views.ViewAudio.as_view(), name = 'emotion'),
]

if settings.DEBUG:
    urlpatterns += [
        url(r'media/(?P<path>.*)$', serve, {
            'document_root' : settings.MEDIA_ROOT
        }),
    ]