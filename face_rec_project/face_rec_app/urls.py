from django.conf.urls.static import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.urls import path
from django.conf import settings

from . import views

urlpatterns = [
    path('', views.home, name="stream"),
    path('webcam_stream', views.webcam_stream, name='webcam_stream'),
    path('webcam_stream/stream', views.stream, name='stream_output'),
    path('webcam_stream/started', views.start_stream, name='start_streaming'),
    path('webcam_stream/stopped', views.stop_stream, name='stop_streaming'),
    path('video_processing', views.home_vp, name='video_processing_home'),
    path('video_processing/processed', views.process_video, name='video_processing_process'),
    path('video_processing/show_video', views.video_stream, name='video_processing_output'),
    path('video_processing/start', views.start_displaying, name='video_processing_start_output'),
    path('image_collage', views.home_ic, name='image_collage_home'),
    path('image_collage/processed', views.process_images, name='image_collage_process'),
    path('image_collage/show_images', views.image_collage, name='image_collage_start_output'),
] + staticfiles_urlpatterns() + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
