from django import forms
from django.forms import ClearableFileInput
from .models import UploadFiles, UploadFile


# Models used for uploading images and the video
class ImagesUpload(forms.ModelForm):
    class Meta:
        model = UploadFiles
        fields = ['images']
        widgets = {
            'images': ClearableFileInput(attrs={'multiple': True, 'value': "Choose images"})
        }


class VideoUpload(forms.ModelForm):
    class Meta:
        model = UploadFile
        fields = ['video']
        widgets = {
            'video': ClearableFileInput(attrs={'value': "Choose video"})
        }

