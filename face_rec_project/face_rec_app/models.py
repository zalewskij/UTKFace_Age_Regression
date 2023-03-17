from django.db import models


class UploadFiles(models.Model):
    images = models.FileField(upload_to='images/')


class UploadFile(models.Model):
    video = models.FileField(upload_to='video/')


class Image(models.Model):
    picture = models.ImageField(upload_to='images_out/')
