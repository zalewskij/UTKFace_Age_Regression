# Generated by Django 4.1 on 2023-01-26 13:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("face_rec_app", "0003_uploadfile_alter_image_picture"),
    ]

    operations = [
        migrations.AlterField(
            model_name="uploadfile",
            name="video",
            field=models.FileField(blank=True, null=True, upload_to="video_out/"),
        ),
    ]
