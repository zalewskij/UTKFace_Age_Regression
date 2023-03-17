# Generated by Django 4.1 on 2023-01-26 14:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("face_rec_app", "0005_alter_uploadfile_video"),
    ]

    operations = [
        migrations.AlterField(
            model_name="image",
            name="picture",
            field=models.ImageField(upload_to="images_out/"),
        ),
        migrations.AlterField(
            model_name="uploadfile",
            name="video",
            field=models.FileField(upload_to="video/"),
        ),
        migrations.AlterField(
            model_name="uploadfiles",
            name="images",
            field=models.FileField(upload_to="images/"),
        ),
    ]
