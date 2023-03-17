# Generated by Django 4.1 on 2023-01-26 12:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("face_rec_app", "0002_image"),
    ]

    operations = [
        migrations.CreateModel(
            name="UploadFile",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("video", models.FileField(blank=True, null=True, upload_to="video/")),
            ],
        ),
        migrations.AlterField(
            model_name="image",
            name="picture",
            field=models.ImageField(blank=True, null=True, upload_to="images_out/"),
        ),
    ]
