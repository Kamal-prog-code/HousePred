# Generated by Django 3.1.3 on 2020-12-11 19:42

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('APP', '0003_auto_20201212_0052'),
    ]

    operations = [
        migrations.AddField(
            model_name='profile',
            name='Testdate',
            field=models.DateTimeField(blank=True, default=datetime.datetime.now),
        ),
    ]
