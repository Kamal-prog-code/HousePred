# Generated by Django 3.1.3 on 2020-12-11 19:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('APP', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='profile',
            name='latitude',
            field=models.CharField(max_length=50, null=True),
        ),
        migrations.AddField(
            model_name='profile',
            name='zipcode',
            field=models.CharField(max_length=50, null=True),
        ),
    ]