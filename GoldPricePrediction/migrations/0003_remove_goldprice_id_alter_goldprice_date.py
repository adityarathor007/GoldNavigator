# Generated by Django 4.0.3 on 2024-11-17 05:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('GoldPricePrediction', '0002_goldprice_delete_user'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='goldprice',
            name='id',
        ),
        migrations.AlterField(
            model_name='goldprice',
            name='date',
            field=models.DateField(primary_key=True, serialize=False),
        ),
    ]
