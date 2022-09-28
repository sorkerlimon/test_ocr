from django.db import models

# Create your models here.


import uuid

# Create your models here.

class Esr(models.Model):
    create = models.DateTimeField(auto_now_add=True)
    esr = models.FloatField(blank=True, null=True)
    id = models.UUIDField(default=uuid.uuid4,unique=True,primary_key=True,editable=False)

    def __str__(self):
        return f'ESR : {str(self.esr)} --- {self.id}'


class Imageadd(models.Model):
    
    image = models.ImageField(default='default.jpg',upload_to='blood/')
    date = models.DateTimeField(auto_now_add=True)
    id = models.UUIDField(default=uuid.uuid4,unique=True,primary_key=True,editable=False)

    
    def __str__(self):
        return str(self.image)