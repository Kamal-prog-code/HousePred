from django.db import models
from django.contrib.auth.models import User
from datetime import datetime    

class users(models.Model):  
    user = models.OneToOneField(User,on_delete=models.CASCADE,null=True)
    User_Name = models.CharField(max_length=20) 
    Email = models.CharField(max_length=40) 
    Password  = models.CharField(max_length=20)  
      
    def __str__(self):
        return self.User_Name 

class Prediction(models.Model):
    user = models.CharField(max_length=50,null=True)
    HouseRate = models.CharField(max_length=50,null=True)

    def __str__(self):
        return str(self.user)