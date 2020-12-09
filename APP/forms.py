from django import forms
from .models import *

class usersForm(forms.ModelForm):
    class Meta:
        model = users
        fields = ['User_Name','Email','Password']