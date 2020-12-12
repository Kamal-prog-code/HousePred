
from django.shortcuts import render
from django.shortcuts import render, redirect 
from django.http import HttpResponse
import pandas as pd
from .models import *
import joblib
from django.shortcuts import render,redirect
from django.template import Context, loader
from django.http import HttpResponse
from django.contrib.auth import login, authenticate
from django.contrib.auth.models import User, auth
from django.contrib.auth.views import *
import pickle
import joblib
from .forms import usersForm
import os

def signup(request):
    if request.method == 'POST':
        stu = usersForm(request.POST)
        if stu.is_valid():
            user = User.objects.create_user(username=stu.cleaned_data['User_Name'],
                                            password=stu.cleaned_data['Password'],
                                            email=stu.cleaned_data['Email'])

            user.save()
            stu.save()
            return redirect('login')
    else:
        stu = usersForm()
    return render(request,'signup.html',{'form':stu})

def login(request):
    if request.method == 'POST':
        user = User()
        username = request.POST['user']
        password = request.POST['pass']
        user = authenticate(username=username, password=password)
        context = {'user':request.user}
        if user is not None:
            auth.login(request, user)
            return redirect('home')
        else:
            return render(request,'signin.html',context)
    else:
        return render(request,'signin.html')


def home(request):
	return render(request,'home.html')

def pred(request):
	if(request.method=='POST'):
		pr = Profile()

		data=list((request.POST).dict().values())[1:]
		num_data=[ float(i) for i in data ]

		df_data=pd.DataFrame([num_data],columns=['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','sqft_living15','sqft_lot15'])
		print(df_data)
		reg=joblib.load("prediction/Regmodel.pkl")
		y_pred = reg.predict(df_data)
		
		print(y_pred)
		pr.usern = str(request.user)
		pr.bedroom=request.POST.get('bed')
		pr.zipcode=request.POST.get('zip')
		pr.HouseRate=y_pred[0]
		pr.save()
		return render(request,'pred.html',{'result':y_pred})
	else:
		return render(request, 'pred.html')

def profile(request):
	res = Profile.objects.all()
	return render(request,'profile.html',{'res':res})





