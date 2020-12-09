
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
		# YearBuilt=request.POST.get('yb',None)
		# Neighborhood=request.POST.get('ng',None)
		# OverallQual=request.POST.get('ovq',None)
		# YearRemodAdd=request.POST.get('yra',None)
		# ExterQual=request.POST.get('eq',None)
		# Foundation=request.POST.get('foun',None)
		# BsmtQual=request.POST.get('bsq',None)
		# TotalBsmtSF=request.POST.get('tb',None)
		# GrLivArea=request.POST.get('gla',None)
		# fstFlrSF=request.POST.get('1ff',None)
		# FullBath=request.POST.get('fb',None)
		# KitchenQual=request.POST.get('kq',None)
		# TotRmsAbvGrd=request.POST.get('trag',None)
		# GarageFinish=request.POST.get('gf',None)
		# GarageCars=request.POST.get('gc',None)
		# GarageArea=request.POST.get('ga',None)

		data=list((request.POST).dict().values())[1:]
		num_data=[i for i in data ]
		df_data=pd.DataFrame([num_data],columns=['YearBuilt','Neighborhood','OverallQual','YearRemodAdd','ExterQual','Foundation','BsmtQual','TotalBsmtSF','GrLivArea','1stFlrSF','FullBath','KitchenQual','TotRmsAbvGrd','GarageFinish','GarageCars','GarageArea'])
		print(df_data)
		reg=joblib.load("prediction/modl.pkl")
		y_pred = reg.predict(df_data)[0]
		
		print(y_pred)
		pr.usern = str(request.user)
		pr.HouseRate=y_pred
		pr.save()
		return render(request,'pred.html',{'result':y_pred})
	else:
		return render(request, 'pred.html')

def profile(request):
	return render(request,'profile.html',{})





