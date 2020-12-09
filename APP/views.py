
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
from sklearn.linear_model import LogisticRegression
def login(request):
    if request.method == 'POST':
        user = User()
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        context = {'user':request.user}
        if user is not None:
            auth.login(request, user)
            return redirect('home')
        else:
            return render(request,'login.html',context)
    else:
        return render(request,'login.html')


def home(request):
	return render(request,'home.html')

def pred(request):
	if(request.method=='POST'):
		test=DiabTest()
		test.Pregnancies=float(request.POST.get('pg',None))
		test.Glucose=float(request.POST.get('gl',None))
		test.BP=request.POST.get('bp',None)
		test.Skinthickness=float(request.POST.get('st',None))
		test.Insulin=float(request.POST.get('ins',None))
		test.BMI=float(request.POST.get('bmi',None))
		test.Diabetic_pf=float(request.POST.get('dpf',None))
		test.age=float(request.POST.get('age',None))
		data=list((request.POST).dict().values())[1:]
		num_data=[i for i in data ]
		df_data=pd.DataFrame([num_data],columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
		cls=joblib.load("prediction/dblogR.pkl")
		y_pred = cls.predict(df_data)[0]
		
		if(y_pred==0):
			str_res='Non Diabetic'
		else:
			str_res='Diabetic'
		print(str_res)
		test.user = str(request.user)
		test.Res=str_res
		test.save()
		return render(request,'diab_test.html',{'result':str_res})
	else:
		return render(request, 'diab_test.html')

def profile(request):
	heartT = HeartTest.objects.filter(user=str(request.user))
	diabT = DiabTest.objects.filter(user=str(request.user))
	return render(request,'profile.html',{'heartT':heartT,'diabT':diabT})





