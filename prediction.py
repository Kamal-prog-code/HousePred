
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
import numpy as np
from numpy import arange
import os
import pickle
data = pd.read_csv("kc_house_data.csv")
import warnings
warnings.filterwarnings('ignore')


# import seaborn as sns
# plt.figure(figsize=(16,9))
# sns.heatmap(dataset.isnull())
miss_col = data.columns[data.isnull().any()]
print(miss_col)
# data['bedrooms'].value_counts().plot(kind='bar')
# plt.title('number of Bedroom')
# plt.xlabel('Bedrooms')
# plt.ylabel('Count')
# sns.despine()
# plt.figure(figsize=(10,10))
# sns.jointplot(x=data.lat.values, y=data.long.values, size=10)
# plt.ylabel('Longitude', fontsize=12)
# plt.xlabel('Latitude', fontsize=12)
# plt.show()

# sns.despine()
# plt.scatter(data.price,data.sqft_living)
# plt.title("Price vs Square Feet")
# plt.scatter(data.price,data.long)
# plt.title("Price vs Location of the area")
# plt.scatter(data.price,data.lat)
# plt.xlabel("Price")
# plt.ylabel('Latitude')
# plt.title("Latitude vs Price")

# plt.scatter(data.bedrooms,data.price)
# plt.title("Bedroom and Price ")
# plt.xlabel("Bedrooms")
# plt.ylabel("Price")
# plt.show()
# sns.despine()
# plt.scatter((data['sqft_living']+data['sqft_basement']),data['price'])
# plt.scatter(data.waterfront,data.price)
# plt.title("Waterfront vs Price ( 0= no waterfront)")
# train1 = data.drop(['id', 'price'],axis=1)
# data.floors.value_counts().plot(kind='bar')
# plt.scatter(data.condition,data.price)
# plt.scatter(data.zipcode,data.price)
# plt.title("Which is the pricey location by zipcode?")
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

reg = LinearRegression()

labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date ]
data['date'] = conv_dates
train1 = data.drop(['id', 'price'],axis=1)

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaler.fit(train1)
x_std=scaler.transform(train1)

from sklearn.decomposition import PCA
pca=PCA().fit(x_std)

corr_mat = data.corr()

cols_to_drop = []
CORR_THRESH = 0.05
for col in train1:
    corr = data[col].corr(data['price'])
    if (abs(corr) < CORR_THRESH):
        cols_to_drop.append(col)

df=train1
for col in df:
        if col in cols_to_drop:
            df.drop(labels=[col], axis=1, inplace=True)
df=df.drop('date',axis=1)                    
print(df.keys())
print(df)
x_train1 , x_test1 , y_train1 , y_test1 = train_test_split(df, labels , test_size = 0.20)
reg.fit(x_train1,y_train1)
predrr = reg.predict(x_test1)
print("Linear Regression Score : ",r2_score(y_test1, predrr)*100)

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
parameters = {'alpha':list(arange(1e-1,2.0,0.1))} 

ridge = Ridge()
ridge_reg =GridSearchCV(ridge,parameters,scoring = 'neg_mean_squared_error',cv=5)
ridge_reg.fit(x_train1,y_train1)

ridgeReg = Ridge(alpha=ridge_reg.best_params_['alpha'], normalize=True)
ridgeReg.fit(x_train1,y_train1)
predrr = ridgeReg.predict(x_test1)

print("Ridge Regression Score : ",r2_score(y_test1, predrr)*100)

from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

lasso = Lasso()
lasso_reg =GridSearchCV(lasso,parameters,scoring = 'neg_mean_squared_error',cv=5)
lasso_reg.fit(x_train1,y_train1)
lassoReg = Lasso(alpha=lasso_reg.best_params_['alpha'], normalize=True)
lassoReg.fit(x_train1,y_train1)
predrr = lassoReg.predict(x_test1)

print("Lasso Regression Score : ",r2_score(y_test1, predrr)*100)

from sklearn import ensemble
ensreg = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'ls')
ensreg.fit(x_train1,y_train1)
predrr = ensreg.predict(x_test1)

print("Gradient Boosting Score : ",r2_score(y_test1, predrr)*100)

# save_path = 'prediction/'
# completeName = os.path.join(save_path, "Regmodel.pkl")         
# pickle.dump(ensreg, open(completeName, 'wb'))
