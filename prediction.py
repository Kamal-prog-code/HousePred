import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

pd.pandas.set_option('display.max_columns',None)
dataset=pd.read_csv('train.csv')
dataset=dataset.set_index("Id")
numerical_features = dataset.select_dtypes(include=['int64','float64']).columns
Per_null = (dataset.isnull().sum()/dataset.shape[0]) * 100

drop_col = Per_null[Per_null>20].keys()
dataset = dataset.drop(drop_col,'columns')
miss_col = dataset.columns[dataset.isnull().any()]

for i in miss_col:
    dataset[i] = dataset[i].replace(np.nan,dataset[i].mode()[0])
categorical_features=dataset.select_dtypes(include=['object']).columns

for feature in categorical_features:
    labels_ordered=dataset.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    dataset[feature]=dataset[feature].map(labels_ordered)

feature_scale=[feature for feature in dataset.columns if feature not in ['SalePrice']]

scaler=MinMaxScaler()
scaler.fit(dataset[feature_scale])

data = pd.concat([dataset[['SalePrice']].reset_index(drop=False),pd.DataFrame(scaler.transform(dataset[feature_scale]), columns=feature_scale)],axis=1)
data=data.set_index("Id")

corr_scores = data.corr()
corr_scores_features = corr_scores.index[abs(corr_scores["SalePrice"]) >= 0.5]

X=data[corr_scores_features].values
Y=data['SalePrice'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

Lin_classifier = LinearRegression()
mse=cross_val_score(Lin_classifier,X_train,y_train,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print("Linear error rate",mean_mse*100)


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge_classifier = Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge_classifier,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train,y_train)
print("Ridge error rate: ",ridge_regressor.best_score_*100)