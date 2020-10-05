import numpy as np #linear algebra
import pandas as pd #data processing,CSV i/o
import seaborn as sns #data visualization
import matplotlib.pyplot as plt 
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression #For prediction model
from sklearn.model_selection import train_test_split #For splitting dataset into test and train
from sklearn.metrics import mean_squared_error 

data = pd.read_csv("bitcoin_dataset.csv")
data_head = data.head() #showing top 5 rows
value = data['btc_market_price'][1023] #accessing column values


#print(data_head)
#print(value)
#using joint plot we can see correlation between two features
print(len(data.columns))
x=data.columns[1]

#for i in range(2,len(data.columns)-1):
#	y=data.columns[i]
#	sns.jointplot(x,y,data = data,stat_func = pearsonr) #Find correlation of btc_market_price with very other feature
	
#plt.show() 

#Fiiling NaN values with mean of above and below available value

#useful features with high correlation cofficient
uf = data[['btc_market_price','btc_market_cap','btc_n_transactions','btc_miners_revenue','btc_cost_per_transaction','btc_difficulty','btc_hash_rate','btc_cost_per_transaction_percent']]
uf.fillna((uf.ffill()+uf.bfill())/2,inplace=True)
#print(uf.head())

#splitting dataset into dependent and independent variables
X = uf.iloc[:,1:]
#print(X)
Y = uf['btc_market_price']   #Actual true value
#print(Y)
#splitting data into train and test
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size =0.3)
#print(x_train)

#Prediction model
clf=LinearRegression()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print(clf.intercept_)
print(clf.coef_)


accuracy = clf.score(x_test,y_test)
print(accuracy)

mse1 = mean_squared_error(y_test,y_pred) #True value = y_test,predicted value = y_pred
print(mse1)
mse2 = np.square(np.subtract(y_test,y_pred)).mean()
print(mse2)