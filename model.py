import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
data = pd.read_csv('complete_data_set/TestTS.csv', parse_dates=['Date'], index_col='Date',date_parser=dateparse)
print(data.head())
#data.plot()
#plt.show()

# data clean
sumOfNull = data.isnull().sum()
print('Null Values :-',sumOfNull)
modifiedData = data.dropna()
sumOfNull = modifiedData.isnull().sum()
print('Null Values in modified data:-',sumOfNull)
#data.plot()
#plt.show()

# fit model
model = ARIMA(modifiedData, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())

#Rolling Forecast ARIMA Model
X = modifiedData.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.plot(predictions, color='red')
plt.plot(test)
plt.show()