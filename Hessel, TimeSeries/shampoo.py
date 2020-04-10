# load and plot dataset
from pandas import read_excel
from datetime import datetime
from matplotlib import pyplot
import os
# load dataset
def parser(x):
	return x

thisdir = os.getcwd()

series = read_excel(r'Documents\Python Scripts\data\MAAND OPEN JAREN.xlsx', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# summarize first few rows
print(series.head())
# line plot
series.plot()
pyplot.show()
#%%
from pandas import read_excel
from datetime import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
def parser(x):
	return x
series = read_excel(r'Documents\Python Scripts\data\MAAND OPEN JAREN.xlsx', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
autocorrelation_plot(series)
pyplot.show()
#%%
from pandas import read_excel
from datetime import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot

def parser(x):
	return x
series = read_excel(r'Documents\Python Scripts\data\MAAND OPEN JAREN.xlsx', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
## fit model
model = ARIMA(series, order=(1,0,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())
#%%
from pandas import read_excel
from datetime import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

def parser(x):
	return x
series = read_excel(r'Documents\Python Scripts\data\MAAND OPEN JAREN.xlsx', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
#
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# Doorloop de testgevallen
for t in range(len(test)):
#   Maak een model op basis van de historische
#   observaties in de trainingset        
    model = ARIMA(history, order=(1,0,1))
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
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()