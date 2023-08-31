#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Recent versions of two useful libraries are installed

get_ipython().system('pip install pandas==1.3.3')
get_ipython().system('pip install numpy==1.20.3')


# In[2]:


# Last month’s historical data are downloaded for some of the largest companies listed on the FTSE MIB

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Symbol list of companies listed on FTSE MIB
companies = ["AZM.MI", "ENI.MI", "ENEL.MI", "ISP.MI", "LDO.MI"]

# Today and date a month ago
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')

# Initialize an empty DataFrame for data
historical_data = pd.DataFrame()

# Extract data for each company
for company_symbol in companies:
    company_data = yf.download(company_symbol, start=start_date, end=end_date)
    historical_data[company_symbol] = company_data['Close']

# Print historical data of companies
print(historical_data)


# In[2]:


# Other useful libraries are installed

get_ipython().system('pip install yfinance numpy statsmodels matplotlib')


# In[13]:


# Based on the extracted data of the last month, a forecast is made for the day after the run using an ARIMA model.

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Symbol of the company you want to forecast
company_symbol = "AZM.MI"

# Today and forecast date (tomorrow)
end_date = datetime.today().strftime('%Y-%m-%d')
forecast_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')

try:
    # Extract historical closing price data
    company_data = yf.download(company_symbol, start=start_date, end=end_date)
    closing_prices = company_data['Close']

    # Remove Nan or infinite values
    closing_prices = closing_prices.dropna()

    if len(closing_prices) < 2:
        raise ValueError("There is not enough data to make a prediction.")

    # Set the frequency of the date index to’D' (daily)
    closing_prices = closing_prices.asfreq('D')

    # Interpolation to fill missing values
    closing_prices = closing_prices.interpolate()

    # Creation of the ARIMA model
    model = ARIMA(closing_prices, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    
    # Make the forecast for tomorrow
    forecast, stderr, conf_int = model_fit.forecast(steps=1)

    # Print the forecast for tomorrow
    print("Forecast date:", forecast_date)
    print("Forecast for tomorrow:", forecast[0])

    # Plot of historical data and forecast
    plt.figure(figsize=(10, 6))
    plt.plot(closing_prices.index, closing_prices.values, label='Historical Data')
    plt.plot([closing_prices.index[-1], pd.to_datetime(forecast_date)], [closing_prices.iloc[-1], forecast[0]], marker='o', color='r', label='Forecast for tomorrow')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.title(f'Forecast share price for {company_symbol}')
    plt.xlabel('Date')
    plt.ylabel('Closing price')
    plt.show()

except Exception as e:
    print("An error has occurred:", e)
    
# The result of the script is given from the date of forecast 
# and from the relative forecasted price 
# more a diagram that shows the course of the historical data and the forecasted value.


# In[1]:


# This script is to run two days after the previous ones 
# to compare the expected price and the actual price if the actual price is available for that day.

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Symbol of the company you want to forecast
company_symbol = "AZM.MI"

# Today
end_date = datetime.today().strftime('%Y-%m-%d')

try:
    # Extract historical closing price data for today’s date
    company_data = yf.download(company_symbol, start=end_date, end=end_date)
    closing_prices = company_data['Close']

    if closing_prices.empty:
        raise ValueError("No data available for today’s date.")

    # Creation of the ARIMA model
    model = ARIMA(closing_prices, order=(5,1,0))
    model_fit = model.fit(disp=0)
    
    # Make the forecast for today
    forecast, stderr, conf_int = model_fit.forecast(steps=1)

    # Calculate the delta between predicted and actual price
    delta = forecast[0] - closing_prices.iloc[-1]
    
    # Print the forecast for today and the delta
    print("Forecast for today:", forecast[0])
    print("Actual price today:", closing_prices.iloc[-1])
    print("Delta:", delta)

    # Plot of historical data and forecast
    plt.figure(figsize=(10, 6))
    plt.plot(closing_prices.index, closing_prices.values, label='Historical data')
    plt.plot([closing_prices.index[-1], closing_prices.index[-1]], [closing_prices.iloc[-1], forecast[0]], marker='o', color='r', label='Forecast for today')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.title(f'Forecast share price for {company_symbol}')
    plt.xlabel('Date')
    plt.ylabel('Closing price')
    plt.show()

except Exception as e:
    print("An error has occurred:", e)


# In[ ]:




