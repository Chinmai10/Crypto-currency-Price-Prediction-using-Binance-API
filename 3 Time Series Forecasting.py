#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from prophet import Prophet

INPUT_FILE = "bitcoin_daily_prices.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=['date'], usecols=["date", "close"])
df.columns = ["y", "ds"]
df


# In[2]:


# Initialize the prophet class
m = Prophet()


# In[3]:


m.fit(df)


# In[4]:


future = m.make_future_dataframe(periods=30)
future


# In[5]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[6]:


m.plot(forecast)


# In[7]:


plot = m.plot_components(forecast)


# ## References:
# 
# * https://facebook.github.io/prophet/docs/quick_start.html
# * https://otexts.com/fpp2/
