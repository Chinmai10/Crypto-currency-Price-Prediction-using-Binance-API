
# coding: utf-8

# ## Data Preparation
# 
# The first step is to get the data. 
# 
# In this example, let us get the data from binance.com. We can use the binance python api.
# 
# https://python-binance.readthedocs.io/en/latest/

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from binance.client import Client
client = Client()


# In[ ]:


info = client.get_symbol_info('BTCUSDT')
info


# In[3]:


klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jan, 2020")
len(klines)


# In[4]:


bitcoin_df = pd.DataFrame(klines)
bitcoin_df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote", "no_trades", "base_buy", "quote_buy", "ignore"]
bitcoin_df


# In[5]:


import datetime
bitcoin_df["date"] = bitcoin_df["open_time"].apply(lambda x: datetime.datetime.fromtimestamp(x/1000))
bitcoin_df


# In[6]:


bitcoin_df.to_csv("bitcoin_daily_prices.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:




