#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()



# In[2]:


df = pd.read_csv("bitcoin_daily_prices.csv", parse_dates=['date'])
df.head()


# In[3]:


fig, ax = plt.subplots(figsize=(12,8))
sns.lineplot(y = df["close"].values, x=df["date"].values, alpha=0.8, color=color[3], ax=ax)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price in USD', fontsize=12)
plt.title("Closing price distribution of bitcoin", fontsize=15)
plt.show()



import plotly.graph_objects as go

fig = go.Figure(
    data=[go.Candlestick(
        x=df["date"], 
        open=df['open'], 
        high=df['high'], 
        low=df['low'], 
        close=df['close'])
         ]
)
fig.show()

