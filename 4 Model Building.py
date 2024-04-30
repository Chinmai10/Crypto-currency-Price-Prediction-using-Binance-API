#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics, model_selection
from xgboost import XGBRegressor, XGBClassifier
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("bitcoin_daily_prices.csv", parse_dates=["date"])
df.head()


# In[3]:


# Feature Engineering
df = pd.DataFrame(df["close"].values)

# Let us take the last 10 days values as inputs
n_in = 20
cols = list()
for i in range(n_in, 0, -1):
    cols.append(df.shift(i))

# Let us take the next days value as the target
n_out = 1
cols.append(df.shift(-n_out))

# Concat all to create the data
full_df = pd.concat(cols, axis=1)
full_df


# In[4]:


# Drop the Nan values
full_df = full_df.dropna().reset_index(drop=True)

# Create column names
full_df.columns = [f't{i}' for i in range(n_in, 0, -1)] + ['target']
full_df


# In[5]:


# Input and Target features
input_col_names = [f't{i}' for i in range(n_in, 0, -1)]
target_col_name = 'target'


# In[6]:


# Normalize the features
full_df['target'] = full_df['target'] - full_df['t1']
for col in input_col_names:
    full_df[col] = full_df[col] - full_df['t1']
full_df


# In[7]:


# Split the dataset into train and validation dataset
tfold = model_selection.TimeSeriesSplit(n_splits=2, test_size=100)
for dev_index, val_index in tfold.split(full_df):
    train_df = full_df.iloc[dev_index, :]
    val_df = full_df.iloc[val_index, :]
train_df


# In[8]:


val_df


# In[9]:


train_X = train_df[input_col_names]
train_y = train_df[target_col_name]

val_X = val_df[input_col_names]
val_y = val_df[target_col_name]


# ## Model Building

# In[10]:


model = XGBRegressor(objective='reg:squarederror', n_estimators=200)
model.fit(train_X, train_y)


# In[11]:


preds = model.predict(val_X)
rmse = np.sqrt(metrics.mean_squared_error(val_y, preds))
print("RMSE: %f" % (rmse))


# In[12]:


pred_df = pd.DataFrame({"Actuals":val_y, "Predictions":preds})
pred_df


# ## Classification Model

# In[13]:


full_df["binary_target"] = (full_df["target"] > full_df["t1"]).astype(int)
full_df


# In[14]:


train_df = full_df.iloc[dev_index, :]
val_df = full_df.iloc[val_index, :]

target_col_name = "binary_target"
train_X = train_df[input_col_names]
train_y = train_df[target_col_name]

val_X = val_df[input_col_names]
val_y = val_df[target_col_name]


# In[15]:


model = XGBClassifier(objective='binary:logistic', n_estimators=50)
model.fit(train_X, train_y)


# In[16]:


preds = model.predict_proba(val_X)[:,1]
auc = metrics.roc_auc_score(val_y, preds)
print(auc)


# In[19]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(val_y, preds>0.4, labels=[0, 1])
disp = ConfusionMatrixDisplay(cm, display_labels=["Down", "Up"])
disp.plot()
plt.show()


# In[ ]:




