#!/usr/bin/env python
# coding: utf-8

# ## Feature Engineering
# Creating New Features and Saving to csv File
# 

# ### 2.1) Creating more lag features
# lag_1 -> last week's sales,
# lag_2 -> sales 2 weeks ago and so on

# In[3]:


import pandas as pd 

df = pd.read_csv('../data/Walmart.csv')


for lag in [1, 2, 3, 4, 5, 6, 7]:
    df[f'lag_{lag}'] = df.groupby('Store')['Weekly_Sales'].shift(lag)  



# In[8]:


df.columns = df.columns.str.lower()
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')


# ### 2.2) Rolling window statistics
# Rolling windows capture short-term trends

# In[9]:


# Rolling means and std
df['rolling_mean_4'] = df.groupby('store')['weekly_sales'].shift(1).rolling(window=4).mean()
df['rolling_std_4']  = df.groupby('store')['weekly_sales'].shift(1).rolling(window=4).std()

df['rolling_mean_12'] = df.groupby('store')['weekly_sales'].shift(1).rolling(window=12).mean()


# ### 2.3) Date-based features

# In[10]:


df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
df['dayofweek'] = df['date'].dt.dayofweek


# ### 2.4) Holiday Lag

# In[11]:


df['holiday_lag1'] = df['holiday_flag'] * df['lag_1']


# ### 2.5) External features smoothing

# In[12]:


for col in ['temperature', 'fuel_price', 'cpi', 'unemployment']:
    df[f'{col}_rolling4'] = df[col].rolling(window=4).mean()


# ### 2.6) Handle missing values and saving to new file 
# 

# In[14]:


df.dropna(inplace=True)
df.to_csv('../data/walmart_features.csv', index=False)

