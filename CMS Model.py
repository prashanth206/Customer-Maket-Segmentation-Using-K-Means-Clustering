#!/usr/bin/env python
# coding: utf-8

# In[95]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[96]:


data = pd.read_csv("CC General.csv")


# In[97]:


data.head()


# In[98]:


data.columns


# In[99]:


data.info()


# In[100]:


data["BALANCE"].mean()


# In[101]:


data["BALANCE"].max()


# In[102]:


data["BALANCE"].min()


# In[103]:


data.describe()


# In[104]:


data[data["ONEOFF_PURCHASES"] == data["ONEOFF_PURCHASES"].max()]


# In[105]:


data[data["CASH_ADVANCE"] == data["CASH_ADVANCE"].max()]


# In[106]:


sns.heatmap(data.isnull(), yticklabels = False, cbar = False, cmap = "Blues")


# In[107]:


data.isnull().sum()


# In[108]:


data.loc[(data["MINIMUM_PAYMENTS"].isnull() == True), 'MINIMUM_PAYMENTS'] = data["MINIMUM_PAYMENTS"].mean()


# In[109]:


data.isnull().sum()


# In[110]:


data.loc[(data["CREDIT_LIMIT"].isnull() == True), 'CREDIT_LIMIT'] = data["CREDIT_LIMIT"].mean()


# In[111]:


data.isnull().sum()


# In[112]:


data.duplicated().sum()


# In[113]:


sns.heatmap(data.isnull(), yticklabels = False, cbar = False, cmap = "Blues")


# In[114]:


data.drop("CUST_ID" , axis = 1, inplace = True)


# In[115]:


data.head()


# In[116]:


len(data.columns)


# In[117]:


data.columns


# In[118]:


plt.figure(figsize = (10,50))
for i in range(len(data.columns)):
    plt.subplot(17, 1, i+1)
    sns.distplot(data[data.columns[i]], kde_kws = {'color': "b", "lw": 3, "label": "KDE"}, hist_kws = {"color":'g'})
    plt.title(data.columns[i])
plt.tight_layout()


# In[119]:


correlations = data.corr()
f, ax = plt.subplots(figsize = (20, 10))
sns.heatmap(correlations, annot = True)


# In[120]:


scaler = StandardScaler()
data_s = scaler.fit_transform(data)


# In[121]:


data_s.shape


# In[122]:


data_s


# In[123]:


scores = []
range_values = range(1, 20)
for i in range_values:
    kmean = KMeans(n_clusters = i)
    kmean.fit(data_s)
    scores.append(kmean.inertia_)
plt.plot(scores, 'bx-')


# In[124]:


scores = []
range_values = range(1, 20)
for i in range_values:
    kmean = KMeans(n_clusters = i)
    kmean.fit(data_s[:, :8])
    scores.append(kmean.inertia_)
plt.plot(scores, 'bx-')


# In[125]:


kmeans = KMeans(4)
kmeans.fit(data_s)
label = kmeans.labels_
label


# In[126]:


kmeans.cluster_centers_


# In[127]:


cluster_centers = pd.DataFrame(data=kmeans.cluster_centers_, columns = [data.columns])
cluster_centers


# In[128]:


cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers, columns = [data.columns])
cluster_centers


# In[129]:


data.loc[0]


# In[130]:


test = [[5000, 0.9, 3000, 1000, 500, 500, 0.8, 0.6, 0.3, 0.6, 2, 8, 2000, 2000, 120, 0.111, 10], 
       [5000, 0.9, 3000, 1000, 500, 500, 0.8, 0.6, 0.3, 0.6, 2, 8, 2000, 2000, 120, 0.111, 10],
       [5000, 0.9, 3000, 1000, 500, 500, 0.8, 0.6, 0.3, 0.6, 2, 8, 2000, 2000, 120, 0.111, 10],
       [5000, 0.9, 3000, 1000, 500, 500, 0.8, 0.6, 0.3, 0.6, 2, 8, 2000, 2000, 120, 0.111, 10],
        [40.900749, 0.818182, 95.400000, 0.000000, 95.400000, 0.000000,0.166667, 0.000000, 0.083333, 0.000000, 0.000000, 2.000000,1000.000000,201.802084 , 139.509787, 0.000000, 12.000000],
        [40.900749, 0.818182, 95.400000, 0.000000, 95.400000, 0.000000,0.166667, 0.000000, 0.083333, 0.000000, 0.000000, 2.000000,1000.000000,201.802084 , 139.509787, 0.000000, 12.000000],[40.900749, 0.818182, 95.400000, 0.000000, 95.400000, 0.000000,0.166667, 0.000000, 0.083333, 0.000000, 0.000000, 2.000000,1000.000000,201.802084 , 139.509787, 0.000000, 12.000000]
       ]
df = pd.DataFrame(test, columns = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
       'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE'])

test_s = scaler.fit_transform(df)
kmeans.fit_predict(test_s)


# In[131]:


label.shape


# In[132]:


label.max()


# In[133]:


label.min()


# In[134]:


y = kmeans.fit_predict(data_s)
y


# In[135]:


data_cluster = pd.concat([data, pd.DataFrame({"cluster": label})], axis = 1)
data_cluster


# In[136]:


for i in data.columns:
    plt.figure(figsize = (35, 5))
    for j in range(7):
        plt.subplot(1,7, j+1)
        cluster = data_cluster[data_cluster['cluster'] == j]
        cluster[i].hist(bins = 20)
    plt.show()


# In[137]:


pca = PCA(n_components = 2)
principal_comp = pca.fit_transform(data_s)
principal_comp


# In[138]:


pca_df = pd.DataFrame(data = principal_comp, columns = ['pca1', 'pca2'])
pca_df


# In[139]:


pca_df = pd.concat([pca_df, pd.DataFrame({'cluster':label})], axis = 1)
pca_df


# In[168]:


import matplotlib.patches as mpatches

plt.figure(figsize = (10,10))
ax = sns.scatterplot(x = 'pca1', y = 'pca2', hue = "cluster", data = pca_df,  palette = ['red', 'green', 'blue', 'yellow'])
red_patch = mpatches.Patch(color='red', label='VIP')
green_patch = mpatches.Patch(color='green', label='New Customers')
blue_patch = mpatches.Patch(color='blue', label='Transactors')
yellow_patch = mpatches.Patch(color='yellow', label='Revolvers')
plt.legend(handles=[red_patch, green_patch, blue_patch, yellow_patch])


plt.show()


# # Legend
# 
# #### 0 - VIP
# #### 1 - New Customers(# purchase txns are less)
# #### 2 - Transactors
# #### 3 - Revolvers

# In[169]:


import pickle


# In[171]:


pickle.dump(kmeans, open('model.pkl','wb'))


# In[ ]:




