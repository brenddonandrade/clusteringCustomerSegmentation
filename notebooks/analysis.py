# %% [markdown]
# ## About the dataset

# %% [markdown]
# Ref: [Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
# 
# 
# This is a transactional data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.
# 

# %% [markdown]
# ### Variables 
# ```
# - InvoiceNo: a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation
# - StockCode: a 5-digit integral number uniquely assigned to each distinct product
# - Description: product name
# - Quantity: the quantities of each product (item) per transaction
# - InvoiceDate: the day and time when each transaction was generated
# - UnitPrice: product price per unit
# - CustomerID: a 5-digit integral number uniquely assigned to each customer
# - Country: the name of the country where each customer resides
# ```
# 

# %% [markdown]
# ## Code

# %% [markdown]
# ### Importing Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
import time

# %% [markdown]
# ### Read and Preprocessing Data

# %%
# current_path = os.getcwd()
# data_path = os.path.join(current_path.replace('/notebooks', ''), 'data')  # ✅ Correto
# file = os.path.join(data_path,'clientes.xlsx')

# df_clients = pd.read_excel(file)

# %%
# backup df
df_save = df_clients.copy()

# %%
df_clients.head()

# %%
df_clients.describe()

# %%
# verify NaN values
df_clients.isna().sum()

# %%
# clean NaN values
df_clients = df_clients.dropna()
df_clients.isna().sum()

# %%
df_clients

# %%
# Normalize data
scaler = StandardScaler()

X = scaler.fit_transform(df_clients[['Quantity', 'UnitPrice']])

# %% [markdown]
# ### Finding the optimal number of clusters (k)

# %%
sil_scores = []
range_n_clusters = list(range(2, 11))

for k in tqdm(range_n_clusters):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    sil_scores.append(score)

# Plot
plt.plot(range_n_clusters, sil_scores, marker='o')
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Escolha do melhor k")
plt.savefig('bestChoice.png', dpi=300)
# plt.show()

melhor_k = range_n_clusters[np.argmax(sil_scores)]
print("Melhor número de clusters:", melhor_k)

# %% [markdown]
# ### Apply the KMeans model with best k

# %%
start = time.time()
kmeans = KMeans(n_clusters=melhor_k, random_state=42)
end_1 = time.time()
df_clients['cluster'] = kmeans.fit_predict(X)
end_2 = time.time()

print(f'First {end_1 - start:.2f} seconds')
print(f'Second {end_2 - end_1:.2f} seconds')


# %% [markdown]
# ### Cluster analysis and visualization

# %%
# Desnormalizar se necessário para interpretação
df_clusters = pd.concat([pd.DataFrame(X, columns=df_clients.columns[:-1]), df_clients['cluster']], axis=1)

# Visualização em 2D com PCA ou apenas 2 variáveis principais
sns.scatterplot(data=df_clusters, x='Renda_Anual', y='Gastos_Cartao', hue='cluster', palette='tab10')
plt.title("Segmentação de Clientes")
# plt.show()
plt.savefig('customerSegmentation.png', dpi=300)

# Estatísticas por cluster
personas = df_clients.groupby('cluster').mean().round(2)
print(personas)


# %% [markdown]
# ### Definition of Personas


