import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r'/kaggle/input/ham-veri-veri-madenciligi/ham_veri_veri_madenciligi.xlsx'
data = pd.read_excel(file_path)

data = data[['Kira']]
data = data.dropna(subset=['Kira'])
data['Kira'] = data['Kira'].replace({' TL': ''}, regex=True)
data['Kira'] = pd.to_numeric(data['Kira'], errors='coerce')
data = data.dropna(subset=['Kira'])

data.to_excel('islenmis_veri.xlsx', index=False)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['Kira']])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data['kume'] = kmeans.fit_predict(data_scaled)
kume_merkezleri = kmeans.cluster_centers_
kume_merkezleri = scaler.inverse_transform(kume_merkezleri)

output_data = data.copy()
output_data['Küme Merkezleri'] = output_data['kume'].map(lambda x: kume_merkezleri[x][0])
output_data.to_excel('kumelenmis_veri.xlsx', index=False)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x=data.index, y='Kira', hue='kume', palette='viridis', s=100, alpha=0.7)
plt.title('Kira Fiyatlarına Göre Kümeleme Sonuçları')
plt.xlabel('Ev İlanı')
plt.ylabel('Kira Fiyatı')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x=data.index, y='Kira', hue='kume', palette='viridis', s=100, alpha=0.7, marker='o')
plt.scatter(x=kume_merkezleri[:, 0], y=kume_merkezleri[:, 0], c='red', s=300, marker='X', label='Küme Merkezleri')
plt.title('Kira Fiyatlarına Göre Kümeleme Sonuçları ve Küme Merkezleri', fontsize=16)
plt.xlabel('Ev İlanları', fontsize=12)
plt.ylabel('Kira Fiyatı (TL)', fontsize=12)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='kume', y='Kira', data=data, palette='viridis')
plt.title('Küme Başına Kira Fiyatlarının Dağılımı', fontsize=16)
plt.xlabel('Küme', fontsize=12)
plt.ylabel('Kira Fiyatı (TL)', fontsize=12)
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(data=data, x='Kira', hue='kume', multiple='stack', palette='viridis', kde=True)
plt.title('Küme Başına Kira Fiyatlarının Histogramı', fontsize=16)
plt.xlabel('Kira Fiyatı (TL)', fontsize=12)
plt.ylabel('Frekans', fontsize=12)
plt.show()


plt.figure(figsize=(10, 6))
sns.barplot(x=[f'Küme {i+1}' for i in range(kume_merkezleri.shape[0])], y=kume_merkezleri[:, 0])
plt.title('Küme Merkezlerinin Kira Fiyatı (TL) Değerleri', fontsize=16)
plt.xlabel('Küme', fontsize=12)
plt.ylabel('Kira Fiyatı (TL)', fontsize=12)
plt.show()


kume_min_max = data.groupby('kume')['Kira'].agg(['min', 'max']).reset_index()
plt.figure(figsize=(10, 6))
bar_width = 0.3
sns.barplot(x='kume', y='min', data=kume_min_max, color='skyblue', label='Min Kira Fiyatı', width=bar_width, alpha=0.7)
sns.barplot(x='kume', y='max', data=kume_min_max, color='salmon', label='Max Kira Fiyatı', width=bar_width, alpha=0.7, dodge=True)
plt.title('Her Küme İçin Kira Fiyatlarının Min-Max Değerleri', fontsize=16)
plt.xlabel('Küme', fontsize=12)
plt.ylabel('Kira Fiyatı (TL)', fontsize=12)
plt.legend(title='Kira Değeri')
plt.show()

std_devs = data.groupby('kume')['Kira'].std()
plt.figure(figsize=(8, 6))
sns.barplot(x=std_devs.index, y=std_devs.values, palette="viridis")
plt.title('Her Küme İçin Kira Değerlerinin Standart Sapmaları')
plt.xlabel('Küme')
plt.ylabel('Standart Sapma')
plt.xticks(rotation=0)
plt.show()

kume_sayilari = data['kume'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=kume_sayilari.index, y=kume_sayilari.values, palette="viridis")
plt.title('Her Küme İçin Veri Sayıları')
plt.xlabel('Küme')
plt.ylabel('Veri Sayısı')
plt.xticks(rotation=0)
plt.show()
