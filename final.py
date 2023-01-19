# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 12:00:02 2023

@author: Administrator
"""




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from uncertainties import ufloat
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit




font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

plt.rc('font', **font)

def worldbankdatafile(f):
    '''this function will call the worldbank data file in its original format
    and will return the transpose of the file'''
    worldData = f
    trans = worldData.T
    return worldData,trans


def logistics(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f

ePower = pd.read_csv('C://Users//Administrator//Desktop//EnergyPower.csv')
eUse = pd.read_csv('C://Users//Administrator//Desktop//EnergyUse.csv')

Energy_power = worldbankdatafile(ePower)
energy_use = worldbankdatafile(eUse)
ePower

ePower_df = worldbankdatafile(ePower)[0] #calling only original format
eUse_df = worldbankdatafile(eUse)[0]



Countries_ePower = ePower_df[(ePower_df['Indicator Code'] == 'EG.USE.ELEC.KH.PC')]
c_ePower = Countries_ePower.drop(['2021'], axis=1)
c_ePower = c_ePower.drop(['2020'], axis=1)
c_ePower.rename(columns = {'Unnamed: 4':'1989'}, inplace = True)
c_ePower = c_ePower.drop(['1989'], axis=1)


Countries_eUse = eUse_df[(eUse_df['Indicator Code'] == 'EG.USE.PCAP.KG.OE')]
c_eUse = Countries_eUse.drop(['2021'], axis=1)
c_eUse = c_eUse.drop(['2020'], axis=1)
c_eUse.rename(columns = {'Unnamed: 4':'1989'}, inplace = True)
c_eUse = c_eUse.drop(['1989'], axis=1)


c_ePower = c_ePower.round(2)
c_ePower.dropna()


c_eUse = c_eUse.round(2)
c_eUse.dropna()
print(c_eUse)

ePower_eUse1990 = c_ePower[['Country Name','1990']]
ePower_eUse1990.rename(columns = {'1990':'EnergyPower1990'}, inplace = True)
ePower_eUse1990['EnergyUse1990'] = c_eUse[['1990']]
ePower_eUse1990 = ePower_eUse1990.dropna(axis=0)
print(ePower_eUse1990)




# Energy Power of all countries in the year 2014 
ePower_eUse2014 = c_ePower[['Country Name','2014']]
ePower_eUse2014.rename(columns = {'2014':'EnergyPower2014'}, inplace = True)
ePower_eUse2014['EnergyUse2014'] = c_eUse[['2014']]
ePower_eUse2014 = ePower_eUse2014.dropna(axis=0)
print(ePower_eUse2014)


# Normalisation using minmax scaling
scaler = MinMaxScaler()
scaler.fit(ePower_eUse1990[['EnergyPower1990']])
ePower_eUse1990['EnergyPower1990'] = scaler.transform(ePower_eUse1990[['EnergyPower1990']])
scaler.fit(ePower_eUse1990[['EnergyUse1990']])
ePower_eUse1990['EnergyUse1990'] = scaler.transform(ePower_eUse1990[['EnergyUse1990']])
print(ePower_eUse1990)

scaler.fit(ePower_eUse2014[['EnergyPower2014']])
ePower_eUse2014['EnergyPower2014'] = scaler.transform(ePower_eUse2014[['EnergyPower2014']])
scaler.fit(ePower_eUse2014[['EnergyUse2014']])
ePower_eUse2014['EnergyUse2014'] = scaler.transform(ePower_eUse2014[['EnergyUse2014']])
print(ePower_eUse2014)


x = ePower_eUse1990['EnergyPower1990']
y = ePower_eUse1990['EnergyUse1990']
y = y.to_frame()
extcol =  ePower_eUse1990['EnergyPower1990']

extcol = extcol.to_frame()
print(extcol)


data = y.join(extcol)
print(data)



inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method for clustering')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig('elbow1990.png', format='png',dpi=600,
            bbox_inches='tight')
plt.show()



kmeans = KMeans(n_clusters= 3)
 
#prediction the labels of clusters.
pred = kmeans.fit(data)
print(pred)
center = kmeans.cluster_centers_
print(center)


data['cluster'] = kmeans.labels_

# creating a plot which shpws clusters using pyplot
for i in range(5):
    cluster_data = data[data['cluster'] == i]
    plt.scatter(cluster_data['EnergyPower1990'], cluster_data['EnergyUse1990'], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('argicultural_land')
plt.ylabel('cereal_yield')
plt.title('Cluster Membership and Centers')
plt.legend()
plt.show()




