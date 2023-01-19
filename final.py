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
for i in range(3):
    cluster_data = data[data['cluster'] == i]
    plt.scatter(cluster_data['EnergyPower1990'], cluster_data['EnergyUse1990'], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('EnergyPower1990')
plt.ylabel('EnergyUse1990')
plt.title('Cluster Membership and Centers')
plt.legend()
plt.show()

x = ePower_eUse2014['EnergyPower2014']
y = ePower_eUse2014['EnergyUse2014']
y = y.to_frame()
extcol =  ePower_eUse2014['EnergyPower2014']

extcol = extcol.to_frame()
print(extcol)


data = y.join(extcol)
print(data)

kmeans = KMeans(n_clusters= 3)
 
#prediction the labels of clusters.
pred = kmeans.fit(data)
print(pred)
center = kmeans.cluster_centers_
print(center)


data['cluster'] = kmeans.labels_

# creating a plot which shpws clusters using pyplot
for i in range(3):
    cluster_data = data[data['cluster'] == i]
    plt.scatter(cluster_data['EnergyPower2014'], cluster_data['EnergyUse2014'], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('EnergyPower2014')
plt.ylabel('EnergyUse2014')
plt.title('Cluster Membership and Centers')
plt.legend()
plt.show()



ePowerofChina = ePower[ ePower['Country Name']=='China']
del  ePowerofChina['Country Name']
del  ePowerofChina['Indicator Name']
del  ePowerofChina['Country Code']
del  ePowerofChina['Indicator Code']
del  ePowerofChina['2020']
del  ePowerofChina['2021']
print(type( ePowerofChina))
gdparr =  ePowerofChina.values.tolist()
print(gdparr)


year = []
for i in range(31):
    year.append(1989+i)


x = []
for i in range(31):
    x.append(i)

dfChina = pd.DataFrame(columns = ['years','Energy Power'],
                    index = x )
print(dfChina.loc[0][0])
for i in range(31):
    dfChina.loc[i] = [year[i],gdparr[0][i]]
    dfChina = dfChina.dropna(axis=0)
    dfChina['years'] = dfChina['years'].astype(float)
print(dfChina)


parameter, curve = curve_fit(logistics, dfChina["years"],
                        dfChina["Energy Power"])
print("Fit parameter", parameter)
dfChina["log"] = logistics(dfChina["years"], *parameter)

plt.figure()
plt.plot(dfChina["years"], dfChina["Energy Power"], label="data")
plt.plot(dfChina["years"], dfChina["log"], label="fit")

plt.legend()
plt.title("First fit attempt")
plt.xlabel("year")
plt.ylabel("Energy Power of China")
plt.show()
print()


# estimating turning year: 2000
parameter = [66, 0.02, 2000]
dfChina["log"] = logistics(dfChina["years"], *parameter)

plt.figure()
plt.plot(dfChina["years"], dfChina["Energy Power"], label="data")
plt.plot(dfChina["years"],dfChina["log"], label="fit")

plt.legend()
plt.xlabel("years")
plt.ylabel("Energy Power of China")
plt.title("Improved start value")
plt.show()


parameter, curve = curve_fit(logistics, dfChina["years"],  dfChina["Energy Power"],
                         p0 = [66, 0.02, 2000])
print("Fit parameter", parameter)
dfChina["log"] = logistics(dfChina["years"], *parameter)
sigma = np.sqrt(np.diagonal(curve))
a = ufloat(parameter[0], sigma[0])
b = ufloat(parameter[1], sigma[1])
x_pred = np.linspace(1995, 2025, 20)
text_res = "Best fit parameters:\na = {}\nb = {}".format(a, b)
print(text_res)

plt.figure()
plt.plot(dfChina["years"],dfChina["Energy Power"], label="data")
plt.plot(x_pred, logistics(x_pred, *parameter), 'red', label="fit")
bound_upper = logistics(x_pred, *(parameter + sigma))
bound_lower = logistics(x_pred, *(parameter - sigma))
# plotting confidence intervals
plt.fill_between(x_pred, bound_lower, bound_upper,
                 color='black', alpha=0.15, label="Confidence")
plt.legend()
plt.title("Final logistics function")
plt.xlabel("years")
plt.ylabel("Energy Power of China")
plt.show()
