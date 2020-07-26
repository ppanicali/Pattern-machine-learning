# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:01:27 2020

@author: Paolo

estrae massimi e minimi significativi
visualizza la distribuzione oraria e del giorno del mese e di fasce di prezzo per individuare
se esiste una concentrazione di swwing in certi orari o giorni e per visualizzare se esistono
fasce di prezzo su cui i prezzi tendono a girare, questo sarebbe gia una specie di market profile o meglio
una specie di individuazione delle resistenze, tuttavia l obiettivo e vedere se questi teorici supporti resistenze
sono ripetitivi nel tempo

successivamente si passa ad analizzare il tipo di candele presenti sugli swing bull and bear, in modo da paragonarli
e formare le caratteristiche di due classi da contrapporre all andamento normale

dai tratti salienti delle candele negli swing si spera inoltre di individuare dei pattern
"""
import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns
#from datetime import timedelta, date


import numpy as np

#carica dati dal file in df
df = pd.read_csv('C:\\Users\\Paolo\\Documents\\Python_EURUSD_bar//eurusd1h.csv',sep=";",decimal=",", engine='python')
#trasforma la colonna Date in Datetime con il suo formato
df['Date'] =  pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M:%S')
#mostra tipo dati delle colonne di df
df.dtypes
df=df.sort_values(by=['Date'])
#avendo ordinato per data tuttavia l indice generale risulta alterato, percui viene resettato con resetindex
df=df.reset_index(drop=True)

time=df['Date']
df.insert(loc=1, column='Time', value=time)


#df['Date'] = pd.to_datetime(df['Date'],dayfirst=True).dt.strftime('%Y.%m.%d')
#df['Time'] = pd.to_datetime(df['Time']).dt.strftime('%H:%M')




#rows per calcolo max e min
rows_exp=20000
#creo un dataframe provvisorio piccolino
df2=df[0:rows_exp]
dimension_p=24

dimension_n=24

df_L=pd.DataFrame(columns=["Date","Open","High","Low","Close","Body","Shadow","Open_n","High_n","Low_n","Close_n","Body_n","Shadow_n","Open_p","High_p","Low_p","Close_p","Body_p","Shadow_p"])
df_Lfail=pd.DataFrame(columns=["Date","Open","High","Low","Close","Body","Shadow","Open_n","High_n","Low_n","Close_n","Body_n","Shadow_n","Open_p","High_p","Low_p","Close_p","Body_p","Shadow_p"])


for index in range(0,len(df2_B)):
    #esegui solo de index e superiore a dimension
    if index >= dimension_p and index<=rows_exp-dimension_n:
        ## MINIMI VERI
        #dataframe con i dati precedenti al valore attuale 
        df_pastL = df2_B[index-dimension_p:index]
        #estrapolo il minimo da i dimension dati prima
        min_pastL=df_pastL['Low'].min()
        #dataframe con i dati successivi al valore attuale
        df_nextL = df2_B[index+1:index+dimension_n+1]
        #estrapolo il minimo da i dimension dati successivi
        min_nextL=df_nextL['Low'].min()
        #next bar positive boolean
        third_pos=df2_B.at[index,'Close_n']  > df2_B.at[index,'Open_n']
        third_over01 = (df2_B.at[index,'Close_n'] - df2_B.at[index,'Open_n'])> (df2_B.at[index,'Close_n']/100*0.1)
        first_neg = df2_B.at[index,'Close_p']  < df2_B.at[index,'Open_p']
        #se il min attuale e inferiore ai dimension prima ed i dimension dopo lo appendo al dataframe dei minimi
        current_L=df2_B.at[index,'Low']       
        if current_L < min_pastL and current_L < min_nextL and third_over01  :#and third_over01:      
            df_L=df_L.append(df2_B.loc[index])
        ## MINIMI FALLITI    
        if current_L < min_pastL and current_L > min_nextL and third_over01 :# and third_over01 :
            df_Lfail=df_Lfail.append(df2_B.loc[index])    
   
    
  
#MOSTRA RISULTATI DI MASSIMI E MINIMI TROVATI
sns.set(rc={'figure.figsize':(11, 4)})
x_H=df2["High"]
x_L=df2["Low"]

x2_H=df_H["High"]
x2_L=df_L["Low"]

x_H.plot(linewidth=0.7);
x2_H.plot(linewidth=0.7);   

x_L.plot(linewidth=0.7);
x2_L.plot(linewidth=0.7); 
#######################################################





##CALCOLA BODY, WICK E SHADOW DEL DATABASE DEI MASSIMI IN PERCENTUALE
df_H2=df_H.copy()

#aggiungi colonna di ritorni
Bodies=df_H2['Close']-df_H2['Open']
df_H2.insert( loc=7,column='Body', value=Bodies)

#aggiungi colonna di wick
df_H2.insert( loc=8,column='Wick',value=0)
df_H2=df_H2.reset_index(drop=True)

for index in range(0,len(df_H2.index)):
    if df_H2.at[index,'Close'] >= df_H2.at[index,'Open']:
        wick_size=df_H2.at[index,'High'] - df_H2.at[index,'Close'] 
    elif df_H2.at[index,'Open'] > df_H2.at[index,'Close']:
        wick_size=df_H2.at[index,'High'] - df_H2.at[index,'Open']
    df_H2.loc[index,'Wick']=wick_size    

#aggiungi colonna di shadow
df_H2.insert( loc=9,column='Shadow',value=0)
for index in range(0,len(df_H2.index)):
    if df_H2.at[index,'Close'] >= df_H2.at[index,'Open']:
        wick_size=df_H2.at[index,'Open'] - df_H2.at[index,'Low'] 
    elif df_H2.at[index,'Open'] > df_H2.at[index,'Close']:
        wick_size=df_H2.at[index,'Close'] - df_H2.at[index,'Low']
    df_H2.loc[index,'Shadow']=wick_size

df_H2['Body'] = df_H2['Body'] / df_H2['Close']*100   
df_H2['Wick'] = df_H2['Wick'] / df_H2['Close']*100 
df_H2['Shadow'] = df_H2['Shadow'] / df_H2['Close']*100     






#CALCOLA BODY, WICK E SHADOW DEL DATAFRAME DEI MINIMI

df_L2=df_L.copy()

#aggiungi colonna di ritorni
Bodies=df_L2['Close']-df_L2['Open']
df_L2.insert( loc=7,column='Body', value=Bodies)

#aggiungi colonna di wick
df_L2.insert( loc=8,column='Wick',value=0)
df_L2=df_L2.reset_index(drop=True)

for index in range(0,len(df_L2.index)):
    if df_L2.at[index,'Close'] >= df_L2.at[index,'Open']:
        wick_size=df_L2.at[index,'High'] - df_L2.at[index,'Close'] 
    elif df_L2.at[index,'Open'] > df_L2.at[index,'Close']:
        wick_size=df_L2.at[index,'High'] - df_L2.at[index,'Open']
    df_L2.loc[index,'Wick']=wick_size    

#aggiungi colonna di shadow
df_L2.insert( loc=9,column='Shadow',value=0)
for index in range(0,len(df_L2.index)):
    if df_L2.at[index,'Close'] >= df_L2.at[index,'Open']:
        wick_size=df_L2.at[index,'Open'] - df_L2.at[index,'Low'] 
    elif df_L2.at[index,'Open'] > df_L2.at[index,'Close']:
        wick_size=df_L2.at[index,'Close'] - df_L2.at[index,'Low']
    df_L2.loc[index,'Shadow']=wick_size

df_L2['Body'] = df_L2['Body'] / df_L2['Close']*100   
df_L2['Wick'] = df_L2['Wick'] / df_L2['Close']*100 
df_L2['Shadow'] = df_L2['Shadow'] / df_L2['Close']*100     
    





#CALCOLA BODY WICK E SHADOW DEL DATAFRAME CON TUTTO IL CAMPIONE PER COMPARARE CON QUELLI DI MAX E MINIMI

#aggiungi colonna di ritorni
df_serie=df2.copy()
Bodies=df_serie['Close']-df_serie['Open']
df_serie.insert( loc=7,column='Body', value=Bodies)

#aggiungi colonna di wick
df_serie.insert( loc=8,column='Wick',value=0)
df_serie=df_serie.reset_index(drop=True)

for index in range(0,len(df_serie.index)):
    if df_serie.at[index,'Close'] >= df_serie.at[index,'Open']:
        wick_size=df_serie.at[index,'High'] - df_serie.at[index,'Close'] 
    elif df_serie.at[index,'Open'] > df_serie.at[index,'Close']:
        wick_size=df_serie.at[index,'High'] - df_serie.at[index,'Open']
    df_serie.loc[index,'Wick']=wick_size    

#aggiungi colonna di shadow
df_serie.insert( loc=9,column='Shadow',value=0)
for index in range(0,len(df_serie.index)):
    if df_serie.at[index,'Close'] >= df_serie.at[index,'Open']:
        wick_size=df_serie.at[index,'Open'] - df_serie.at[index,'Low'] 
    elif df_serie.at[index,'Open'] > df_serie.at[index,'Close']:
        wick_size=df_serie.at[index,'Close'] - df_serie.at[index,'Low']
    df_serie.loc[index,'Shadow']=wick_size    
 

df_serie['Body'] = df_serie['Body'] / df_serie['Close']*100   
df_serie['Wick'] = df_serie['Wick'] / df_serie['Close']*100 
df_serie['Shadow'] = df_serie['Shadow'] / df_serie['Close']*100 

###################################################################################################3
## GRAFICI DI COMPARAZIONE DELLA DISTRIBUZIONE DELLE CARATTERISTICHE BODY, WICK E SHADOW

#DISTRIBUZIONE DEI BODI NELLE CANDELE DI MINIMO
bodyGraph=df_L2['Body'].plot.hist(title='body size',bins=50,alpha=0.5,color='blue',normed=True)   
bodyGraph2=df_serie['Body'].plot.hist(title='body size',bins=50,alpha=0.2,color='red', normed=True) 




bodyGraph=df_H2['Shadow'].plot.hist(title='body size',bins=300,alpha=0.5,color='blue',normed=True)   
bodyGraph2=df_serie['Body'].plot.hist(title='body size',bins=300,alpha=0.2,color='red', normed=True)  

Scatter_Wick_body=df_H2.plot.scatter(x='Body',y='Wick',alpha=0.5)
Scatter_Wick_body=df_serie.plot.scatter(x='Body',y='Wick',alpha=0.5)


#dataframe di massimi e minimi con indice resettato non per visualizzazione grafica
#df_H=df_H.reset_index(drop=True)
#df_L=df_L.reset_index(drop=True)  
    
# #dataframe di frequenza highs
# df_orariH = df_H['Time'].dt.hour
# print (df_orariH)

# #dataframe di frequenza lows
# df_orariL = df_L['Time'].dt.hour
# print (df_orariL)

# #distribuzione oraria h
# ax = df_orariH.plot.hist(title='distribuzione oraria dei massimi',bins=24,alpha=0.5,color='red')

# #distribuzione oraria l
# bx = df_orariL.plot.hist(title='distribuzione oraria dei minimi',bins=24,alpha=0.5,color='blue', subplots=True)


   
# #dataframe massimi
# df_massimi = df_H['High']


# #dataframe di frequenza lows
# df_minimi = df_L['Low']


# #distribuzione oraria h
# cx = df_massimi.plot.hist(title='distribuzione valori massimi',bins=1000,alpha=0.5,color='red')

# #distribuzione oraria l
# dx = df_minimi.plot.hist(title='distribuzione valori minimi',bins=1000,alpha=0.5,color='blue', subplots=True)

 
# #dataframe massimi
# df_volumiH = df_H['Volume']


# #dataframe di frequenza lows
# df_volumiL = df_H['Volume']


# #distribuzione oraria h
# fx = df_volumiL.plot.hist(title='distribuzione volumi',bins=500,alpha=0.5,color='red')

#x=df_L.groupby(df["Time"].dt.hour().count().plot((kind="bar"))
             
#print(df["Time"].dt.hour())             
